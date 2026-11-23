"""FastAPI: endpoints, lifespan, xác thực, giới hạn body, định dạng lỗi.

Chạy bằng factory (không tạo app lúc import):

    uvicorn sam3_service.app:create_app --factory --host 0.0.0.0 --port 8000

Chỉ chạy MỘT worker Uvicorn: mỗi worker nhân bản model lên GPU.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import os
import re
import signal
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Callable

from anyio import CapacityLimiter, to_thread
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import SCHEMA_VERSION, __version__
from .config import Settings, get_settings
from .images import (
    build_instance,
    decode_image,
    validate_concept_params,
    validate_boxes,
    validate_points_and_box,
)
from .inference import EngineUnavailable, InferenceEngine, RequestTimeout
from .model import RawInstance, SegmentationModel
from .schemas import (
    ApiError,
    BatchInfo,
    ErrorBody,
    ErrorResponse,
    ImageInfo,
    InstanceRequest,
    PredictRequest,
    PredictResponse,
    Timing,
)

log = logging.getLogger("sam3_service.app")
_RID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")
_DEMO_HTML = Path(__file__).parent / "static" / "demo.html"


@dataclass
class Context:
    settings: Settings
    engine: InferenceEngine
    model: SegmentationModel
    limiter: CapacityLimiter
    ready: bool = False
    load_error: str | None = None
    tasks: list = field(default_factory=list)


def _error_response(
    status: int,
    code: str,
    message: str,
    request_id: str | None,
    headers: dict[str, str] | None = None,
    details: list[dict] | None = None,
) -> JSONResponse:
    body = ErrorResponse(
        error=ErrorBody(code=code, message=message, request_id=request_id, details=details)
    )
    return JSONResponse(status_code=status, content=body.model_dump(exclude_none=True), headers=headers)


class _BodyTooLarge(Exception):
    pass


class GatewayMiddleware:
    """ASGI thuần: request id, xác thực Bearer cho /v1/*, giới hạn body.

    Xác thực và kiểm Content-Length diễn ra TRƯỚC khi đọc body, để client chưa
    xác thực không bắt server nhận hàng chục MB.
    """

    def __init__(self, app, settings: Settings) -> None:
        self.app = app
        self.cfg = settings
        key = settings.api_key.get_secret_value() if settings.api_key else ""
        self._key = key.encode()

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
        rid = headers.get("x-request-id", "")
        if not _RID_RE.match(rid):
            rid = uuid.uuid4().hex
        state = scope.setdefault("state", {})
        state["request_id"] = rid
        started = False

        too_large = False
        replaced_body_sent = False

        async def send_wrapper(message):
            nonlocal started, replaced_body_sent
            if too_large:
                # FastAPI bọc lỗi đọc body thành 400; ghi đè thành 413 đúng hợp đồng.
                if message["type"] == "http.response.start":
                    started = True
                    payload = _error_response(
                        413, "payload_too_large", "Request quá lớn.", state["request_id"]
                    ).body
                    await send(
                        {
                            "type": "http.response.start",
                            "status": 413,
                            "headers": [
                                (b"content-type", b"application/json"),
                                (b"content-length", str(len(payload)).encode()),
                                (b"x-request-id", state["request_id"].encode()),
                            ],
                        }
                    )
                    await send({"type": "http.response.body", "body": payload})
                    replaced_body_sent = True
                return  # bỏ qua mọi phần còn lại của phản hồi gốc
            if message["type"] == "http.response.start":
                started = True
                hdrs = list(message.get("headers", []))
                hdrs.append((b"x-request-id", state["request_id"].encode()))
                message = {**message, "headers": hdrs}
            await send(message)

        async def reject(resp: JSONResponse) -> None:
            await resp(scope, receive, send_wrapper)

        path = scope["path"]
        if path.startswith("/v1/") and not self.cfg.auth_disabled:
            if not self._authorized(headers.get("authorization", "")):
                await reject(
                    _error_response(
                        401,
                        "unauthorized",
                        "Thiếu hoặc sai API key.",
                        rid,
                        headers={"WWW-Authenticate": "Bearer"},
                    )
                )
                return

        if scope["method"] in ("POST", "PUT", "PATCH"):
            cl = headers.get("content-length")
            if cl is not None:
                try:
                    too_big = int(cl) > self.cfg.max_body_bytes
                except ValueError:
                    too_big = True
                if too_big:
                    await reject(_error_response(413, "payload_too_large", "Request quá lớn.", rid))
                    return

        received = 0

        async def receive_wrapper():
            nonlocal received, too_large
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.cfg.max_body_bytes:
                    too_large = True
                    raise _BodyTooLarge
            return message

        try:
            await self.app(scope, receive_wrapper, send_wrapper)
        except _BodyTooLarge:
            if not started:
                await reject(_error_response(413, "payload_too_large", "Request quá lớn.", state["request_id"]))
        else:
            if too_large and not replaced_body_sent and not started:
                await reject(_error_response(413, "payload_too_large", "Request quá lớn.", state["request_id"]))

    def _authorized(self, header: str) -> bool:
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not token or not self._key:
            return False
        return hmac.compare_digest(token.strip().encode(), self._key)


def _default_terminate() -> None:
    os.kill(os.getpid(), signal.SIGTERM)


def create_app(
    settings: Settings | None = None,
    *,
    model: SegmentationModel | None = None,
    terminate: Callable[[], None] | None = None,
) -> FastAPI:
    """``model``: chỉ để test (đưa đối tượng giả vào). Production để None -> model thật."""
    cfg = settings or get_settings()
    logging.basicConfig(
        level=cfg.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    stop_process = terminate or _default_terminate

    def on_fatal(exc: ApiError) -> None:
        if cfg.exit_on_fatal:
            stop_process()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if cfg.auth_disabled:
            log.warning("XÁC THỰC ĐANG TẮT (SAM3_AUTH_DISABLED=true). Không dùng ở production.")
        if model is not None:
            mdl = model
        else:
            from .model import Sam3Model  # noqa: PLC0415 - tránh import Torch khi test

            mdl = Sam3Model(cfg)
        engine = InferenceEngine(max_queue=cfg.queue_size, on_fatal=on_fatal)
        ctx = Context(
            settings=cfg,
            engine=engine,
            model=mdl,
            limiter=CapacityLimiter(cfg.decode_concurrency),
        )
        app.state.ctx = ctx
        engine.start()

        async def load() -> None:
            try:
                # Nạp model trong chính luồng suy luận để mọi thao tác với
                # model/CUDA nằm cùng một luồng.
                await engine.run(mdl.load, timeout_s=None, request_id="load")
                ctx.ready = True
                log.info("Dịch vụ sẵn sàng")
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                ctx.load_error = f"{type(exc).__name__}: {getattr(exc, 'message', exc)}"
                log.critical("Nạp model thất bại: %s", ctx.load_error)
                on_fatal(ApiError(500, "load_failed", ctx.load_error, fatal=True))

        # Process phản hồi /healthz ngay; /readyz chỉ 200 sau khi nạp xong.
        ctx.tasks.append(asyncio.create_task(load()))
        try:
            yield
        finally:
            for t in ctx.tasks:
                t.cancel()
            await to_thread.run_sync(engine.stop)
            await to_thread.run_sync(mdl.close)

    app = FastAPI(
        title="SAM3 service",
        version=__version__,
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url="/v1/openapi.json",  # nằm dưới /v1/ nên cần xác thực
    )
    app.add_middleware(GatewayMiddleware, settings=cfg)

    # ------------------------------------------------------------- lỗi
    def rid_of(request: Request) -> str | None:
        return getattr(request.state, "request_id", None)

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError):
        level = logging.ERROR if exc.status_code >= 500 else logging.WARNING
        log.log(level, "request_id=%s status=%s code=%s msg=%s", rid_of(request), exc.status_code, exc.code, exc.message)
        return _error_response(exc.status_code, exc.code, exc.message, rid_of(request), exc.headers)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        # Không đưa "input" vào phản hồi: có thể chứa cả chuỗi base64 khổng lồ.
        details = [
            {"loc": [str(p) for p in e.get("loc", ())], "msg": str(e.get("msg", "")), "type": str(e.get("type", ""))}
            for e in exc.errors()
        ]
        return _error_response(422, "invalid_request", "Request không hợp lệ.", rid_of(request), details=details)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException):
        return _error_response(exc.status_code, "http_error", str(exc.detail), rid_of(request), exc.headers)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        log.exception("request_id=%s lỗi không xử lý", rid_of(request))
        return _error_response(500, "internal_error", "Lỗi nội bộ.", rid_of(request))

    # ---------------------------------------------------------- endpoints
    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/", include_in_schema=False)
    async def demo_home():
        return RedirectResponse(url="/demo", status_code=307)

    @app.get("/demo", response_class=HTMLResponse, include_in_schema=False)
    async def demo_page():
        return HTMLResponse(_DEMO_HTML.read_text(encoding="utf-8"))

    @app.get("/readyz")
    async def readyz(request: Request):
        ctx: Context = request.app.state.ctx
        if ctx.engine.fatal is not None:
            return JSONResponse({"status": "fatal"}, status_code=503)
        if ctx.load_error is not None:
            return JSONResponse({"status": "failed"}, status_code=503)
        if not ctx.ready:
            return JSONResponse({"status": "loading"}, status_code=503)
        return {"status": "ready"}

    @app.post("/v1/predict", response_model=PredictResponse, response_model_exclude_unset=True)
    async def predict(request: Request, body: PredictRequest):
        ctx: Context = request.app.state.ctx
        c = ctx.settings
        t0 = time.perf_counter()
        rid = body.request_id or request.state.request_id
        request.state.request_id = rid  # request_id trong body ưu tiên hơn header

        if not ctx.ready or ctx.engine.fatal is not None:
            raise EngineUnavailable()

        # Decode/kiểm tra ở threadpool: không chặn event loop, không chiếm luồng GPU.
        image = await to_thread.run_sync(decode_image, body.image_base64, c, limiter=ctx.limiter)
        width, height = image.size

        if isinstance(body, InstanceRequest):
            if body.boxes is not None:
                boxes = validate_boxes(body.boxes, width, height)
                fn = partial(ctx.model.predict_instances, image, boxes)
                n_prompts = f"boxes={len(boxes)}"
            else:
                pts, bx = validate_points_and_box(body.points, body.box, width, height, c)
                fn = partial(ctx.model.predict_instance, image, pts, bx)
                n_prompts = f"points={len(pts)} box={bx is not None}"
        else:
            text, thr, n = validate_concept_params(body.text, body.threshold, body.max_instances, c)
            fn = partial(ctx.model.predict_concept, image, text, thr, n)
            n_prompts = f"threshold={thr} max_instances={n}"

        remaining = c.request_timeout_s - (time.perf_counter() - t0)
        if remaining <= 0:
            raise RequestTimeout("Hết thời gian trước khi xếp hàng đợi.")
        raw, timing = await ctx.engine.run(fn, timeout_s=remaining, request_id=rid)

        batch = {}
        if isinstance(body, InstanceRequest) and body.boxes is not None:
            count = len(body.boxes)
            ids = [item.prompt_id for item in raw]
            if (len(ids) != count or any(type(value) is not int for value in ids)
                    or set(ids) != set(range(count))):
                raise RuntimeError("Model returned incomplete or invalid batch prompt IDs")
        instances = await to_thread.run_sync(_encode, raw, limiter=ctx.limiter)
        if isinstance(body, InstanceRequest) and body.boxes is not None:
            present = {item.prompt_id for item in instances}
            batch["batch"] = BatchInfo(
                count=len(body.boxes),
                empty_prompt_ids=[i for i in range(len(body.boxes)) if i not in present],
            )
        total_ms = (time.perf_counter() - t0) * 1000.0
        log.info(
            "request_id=%s mode=%s image=%dx%d %s instances=%d queue_ms=%.0f infer_ms=%.0f total_ms=%.0f",
            rid, body.mode, width, height, n_prompts, len(instances),
            timing.queue_ms, timing.run_ms, total_ms,
        )
        return PredictResponse(
            schema_version=SCHEMA_VERSION,
            request_id=rid,
            mode=body.mode,
            image=ImageInfo(width=width, height=height),
            model=ctx.model.meta,
            timing=Timing(queue_ms=timing.queue_ms, inference_ms=timing.run_ms, total_ms=total_ms),
            instances=instances,
            **batch,
        )

    return app


def _encode(raw: list[RawInstance]):
    out = []
    for r in raw:
        inst = build_instance(r.mask, r.score, r.score_type, r.prompt_id)
        if inst is not None:  # mask rỗng bị loại
            out.append(inst)
    return out
