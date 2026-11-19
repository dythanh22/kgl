"""Model serving: manage the VM-side server, run inference through Jupyter
messaging, and expose a local OpenAI-compatible shim (no public tunnel)."""
from __future__ import annotations

import base64
import hmac
import ipaddress
import json
import logging
import posixpath
import select
import socket
import threading
import time
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable

from .errors import ConnectionLost, KernelGone, KglError, ProtocolError, ServeError
from .remote import (CANCEL_INFER, INFER, SERVE_HEALTH, SERVE_LOGS, SERVE_START, SERVE_STOP,
                     MarkerParser, new_marker, render)
from .remote_server import MAX_REQUEST_BYTES, chat_args, image_bytes, parse_device_map
from .runtime import Runtime
from .files import upload_file

log = logging.getLogger("kgl.serving")
SERVER_SCRIPT = Path(__file__).with_name("remote_server.py")
MAX_CHAT_BYTES = MAX_REQUEST_BYTES


class ServeManager:
    def __init__(self, rt: Runtime, port: int = 8000) -> None:
        self.rt, self.port = rt, port
        self.dir = posixpath.join(rt.settings.workdir, ".kgl")

    def start(self, model: str, backend: str = "transformers", dtype: str = "float16",
              wait: bool = True, wait_timeout: float = 900.0,
              progress: Callable[[str], None] | None = None, require_gpu: bool = False,
              device_map: str | None = None, prefill_chunk_size: int | None = None,
              predictor: str | None = None) -> dict:
        say = progress or (lambda s: None)
        if require_gpu and backend == "echo":
            raise ServeError("--require-gpu requires a model backend")
        if predictor is not None and (backend != "image-classification" or not posixpath.isabs(predictor)):
            raise ServeError("--predictor requires the image-classification backend and an absolute VM path")
        if device_map is not None:
            if backend != "transformers":
                raise ServeError("--device-map requires the transformers backend")
            try:
                parse_device_map(device_map)
            except ValueError as exc:
                raise ServeError(str(exc)) from exc
        if prefill_chunk_size is not None and (prefill_chunk_size <= 0 or backend != "transformers"):
            raise ServeError("--prefill-chunk-size requires a positive integer and the transformers backend")
        # Upload the server script (small; verified by sha256).
        upload_file(self.rt, SERVER_SCRIPT, posixpath.join(self.dir, "serve.py"))
        args = ["--model", model, "--backend", backend, "--dtype", dtype,
                "--port", str(self.port)]
        if require_gpu:
            args.append("--require-gpu")
        if predictor is not None:
            args.extend(["--predictor", predictor])
        if device_map:
            args.extend(["--device-map", device_map])
        if prefill_chunk_size is not None:
            args.extend(["--prefill-chunk-size", str(prefill_chunk_size)])
        r = self.rt.call(SERVE_START, DIR=self.dir, ARGS=args)
        say("server process already running" if r.get("already_running")
            else f"server process started (pid {r['pid']})")
        if not wait:
            return r
        try:
            result = self.wait_ready(wait_timeout, say, r["instance"], backend,
                                     model if backend != "echo" else "echo")
        except (KglError, TimeoutError, KeyboardInterrupt):
            if r.get("started"):
                try:
                    self.stop(r["instance"])
                except (KglError, TimeoutError) as exc:
                    log.warning("server cleanup was not confirmed: %s", exc)
            raise
        result["started"] = bool(r.get("started"))
        return result

    def health(self, timeout: float = 10.0) -> dict:
        control = self.rt.control_runtime()
        try:
            return (control or self.rt).call(SERVE_HEALTH, timeout=timeout, idempotent=True,
                                             DIR=self.dir, PORT=self.port)
        finally:
            if control:
                control.close()

    def wait_ready(self, timeout: float, say, instance: str, backend: str, model: str) -> dict:
        t0, last = time.monotonic(), None
        while time.monotonic() - t0 < timeout:
            h = self.health(timeout=max(0.01, timeout - (time.monotonic() - t0)))
            state = h["state"]
            if state != last:
                say(f"server state: {state}")
                last = state
            if state == "ready":
                health = h["health"]
                if (health.get("instance"), health.get("backend"), health.get("model")) != (instance, backend, model):
                    raise ServeError("unexpected server identity, backend or model on requested port")
                return h
            if h.get("health", {}).get("status") == "error":
                raise ServeError("model failed to load: " + str(h["health"].get("error"))
                                 + "\nsee `kgl serve logs`")
            if state == "down" and time.monotonic() - t0 > 30:
                raise ServeError("server not responding:\n" + "\n".join(h.get("log_tail", [])))
            time.sleep(min(1.0, max(0, timeout - (time.monotonic() - t0))))
        raise ServeError(f"timed out after {timeout:.0f}s waiting for model; see `kgl serve logs`")

    def stop(self, instance: str | None = None) -> dict:
        control = self.rt.control_runtime() if instance else None
        try:
            return (control or self.rt).call(SERVE_STOP, timeout=20, DIR=self.dir, INSTANCE=instance)
        finally:
            if control:
                control.close()

    def logs(self, n: int = 50) -> list[str]:
        return self.rt.call(SERVE_LOGS, DIR=self.dir, N=n).get("lines", [])


@dataclass
class InferResult:
    status: int = 0
    body: dict | None = None
    text: str = ""
    error: str | None = None
    latency_s: float = 0.0
    ttft_s: float | None = None  # time to first streamed token


def infer_chat(rt: Runtime, port: int, payload: dict,
               on_chunk: Callable[[dict], None] | None = None,
               timeout: float = 600.0, *, expected_instance: str | None = None,
               check_client=None, on_cancel: Callable[[bool], None] | None = None,
               predict: bool = False) -> InferResult:
    """Send an inference request to the VM server via the kernel."""
    res = InferResult()
    t0 = time.monotonic()
    pieces: list[str] = []
    instance = expected_instance or (rt.session.serve or {}).get("instance")
    if predict and not instance:
        raise ServeError("server instance is missing; restart the model server")
    request_id = uuid.uuid4().hex
    got_status = False
    submitted = False

    def mark_submitted() -> None:
        nonlocal submitted
        submitted = True

    def handle(m: dict) -> None:
        nonlocal got_status
        if "chunk" in m:
            if m["chunk"] == "[DONE]":
                return
            try:
                obj = json.loads(m["chunk"])
            except ValueError:
                return
            delta = (obj.get("choices") or [{}])[0].get("delta", {}).get("content")
            if delta:
                if res.ttft_s is None:
                    res.ttft_s = time.monotonic() - t0
                pieces.append(delta)
            if on_chunk:
                on_chunk(obj)
        if "status" in m:
            got_status = True
            res.status = m["status"]
            res.body = m.get("body")
            res.error = m.get("error")

    data = json.dumps(payload, ensure_ascii=False).encode()
    if len(data) > MAX_CHAT_BYTES:
        raise ServeError("inference request exceeds 512 KiB")
    marker = new_marker()
    parser = MarkerParser(handle, marker=marker)
    code = render(INFER, marker=marker, PAYLOAD=base64.b64encode(data).decode(),
                  PORT=port, TIMEOUT=timeout,
                  OPTIONS={"instance": instance, "request_id": request_id, "predict": predict})
    try:
        rt.run(code, lambda name, text: parser.feed(text) if name == "stdout" else None,
               total_timeout=timeout, check=True, check_client=check_client, on_submit=mark_submitted)
    except (KeyboardInterrupt, TimeoutError, ConnectionLost, OSError):
        if not submitted or got_status:
            raise
        control = rt.control_runtime() if instance else None
        confirmed = False
        try:
            if control:
                confirmed = control.call(CANCEL_INFER, timeout=4, PORT=port, INSTANCE=instance,
                                         REQUEST_ID=request_id).get("cancelled") is True
        except (KglError, TimeoutError):
            pass
        finally:
            if control:
                control.close()
        if on_cancel:
            on_cancel(confirmed)
        elif confirmed:
            log.info("remote cancellation confirmed")
        else:
            log.warning("remote cancellation unconfirmed")
        raise
    if not got_status:
        raise ProtocolError("inference ended without a result marker")
    res.latency_s = time.monotonic() - t0
    if predict:
        return res
    if res.body:
        res.text = (res.body.get("choices") or [{}])[0].get("message", {}).get("content", "")
    else:
        res.text = "".join(pieces)
    return res


def make_proxy_handler(rt: Runtime, port: int, model_name: str, api_key: str | None,
                       listen_host: str = "127.0.0.1", demo_page: bytes | None = None):
    last_health = None
    admission = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        timeout = 10

        def log_message(self, fmt, *args):
            log.info("proxy: " + fmt, *args)

        def _json(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _err(self, code: int, msg: str, typ: str = "kgl_error") -> None:
            self._json(code, {"error": {"message": msg, "type": typ}})

        def _authorized(self) -> bool:
            if not api_key:
                return True
            if hmac.compare_digest(self.headers.get("Authorization", "").encode(),
                                   f"Bearer {api_key}".encode()):
                return True
            self._err(401, "invalid api key", "invalid_request_error")
            return False

        def _valid_host(self) -> bool:
            values = self.headers.get_all("Host", [])
            if len(values) == 1:
                host = values[0].lower()
                bound_ip, bound_port = self.connection.getsockname()[:2]
                names = {listen_host.lower(), bound_ip}
                if ipaddress.ip_address(bound_ip).is_loopback:
                    names.add("localhost")
                names -= {"0.0.0.0", "::"}
                allowed = {f"{name}:{bound_port}" for name in names}
                if bound_port == 80:
                    allowed.update(names)
                if host in allowed:
                    origins = self.headers.get_all("Origin", [])
                    if not origins or origins == [f"http://{host}"]:
                        return True
                    self._err(403, "untrusted Origin", "invalid_request_error")
                    return False
            self._err(400, "untrusted Host", "invalid_request_error")
            return False

        def do_GET(self):
            nonlocal last_health
            if not self._valid_host():
                return
            if self.path == "/health":
                return self._json(200, {"status": "ok"})
            if self.path == "/" and demo_page is not None:
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(demo_page)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Frame-Options", "DENY")
                self.end_headers()
                self.wfile.write(demo_page)
                return
            if not self._authorized():
                return
            if self.path == "/ready":
                deadline = time.monotonic() + 5
                try:
                    kernel = rt.client.get_kernel(rt.session.kernel_id, timeout=5)
                except (KglError, TimeoutError) as exc:
                    return self._err(503, str(exc), "kernel_rest_unavailable")
                if kernel.get("execution_state") == "busy":
                    expected = rt.session.serve or {}
                    result = {"state": "busy", "instance": expected.get("instance")}
                    if last_health is not None:
                        checked_at, health = last_health
                        result.update(last_health=health, health_age_s=time.monotonic() - checked_at)
                    return self._json(503, result)
                try:
                    result = ServeManager(rt, port).health(timeout=max(.01, deadline - time.monotonic()))
                except (KglError, TimeoutError) as exc:
                    return self._err(503, str(exc), "runtime_unavailable")
                health = result.get("health", {})
                expected = rt.session.serve or {}
                if health:
                    last_health = (time.monotonic(), health)
                ready = (result["state"] == "ready" and health.get("model") == model_name
                         and health.get("backend") == expected.get("backend", health.get("backend"))
                         and health.get("instance") == expected.get("instance", health.get("instance"))
                         and (health.get("backend") in ("echo", "image-classification")
                              or str(health.get("device")).startswith("cuda")))
                return self._json(200 if ready else 503, result)
            if self.path in ("/v1/models", "/models"):
                return self._json(200, {"object": "list",
                                        "data": [{"id": model_name, "object": "model"}]})
            self._err(404, "not found")

        def do_POST(self):
            if not self._valid_host() or not self._authorized():
                return
            predict = self.path == "/predict"
            image_backend = (rt.session.serve or {}).get("backend") == "image-classification"
            if (not predict and self.path not in ("/v1/chat/completions", "/chat/completions")
                    or predict != image_backend):
                return self._err(404, "route unavailable for the active backend")
            if self.headers.get_content_type() != "application/json":
                return self._err(415, "Content-Type must be application/json", "invalid_request_error")
            try:
                lengths = self.headers.get_all("Content-Length", [])
                if len(lengths) != 1 or self.headers.get("Transfer-Encoding"):
                    raise ValueError("a single Content-Length is required")
                length = int(lengths[0])
                if length > MAX_CHAT_BYTES:
                    return self._err(413, "request exceeds 512 KiB", "invalid_request_error")
                if length <= 0:
                    raise ValueError("request cannot be empty")
                req = json.loads(self.rfile.read(length))
                if predict:
                    image_bytes(req)
                else:
                    chat_args(req, model_name)
            except (ValueError, RecursionError) as exc:
                return self._err(400, str(exc), "invalid_request_error")
            except TimeoutError:
                return self._err(408, "timed out reading request body", "invalid_request_error")
            stream = not predict and req.get("stream") is True
            started = False

            def on_chunk(obj: dict) -> None:
                nonlocal started
                if not started:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    started = True
                self.wfile.write(("data: " + json.dumps(obj) + "\n\n").encode())
                self.wfile.flush()

            def stream_error(message: str):
                log.warning("incomplete stream: %s", message)
                event = {"error": {"message": message, "type": "incomplete_stream"}}
                try:
                    self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
                    self.wfile.flush()
                except OSError:
                    pass

            def check_client():
                readable, _, _ = select.select([self.connection], [], [], 0)
                if readable and not self.connection.recv(1, socket.MSG_PEEK):
                    raise BrokenPipeError("HTTP client disconnected")

            try:
                if not predict:
                    req.setdefault("model", model_name)
                if not admission.acquire(blocking=False):
                    return self._err(503, "inference already running", "worker_busy")
                try:
                    options = {"predict": True} if predict else {}
                    r = infer_chat(rt, port, req, on_chunk if stream else None, check_client=check_client, **options)
                finally:
                    admission.release()
            except (ConnectionLost, KernelGone) as e:
                if started:
                    return stream_error(str(e))
                return self._err(503, f"runtime unavailable: {e}", "kgl_connection_lost")
            except (BrokenPipeError, ConnectionResetError):
                return
            except (KglError, TimeoutError) as e:
                if started:
                    return stream_error(str(e))
                return self._err(504 if isinstance(e, TimeoutError) else 502, str(e))
            if stream and started:
                if r.status != 200:
                    return stream_error(r.error or "incomplete upstream response")
                try:
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except BrokenPipeError:
                    pass
                return
            if r.status == 200 and r.body is not None:
                return self._json(200, r.body)
            status = r.status if r.status in (400, 409, 503, 504) else 502
            self._err(status, r.error or "upstream error")

    return Handler


def make_proxy_server(rt: Runtime, port: int, model_name: str, host: str = "127.0.0.1",
                      listen_port: int = 8001, api_key: str | None = None,
                      *, demo: bool = False) -> ThreadingHTTPServer:
    if host.lower() == "localhost":
        host = "127.0.0.1"
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = False
    if not loopback and not api_key:
        raise ServeError("binding the proxy outside loopback requires KGL_PROXY_KEY or --api-key")
    if demo and not loopback:
        raise ServeError("--demo requires a loopback address")
    demo_page = Path(__file__).with_name("demo.html").read_bytes() if demo else None
    srv = ThreadingHTTPServer((host, listen_port),
                              make_proxy_handler(rt, port, model_name, api_key, host, demo_page))
    srv.daemon_threads = True
    return srv
