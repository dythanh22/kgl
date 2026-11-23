"""Hàng đợi hữu hạn + MỘT luồng thực thi tuần tự cho mọi thao tác với model.

Bất biến quan trọng:

* Chỉ luồng worker này được chạm vào model/processor/predictor (kể cả ``load``).
* Timeout HTTP / client ngắt kết nối **không** dừng được tính toán GPU đang
  chạy. Job đang chạy vẫn giữ luồng worker tới khi xong; request kế tiếp chỉ
  bắt đầu sau đó. Job còn nằm trong hàng đợi mà bị hủy/hết hạn thì bị bỏ qua.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from concurrent.futures import Future, InvalidStateError
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

from .schemas import ApiError

log = logging.getLogger("sam3_service.engine")

T = TypeVar("T")
_STOP = object()


def _set_exception(fut: Future, exc: BaseException) -> None:
    """Đặt lỗi lên future; bỏ qua nếu client đã hủy (future ở trạng thái CANCELLED)."""
    try:
        fut.set_exception(exc)
    except InvalidStateError:
        pass


def _set_result(fut: Future, value) -> None:
    try:
        fut.set_result(value)
    except InvalidStateError:
        pass


class QueueFull(ApiError):
    def __init__(self) -> None:
        super().__init__(
            503, "queue_full", "Hàng đợi đầy, thử lại sau.", headers={"Retry-After": "2"}
        )


class EngineUnavailable(ApiError):
    def __init__(self, message: str = "Dịch vụ chưa sẵn sàng hoặc đang dừng.") -> None:
        super().__init__(503, "not_ready", message, headers={"Retry-After": "5"})


class RequestTimeout(ApiError):
    def __init__(self, message: str) -> None:
        super().__init__(504, "timeout", message)


class InferenceFailed(ApiError):
    def __init__(self) -> None:
        super().__init__(500, "inference_failed", "Suy luận thất bại.")


@dataclass(frozen=True)
class JobTiming:
    queue_ms: float
    run_ms: float


@dataclass
class _Job(Generic[T]):
    fn: Callable[[], T]
    future: "Future[tuple[T, JobTiming]]"
    deadline: float | None
    enqueued: float
    request_id: str


class InferenceEngine:
    def __init__(
        self,
        *,
        max_queue: int,
        on_fatal: Callable[[ApiError], None] | None = None,
        thread_name: str = "sam3-infer",
    ) -> None:
        self._q: "queue.Queue[object]" = queue.Queue(maxsize=max_queue)
        self._on_fatal = on_fatal
        self._thread = threading.Thread(target=self._loop, name=thread_name, daemon=True)
        self._lock = threading.Lock()
        self._accepting = False
        self._busy = False
        self._fatal: ApiError | None = None

    # -- vòng đời ----------------------------------------------------------
    def start(self) -> None:
        with self._lock:
            self._accepting = True
        self._thread.start()

    def stop(self, timeout: float = 15.0) -> None:
        """Ngừng nhận job, hủy job đang chờ, chờ job đang chạy (tối đa timeout)."""
        with self._lock:
            self._accepting = False
            while True:
                try:
                    item = self._q.get_nowait()
                except queue.Empty:
                    break
                if isinstance(item, _Job):
                    _set_exception(item.future, EngineUnavailable("Dịch vụ đang dừng."))
            self._q.put_nowait(_STOP)  # hàng đợi vừa rỗng nên không bị đầy
        self._thread.join(timeout)
        if self._thread.is_alive():
            log.warning("Luồng suy luận vẫn đang chạy sau %.0fs; bỏ qua chờ.", timeout)

    # -- trạng thái --------------------------------------------------------
    @property
    def busy(self) -> bool:
        return self._busy

    @property
    def queue_depth(self) -> int:
        return self._q.qsize()

    @property
    def fatal(self) -> ApiError | None:
        return self._fatal

    # -- gửi job -----------------------------------------------------------
    async def run(
        self,
        fn: Callable[[], T],
        *,
        timeout_s: float | None,
        request_id: str = "-",
    ) -> tuple[T, JobTiming]:
        """Xếp ``fn`` vào hàng đợi và chờ kết quả.

        ``timeout_s`` tính cả thời gian chờ trong hàng đợi.
        """
        if self._fatal is not None:
            raise EngineUnavailable("Model gặp lỗi không phục hồi được.")
        now = time.monotonic()
        fut: "Future[tuple[T, JobTiming]]" = Future()
        job = _Job(
            fn=fn,
            future=fut,
            deadline=None if timeout_s is None else now + timeout_s,
            enqueued=now,
            request_id=request_id,
        )
        with self._lock:
            if not self._accepting:
                raise EngineUnavailable()
            try:
                self._q.put_nowait(job)
            except queue.Full:
                raise QueueFull() from None
        try:
            return await asyncio.wait_for(asyncio.wrap_future(fut), timeout_s)
        except asyncio.TimeoutError:
            # wait_for đã hủy future nếu job chưa chạy; nếu đang chạy thì không
            # hủy được và GPU vẫn tiếp tục tới khi xong lượt này.
            if fut.running():
                raise RequestTimeout(
                    "Hết thời gian trong lúc suy luận (GPU có thể vẫn đang hoàn tất lượt này)."
                ) from None
            raise RequestTimeout("Hết thời gian khi chờ trong hàng đợi.") from None
        except asyncio.CancelledError:
            fut.cancel()  # client ngắt kết nối: job chưa chạy sẽ bị bỏ qua
            raise

    # -- worker ------------------------------------------------------------
    def _loop(self) -> None:
        while True:
            item = self._q.get()
            if item is _STOP:
                return
            job: _Job = item  # type: ignore[assignment]
            if not job.future.set_running_or_notify_cancel():
                continue  # đã bị hủy khi còn chờ
            start = time.monotonic()
            if job.deadline is not None and start > job.deadline:
                _set_exception(job.future, RequestTimeout("Hết hạn khi chờ trong hàng đợi."))
                continue
            self._busy = True
            try:
                value = job.fn()
            except ApiError as exc:
                _set_exception(job.future, exc)
                if exc.fatal:
                    self._mark_fatal(exc)
            except Exception:  # noqa: BLE001 - không để worker chết vì một job
                log.exception("request_id=%s lỗi không mong đợi trong suy luận", job.request_id)
                _set_exception(job.future, InferenceFailed())
            else:
                end = time.monotonic()
                _set_result(
                    job.future,
                    (
                        value,
                        JobTiming(
                            queue_ms=(start - job.enqueued) * 1000.0,
                            run_ms=(end - start) * 1000.0,
                        ),
                    ),
                )
            finally:
                self._busy = False

    def _mark_fatal(self, exc: ApiError) -> None:
        self._fatal = exc
        log.critical("Lỗi không phục hồi được: %s", exc.message)
        if self._on_fatal is not None:
            try:
                self._on_fatal(exc)
            except Exception:  # noqa: BLE001
                log.exception("on_fatal lỗi")
