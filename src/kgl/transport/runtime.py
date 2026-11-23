"""Runtime = session + kernel channel with lazy (re)connect.

Only operations known to be safe to replay may retry after delivery.
"""
from __future__ import annotations

from dataclasses import replace
import logging
import threading
import time

from typing import Callable

from ..core.config import Settings
from ..core.errors import ConnectionLost, ExecutionError, KernelGone, ProtocolError
from .jupyter import ExecResult, JupyterClient, KernelChannel, OnOutput
from ..remote import MarkerParser, last_marker, new_marker, render
from ..core.state import Session

log = logging.getLogger("kgl.runtime")


class Runtime:
    def __init__(self, session: Session, settings: Settings) -> None:
        self.session, self.settings = session, settings
        self.client = JupyterClient(session.base_url, session.headers)
        self._channel: KernelChannel | None = None
        self._lock = threading.RLock()
        self._last_used = None

    def drop_channel(self) -> None:
        with self._lock:
            if self._channel:
                self._channel.close()
            self._channel = None

    def _get_channel(self, remaining: Callable[[], float | None]) -> KernelChannel:
        if self._channel is None or self._channel.closed:
            self._check_kernel(min(self.client.timeout, remaining() or self.client.timeout))
            self._channel = self.client.open_channel(self.session.kernel_id, min(15.0, remaining() or 15.0))
            log.debug("opened kernel channel")
        return self._channel

    def _check_kernel(self, timeout: float) -> None:
        state = self.client.get_kernel(self.session.kernel_id, timeout).get("execution_state")
        if state in ("dead", "restarting"):
            raise KernelGone(f"kernel is {state}")

    def _check_lost_channel(self, timeout: float) -> None:
        try:
            self._check_kernel(timeout)
        except ConnectionLost:
            pass  # Preserve the original uncertain WebSocket outcome.

    def run(self, code: str, on_output: OnOutput | None = None,
            idle_timeout: float | None = None, total_timeout: float | None = None,
            idempotent: bool = False, check: bool = False, check_client=None, on_submit=None,
            *, capture_output: bool = True) -> ExecResult:
        """Execute code on the VM. check=True raises ExecutionError on remote error."""
        deadline = None if total_timeout is None else time.monotonic() + total_timeout

        def remaining() -> float | None:
            if deadline is None:
                return None
            seconds = deadline - time.monotonic()
            if seconds <= 0:
                raise TimeoutError("request deadline exceeded")
            return seconds

        if not self._lock.acquire(timeout=-1 if total_timeout is None else total_timeout):
            raise TimeoutError("kernel busy: deadline exceeded waiting for previous request")
        try:
            if self._last_used is not None and time.monotonic() - self._last_used >= 20:
                self.drop_channel()
            for attempt in (0, 1):
                try:
                    ch = self._get_channel(remaining)
                    res = ch.execute(code, on_output, idle_timeout, remaining(),
                                     self.settings.ping_interval, self._check_lost_channel,
                                     check_client, on_submit, capture_output=capture_output)
                    break
                except ConnectionLost as e:
                    self.drop_channel()
                    if e.sent and (not idempotent or on_output is not None):
                        raise ConnectionLost(f"execution outcome unknown: {e}", sent=True) from e
                    if attempt == 1:
                        raise
                    log.warning("connection lost (%s); reconnecting once", e)
                except TimeoutError:
                    self.drop_channel()  # stream state unknown after a timeout
                    raise
                except (KernelGone, KeyboardInterrupt, OSError):
                    self.drop_channel()
                    raise
        finally:
            self._last_used = time.monotonic()
            self._lock.release()
        if check and (res.error or res.status != "ok"):
            error = res.error or {"ename": "ExecuteReply", "evalue": res.status}
            raise ExecutionError(error["ename"], error["evalue"], error.get("traceback"))
        return res

    def call(self, template: str, check: bool = True, timeout: float = 60.0,
             *, idempotent: bool = False, **values: object) -> dict:
        """Run a snippet and return its last marker (structured reply)."""
        marker = new_marker()
        res = self.run(render(template, marker=marker, **values), check=check,
                       total_timeout=timeout, idempotent=idempotent)
        reply = last_marker(res.text, marker)
        if reply is None:
            raise ProtocolError("remote snippet completed without a result marker")
        return reply

    def run_markers(self, code: str, on_text=None, *, marker: str, **kw) -> tuple[ExecResult, list[dict]]:
        """Run code, forwarding non-marker text to on_text and collecting markers."""
        parser = MarkerParser(marker=marker)

        def handle(name: str, text: str) -> None:
            if name != "stdout":
                if on_text:
                    on_text(name, text)
                return
            out = parser.feed(text)
            if out and on_text:
                on_text(name, out)

        res = self.run(code, handle, **kw)
        rest = parser.flush()
        if rest and on_text:
            on_text("stdout", rest)
        return res, parser.markers

    def kernel_alive(self) -> bool:
        try:
            self.client.get_kernel(self.session.kernel_id)
            return True
        except KernelGone:
            return False

    def control_runtime(self) -> Runtime | None:
        kernel_id = self.settings.control_kernel_id or self.session.control_kernel_id
        if kernel_id and kernel_id != self.session.kernel_id:
            return Runtime(replace(self.session, kernel_id=kernel_id), self.settings)
        return None

    def close(self) -> None:
        self.drop_channel()
