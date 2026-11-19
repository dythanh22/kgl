"""Runtime = session + kernel channel with lazy (re)connect.

Only operations known to be safe to replay may retry after delivery.
"""
from __future__ import annotations

import logging
import threading
import time

from typing import Callable

from .config import Settings
from .errors import ConnectionLost, ExecutionError, KernelGone
from .jupyter import ExecResult, JupyterClient, KernelChannel, OnOutput
from .remote import MarkerParser, last_marker, render
from .state import Session

log = logging.getLogger("kgl.runtime")


class Runtime:
    def __init__(self, session: Session, settings: Settings) -> None:
        self.session, self.settings = session, settings
        self.client = JupyterClient(session.base_url, session.headers)
        self._channel: KernelChannel | None = None
        self._lock = threading.RLock()

    def drop_channel(self) -> None:
        with self._lock:
            if self._channel:
                self._channel.close()
            self._channel = None

    def _get_channel(self, remaining: Callable[[], float | None]) -> KernelChannel:
        if self._channel is None or self._channel.closed:
            self.client.get_kernel(self.session.kernel_id, min(self.client.timeout, remaining() or self.client.timeout))
            self._channel = self.client.open_channel(self.session.kernel_id, min(15.0, remaining() or 15.0))
            log.debug("opened kernel channel")
        return self._channel

    def run(self, code: str, on_output: OnOutput | None = None,
            idle_timeout: float | None = None, total_timeout: float | None = None,
            idempotent: bool = False, check: bool = False) -> ExecResult:
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
            for attempt in (0, 1):
                try:
                    ch = self._get_channel(remaining)
                    res = ch.execute(code, on_output, idle_timeout, remaining(),
                                     self.settings.ping_interval)
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
        finally:
            self._lock.release()
        if check and res.error:
            raise ExecutionError(res.error["ename"], res.error["evalue"], res.error["traceback"])
        return res

    def call(self, template: str, check: bool = True, timeout: float = 60.0, **values: object) -> dict:
        """Run a snippet and return its last marker (structured reply)."""
        res = self.run(render(template, **values), check=check, total_timeout=timeout)
        marker = last_marker(res.text)
        if marker is None:
            raise KernelGone("remote snippet returned no result: " + res.text[-300:])
        return marker

    def run_markers(self, code: str, on_text=None, **kw) -> tuple[ExecResult, list[dict]]:
        """Run code, forwarding non-marker text to on_text and collecting markers."""
        parser = MarkerParser()

        def handle(name: str, text: str) -> None:
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

    def close(self) -> None:
        self.drop_channel()
