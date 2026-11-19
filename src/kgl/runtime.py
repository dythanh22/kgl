"""Runtime = session + kernel channel with lazy (re)connect.

Retry policy: a failed *connection* or *send* is retried once (nothing ran).
A drop after the code was sent is NOT retried unless the caller says the
operation is idempotent (e.g. inference), because it could run code twice.
"""
from __future__ import annotations

import logging
import threading

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

    def _get_channel(self) -> KernelChannel:
        if self._channel is None or self._channel.closed:
            self.client.get_kernel(self.session.kernel_id)  # raises KernelGone
            self._channel = self.client.open_channel(self.session.kernel_id)
            log.debug("opened kernel channel")
        return self._channel

    def run(self, code: str, on_output: OnOutput | None = None,
            idle_timeout: float | None = None, total_timeout: float | None = None,
            idempotent: bool = False, check: bool = False) -> ExecResult:
        """Execute code on the VM. check=True raises ExecutionError on remote error."""
        with self._lock:
            for attempt in (0, 1):
                try:
                    ch = self._get_channel()
                    res = ch.execute(code, on_output, idle_timeout, total_timeout,
                                     self.settings.ping_interval)
                    break
                except ConnectionLost as e:
                    self.drop_channel()
                    if attempt == 1 or (e.sent and not idempotent):
                        raise
                    log.warning("connection lost (%s); reconnecting once", e)
                except TimeoutError:
                    self.drop_channel()  # stream state unknown after a timeout
                    raise
        if check and res.error:
            raise ExecutionError(res.error["ename"], res.error["evalue"], res.error["traceback"])
        return res

    def call(self, template: str, check: bool = True, **values: object) -> dict:
        """Run a snippet and return its last marker (structured reply)."""
        res = self.run(render(template, **values), check=check)
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
