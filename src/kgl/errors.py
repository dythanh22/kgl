"""Exception hierarchy. CLI exit codes: see cli.EXIT_*."""
from __future__ import annotations


class KglError(Exception):
    """Base class for all expected errors (shown without traceback)."""


class AuthError(KglError):
    """HTTP 401/403: proxy URL/token expired or not accepted."""


class NoSession(KglError):
    """No saved session: `kgl connect` has not been run."""


class ConnectionLost(KglError):
    """Network/WebSocket failure.

    `sent` is True when the failure happened after the request may have
    reached the kernel (so blindly retrying could run code twice).
    """

    def __init__(self, message: str, sent: bool = False) -> None:
        super().__init__(message)
        self.sent = sent


class KernelGone(KglError):
    """The kernel id we know about no longer exists on the server."""


class ProtocolError(KglError):
    """The remote operation did not return a valid result."""


class ExecutionError(KglError):
    """Remote code raised an exception."""

    def __init__(self, ename: str, evalue: str, traceback: list[str] | None = None) -> None:
        super().__init__(f"{ename}: {evalue}")
        self.ename, self.evalue, self.traceback = ename, evalue, traceback or []


class ServeError(KglError):
    """Model server (on the VM) failed to start / answer."""
