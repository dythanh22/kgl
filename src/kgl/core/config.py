"""Settings from environment variables (no credentials are stored here)."""
from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass
from pathlib import Path

from ..protocol import MAX_REQUEST_BYTES
from .errors import KglError


def seconds(value: float, name: str, *, allow_zero: bool = False) -> float:
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise KglError(f"{name} must be a finite {'non-negative' if allow_zero else 'positive'} number")
    if value > threading.TIMEOUT_MAX:
        raise KglError(f"{name} exceeds the maximum supported duration")
    return value


@dataclass(frozen=True)
class Settings:
    home: Path
    workdir: str = "/kaggle/working"  # remote default dir for relative paths
    session_limit_hours: float = 12.0  # documented CPU/GPU session cap (verify!)
    chunk_kb: int = 256  # raw bytes per upload/download message
    ping_interval: float = 20.0  # WebSocket keep-alive while waiting
    control_kernel_id: str | None = None
    request_bytes: int = MAX_REQUEST_BYTES
    interrupt_infer_kernel: bool = False

    def approx_session_remaining(self, uptime_s: int | float | None) -> int | None:
        if uptime_s is None:
            return None
        return max(0, int(self.session_limit_hours * 3600 - uptime_s))

    @classmethod
    def from_env(cls) -> "Settings":
        home = Path(os.environ.get("KGL_HOME") or Path.home() / ".config" / "kgl")
        def number(name, default, convert):
            try:
                return convert(os.environ.get(name, default))
            except ValueError:
                raise KglError(f"{name} has an invalid numeric value") from None

        chunk = number("KGL_CHUNK_KB", "256", int)
        if not 1 <= chunk <= 512:
            raise KglError("KGL_CHUNK_KB must be between 1 and 512 (before base64 encoding)")
        request_bytes = number("KGL_REQUEST_BYTES", MAX_REQUEST_BYTES, int)
        if request_bytes <= 0:
            raise KglError("KGL_REQUEST_BYTES must be positive")
        return cls(
            home=home,
            workdir=os.environ.get("KGL_WORKDIR", "/kaggle/working"),
            session_limit_hours=seconds(number("KGL_SESSION_LIMIT_HOURS", "12", float),
                                        "KGL_SESSION_LIMIT_HOURS"),
            chunk_kb=chunk,
            ping_interval=seconds(number("KGL_PING_INTERVAL", "20", float), "KGL_PING_INTERVAL"),
            control_kernel_id=os.environ.get("KGL_CONTROL_KERNEL_ID") or None,
            request_bytes=request_bytes,
            interrupt_infer_kernel=os.environ.get("KGL_INTERRUPT_INFER_KERNEL") == "1",
        )
