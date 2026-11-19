"""Settings from environment variables (no credentials are stored here)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    home: Path
    workdir: str = "/kaggle/working"  # remote default dir for relative paths
    session_limit_hours: float = 12.0  # documented CPU/GPU session cap (verify!)
    chunk_kb: int = 256  # raw bytes per upload/download message
    ping_interval: float = 20.0  # WebSocket keep-alive while waiting

    @classmethod
    def from_env(cls) -> "Settings":
        home = Path(os.environ.get("KGL_HOME") or Path.home() / ".config" / "kgl")
        return cls(
            home=home,
            workdir=os.environ.get("KGL_WORKDIR", "/kaggle/working"),
            session_limit_hours=float(os.environ.get("KGL_SESSION_LIMIT_HOURS", "12")),
            chunk_kb=int(os.environ.get("KGL_CHUNK_KB", "256")),
            ping_interval=float(os.environ.get("KGL_PING_INTERVAL", "20")),
        )
