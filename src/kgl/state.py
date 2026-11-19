"""Local session state. The base URL may contain a bearer-like token, so the
file is created 0600 inside a 0700 directory and URLs are redacted in logs."""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlsplit


def redact_url(url: str) -> str:
    """Drop query/fragment and shorten long path segments (possible tokens)."""
    try:
        sp = urlsplit(url)
    except ValueError:
        return "<invalid-url>"
    parts = [p if len(p) < 16 else p[:4] + "…" for p in sp.path.split("/")]
    return f"{sp.scheme}://{sp.netloc}{'/'.join(parts)}" + ("?…" if sp.query else "")


_SENSITIVE = re.compile(r"(?i)(token|cookie|authorization)")


def redact_headers(headers: dict[str, str]) -> dict[str, str]:
    return {k: ("<hidden>" if _SENSITIVE.search(k) else v) for k, v in headers.items()}


@dataclass
class Session:
    base_url: str
    kernel_id: str
    headers: dict[str, str] = field(default_factory=dict)
    connected_at: float = field(default_factory=time.time)
    serve: dict | None = None  # {"port", "model", "backend", "dir"}


class SessionStore:
    def __init__(self, home: Path) -> None:
        self.home = Path(home)
        self.path = self.home / "session.json"

    def load(self) -> Session | None:
        if not self.path.exists():
            return None
        data = json.loads(self.path.read_text())
        return Session(**data)

    def save(self, session: Session) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        os.chmod(self.home, 0o700)
        tmp = self.path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(asdict(session), f)
        os.replace(tmp, self.path)

    def clear(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
