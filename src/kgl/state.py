"""Local session state. The base URL may contain a bearer-like token, so the
file is created 0600 inside a 0700 directory and URLs are redacted in logs."""
from __future__ import annotations

import json
import os
import re
import tempfile
import time
import uuid
from contextlib import contextmanager

from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from .errors import KglError


def redact_url(url: str) -> str:
    """Only display the endpoint authority, omitting userinfo and URL credentials."""
    try:
        sp = urlsplit(url)
    except ValueError:
        return "<invalid-url>"
    authority = sp.netloc.rsplit("@", 1)[-1]
    return f"{sp.scheme}://{authority}" + ("/…" if sp.path or sp.query else "")


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
    revision: str | None = None


_UNCHECKED = object()


class SessionStore:
    def __init__(self, home: Path) -> None:
        self.home = Path(home)
        self.path = self.home / "session.json"

    def load(self) -> Session | None:
        try:
            return Session(**json.loads(self.path.read_text()))
        except FileNotFoundError:
            return None

    @contextmanager
    def _write_lock(self, expected):
        self.home.mkdir(parents=True, exist_ok=True)
        os.chmod(self.home, 0o700)
        with (self.home / "session.lock").open("a") as lock:
            if os.name == "nt":
                import msvcrt
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(lock, fcntl.LOCK_EX)
            if expected is not _UNCHECKED:
                current = self.load()
                # Old stores have no revision. Their connection timestamp identifies the session.
                def identity(session):
                    if session is None:
                        return None
                    return session.kernel_id, session.connected_at, session.revision

                if identity(current) != identity(expected):
                    raise KglError("session changed in another terminal; run kgl status")
            yield

    def save(self, session: Session, *, expected=_UNCHECKED) -> None:
        with self._write_lock(expected):
            revision = uuid.uuid4().hex
            fd, name = tempfile.mkstemp(prefix=".session-", dir=self.home)
            tmp = Path(name)
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(dict(asdict(session), revision=revision), f)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, self.path)
            finally:
                tmp.unlink(missing_ok=True)
            session.revision = revision

    def clear(self, *, expected=_UNCHECKED) -> None:
        with self._write_lock(expected):
            self.path.unlink(missing_ok=True)
