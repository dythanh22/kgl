"""Helpers shared by CLI commands."""

from functools import wraps
import sys

from ..core.config import Settings
from ..core.errors import NoSession
from ..core.state import Session, SessionStore
from ..transport.runtime import Runtime

EXIT_OK, EXIT_ERROR, EXIT_USAGE, EXIT_CONNECTION, EXIT_REMOTE = 0, 1, 2, 3, 4


def eprint(*a: object) -> None:
    print(*a, file=sys.stderr)


def require_session(store: SessionStore) -> Session:
    s = store.load()
    if s is None:
        raise NoSession("no active session. Run `kgl connect --proxy-url ...` first.")
    return s


def open_runtime(st: Settings, store: SessionStore) -> tuple[Runtime, Session]:
    s = require_session(store)
    return Runtime(s, st), s


def write_stream(name: str, text: str) -> None:
    out = sys.stderr if name == "stderr" else sys.stdout
    out.write(text)
    out.flush()


def with_runtime(fn):
    @wraps(fn)
    def command(args, settings, store):
        rt, session = open_runtime(settings, store)
        try:
            return fn(args, rt, session, store)
        finally:
            rt.close()
    return command
