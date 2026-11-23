"""Parse nonce-scoped results from kernel stdout."""

import json
import uuid
from collections.abc import Callable

MARKER = "@@KGL@@"

PRELUDE = (
    "import json as _j, sys as _s\n"
    "def _emit(o):\n"
    f"    _s.stdout.write({MARKER!r} + _j.dumps(o) + '\\n'); _s.stdout.flush()\n"
)


def new_marker() -> str:
    return MARKER + uuid.uuid4().hex + "@@"


class MarkerParser:
    """Splits a text stream into passthrough text and `@@KGL@@{json}` events."""

    def __init__(self, on_marker: Callable[[dict], None] | None = None,
                 marker: str = MARKER, *, retain: bool = True) -> None:
        self.markers: list[dict] = []
        self._on_marker = on_marker
        self._buf = ""
        self._marker = marker
        self._line_start = True
        self._retain = retain

    def feed(self, text: str) -> str:
        self._buf += text
        out: list[str] = []
        while (i := self._buf.find("\n")) != -1:
            line, self._buf = self._buf[:i], self._buf[i + 1:]
            at_start, self._line_start = self._line_start, True
            if at_start and line.startswith(self._marker):
                try:
                    obj = json.loads(line[len(self._marker):])
                    if not isinstance(obj, dict):
                        raise ValueError("marker must contain an object")
                except ValueError:
                    out.append(line + "\n")
                    continue
                if self._retain:
                    self.markers.append(obj)
                if self._on_marker:
                    self._on_marker(obj)
            else:
                out.append(line + "\n")
        if self._buf and (not self._line_start
                          or not self._marker.startswith(self._buf[: len(self._marker)])):
            out.append(self._buf)
            self._buf = ""
            self._line_start = False
        return "".join(out)

    def flush(self) -> str:
        rest, self._buf = self._buf, ""
        return rest


def last_marker(text: str, marker: str = MARKER) -> dict | None:
    p = MarkerParser(marker=marker)
    p.feed(text if text.endswith("\n") else text + "\n")
    return p.markers[-1] if p.markers else None
