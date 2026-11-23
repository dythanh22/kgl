"""VM snippets and their output protocol."""

import posixpath

from .markers import MARKER, PRELUDE, MarkerParser, last_marker, new_marker
from .renderer import render
from .templates import (
    PROBE,
    UPLOAD_CHUNK,
    COMMIT_UPLOAD,
    DISCARD_UPLOAD,
    FILE_SIZE,
    STAT,
    DOWNLOAD_CHUNK,
    SHELL,
    SERVE_START,
    SERVE_HEALTH,
    SERVE_STOP,
    SERVE_LOGS,
    INFER,
    CANCEL_INFER,
)


def resolve_path(path: str, workdir: str) -> str:
    return path if path.startswith("/") else posixpath.join(workdir, path)
