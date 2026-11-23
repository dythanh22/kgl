"""Render a snippet in an isolated Jupyter namespace."""

import re

from .markers import MARKER, PRELUDE


def render(template: str, *, marker: str = MARKER, **values: object) -> str:
    """Substitute __NAME__ placeholders with repr() of values (safe quoting)."""
    out = re.sub(r"__([A-Z][A-Z0-9_]*?)__",
                 lambda match: repr(values[match[1]]) if match[1] in values else match[0],
                 template)
    return f"exec({(PRELUDE.replace(MARKER, marker) + out)!r}, {{}})"
