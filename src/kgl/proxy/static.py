"""Serve files below a caller-selected directory on loopback."""
import mimetypes
from urllib.parse import unquote, urlsplit


def serve_file(handler, static):
    try:
        name = unquote(urlsplit(handler.path).path).lstrip("/")
        if not name:
            name = "index.html" if (static / "index.html").is_file() else "demo.html"
        target = (static / name).resolve()
        if not target.is_relative_to(static) or not target.is_file():
            return handler._err(404, "not found")
        content = target.read_bytes()
    except (ValueError, OSError, RuntimeError):
        return handler._err(404, "not found")
    handler.send_response(200)
    handler.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
    handler.send_header("Content-Length", str(len(content)))
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("X-Frame-Options", "DENY")
    handler.send_header("X-Content-Type-Options", "nosniff")
    handler.end_headers()
    handler.wfile.write(content)
