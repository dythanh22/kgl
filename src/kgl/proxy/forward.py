"""Buffered HTTP forwarding to a single loopback service on the notebook."""
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from ..client.files import download_file, upload_file
from ..core.errors import ConnectionLost, KernelGone, KglError
from ..remote.templates import FORWARD
from .security import validate_bind, validate_host_origin


def end_to_end(headers):
    # The local and notebook connections have independent framing and authorities.
    excluded = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
                "proxy-connection", "te", "trailer", "transfer-encoding", "upgrade", "host",
                "content-length", "origin", "referer", "cookie", "set-cookie"}
    for name, value in headers:
        if name.lower() == "connection":
            excluded.update(token.strip().lower() for token in value.split(","))
    return [(k, v) for k, v in headers if k.lower() not in excluded]


def exchange(rt, port, method, path, headers, body):
    chunk = rt.settings.chunk_kb * 1024
    remote = "/tmp/kgl-forward-" + uuid4().hex
    with TemporaryDirectory() as tmp:
        local = Path(tmp) / "body"
        large = len(body) > chunk
        result = None
        try:
            if large:
                local.write_bytes(body)
                upload_file(rt, local, remote + ".in")
            result = rt.call(FORWARD, timeout=100, PORT=port, METHOD=method, PATH=path, HEADERS=headers,
                             BODY="" if large else base64.b64encode(body).decode(), FILE=remote + ".in" if large else "",
                             RESULT=remote + ".out", LIMIT=rt.settings.request_bytes, CHUNK=chunk)
            if result.get("oversize"):
                raise KglError("upstream response exceeds the configured limit")
            if result.get("file"):
                download_file(rt, remote + ".out", local)
                data = local.read_bytes()
            else:
                data = base64.b64decode(result["body"], validate=True)
            response_headers = end_to_end(result["headers"])
            if method == "HEAD":
                response_headers.extend((k, v) for k, v in result["headers"]
                                        if k.lower() == "content-length" and v.isdigit())
            return result["status"], response_headers, data
        finally:
            if large or result is None or result.get("file"):
                try:
                    rt.call("from pathlib import Path\nfor p in __PATHS__: Path(p).unlink(missing_ok=True)\n_emit({})",
                            timeout=5, idempotent=True, PATHS=[remote + ".in", remote + ".out"])
                except (KglError, OSError):
                    pass


def make_forward_server(rt, remote_port, host, port):
    host = validate_bind(host, None, False)
    if not 1 <= remote_port <= 65535:
        raise KglError("invalid remote port")

    class Handler(BaseHTTPRequestHandler):
        timeout = 30

        def log_message(self, *args):
            pass

        def forward(self):
            error = validate_host_origin(self.headers, self.connection.getsockname(), host)
            if error:
                return self.send_error(*error)
            if not self.path.startswith("/") or self.path.startswith("//"):
                return self.send_error(400, "expected a path")
            sizes = self.headers.get_all("Content-Length", [])
            if (self.headers.get("Transfer-Encoding") or len(sizes) > 1
                    or (sizes and (not sizes[0].isascii() or not sizes[0].isdigit() or len(sizes[0]) > 18))):
                return self.send_error(400, "invalid Content-Length")
            if not sizes and self.command in ("POST", "PUT", "PATCH"):
                return self.send_error(411)
            size = int(sizes[0]) if sizes else 0
            if size > rt.settings.request_bytes:
                return self.send_error(413)
            try:
                body = self.rfile.read(size)
            except TimeoutError:
                return self.send_error(408)
            if len(body) != size:
                return self.send_error(400, "incomplete request")
            headers = end_to_end(self.headers.items())
            try:
                status, headers, body = exchange(rt, remote_port, self.command, self.path, headers, body)
            except (ConnectionLost, KernelGone):
                return self.send_error(503, "notebook unavailable")
            except TimeoutError:
                return self.send_error(504)
            except (OSError, KglError):
                return self.send_error(502)
            self.send_response(status)
            for key, value in headers:
                self.send_header(key, value)
            if self.command != "HEAD" and status not in (204, 304):
                self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = do_OPTIONS = forward

    return ThreadingHTTPServer((host, port), Handler)
