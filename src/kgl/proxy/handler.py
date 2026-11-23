"""Serve the notebook model through a local HTTP API."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import logging
import select
import socket
import threading
import time

from ..client.infer import infer_chat
from ..client.serve_manager import ServeManager
from ..core.errors import ConnectionLost, KernelGone, KglError
from ..backends import backend_spec
from ..protocol import chat_args, chat_defaults
from ..transport.runtime import Runtime
from .static import serve_file
from .security import is_authorized, validate_bind, validate_host_origin

log = logging.getLogger(__name__)


def make_proxy_handler(rt: Runtime, port: int, model_name: str, api_key: str | None,
                       listen_host: str = "127.0.0.1", static: Path | None = None):
    last_health = None
    admission = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        timeout = 10

        def log_message(self, fmt, *args):
            log.info("proxy: " + fmt, *args)

        def _json(self, code: int, obj: object, *, retry_after: bool = False) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            if retry_after:
                self.send_header("Retry-After", "1")
            self.end_headers()
            self.wfile.write(body)

        def _err(self, code: int, msg: str, typ: str = "kgl_error") -> None:
            self._json(code, {"error": {"message": msg, "type": typ}})

        def _authorized(self) -> bool:
            if is_authorized(self.headers, api_key):
                return True
            self._err(401, "invalid api key", "invalid_request_error")
            return False

        def _valid_host(self) -> bool:
            error = validate_host_origin(self.headers, self.connection.getsockname(), listen_host)
            if error:
                self._err(*error, "invalid_request_error")
                return False
            return True

        def do_GET(self):
            nonlocal last_health
            if not self._valid_host():
                return
            if self.path == "/health":
                return self._json(200, {"status": "ok"})
            if static and self.path not in ("/ready", "/models", "/v1/models"):
                return serve_file(self, static)
            if not self._authorized():
                return
            if self.path == "/ready":
                deadline = time.monotonic() + 5
                try:
                    kernel = rt.client.get_kernel(rt.session.kernel_id, timeout=5)
                except (KglError, TimeoutError) as exc:
                    return self._err(503, str(exc), "kernel_rest_unavailable")
                if kernel.get("execution_state") == "busy":
                    expected = rt.session.serve
                    result = {"state": "busy", "instance": expected.instance if expected else None}
                    if last_health is not None:
                        checked_at, health = last_health
                        result.update(last_health=health, health_age_s=time.monotonic() - checked_at)
                    return self._json(503, result, retry_after=True)
                try:
                    result = ServeManager(rt, port).health(timeout=max(.01, deadline - time.monotonic()))
                except (KglError, TimeoutError) as exc:
                    return self._err(503, str(exc), "runtime_unavailable")
                health = result.get("health", {})
                expected = rt.session.serve
                if health:
                    last_health = (time.monotonic(), health)
                ready = (result["state"] == "ready" and health.get("model") == model_name
                         and (not expected or health.get("backend") == expected.backend)
                         and (not expected or health.get("instance") == expected.instance)
                         and backend_spec(health.get("backend")).proxy_ready(health))
                return self._json(200 if ready else 503, result, retry_after=result["state"] == "busy")
            if self.path in ("/v1/models", "/models"):
                return self._json(200, {"object": "list",
                                        "data": [{"id": model_name, "object": "model"}]})
            self._err(404, "not found")

        def do_POST(self):
            if not self._valid_host() or not self._authorized():
                return
            custom = self.path == "/handle"
            serve = rt.session.serve
            spec = backend_spec(serve.backend if serve else None)
            if (not custom and self.path not in ("/v1/chat/completions", "/chat/completions")
                    or custom != spec.custom):
                return self._err(404, "route unavailable for the active backend")
            if self.headers.get_content_type() != "application/json":
                return self._err(415, "Content-Type must be application/json", "invalid_request_error")
            try:
                lengths = self.headers.get_all("Content-Length", [])
                if len(lengths) != 1 or self.headers.get("Transfer-Encoding"):
                    raise ValueError("a single Content-Length is required")
                length = int(lengths[0])
                if length > rt.settings.request_bytes:
                    return self._err(413, f"request exceeds {rt.settings.request_bytes} bytes", "invalid_request_error")
                if length <= 0:
                    raise ValueError("request cannot be empty")
                req = json.loads(self.rfile.read(length))
                if spec.passthrough:
                    if not isinstance(req, dict):
                        raise ValueError("request must be a JSON object")
                    req = chat_defaults(req, serve.defaults)
                    chat_args(req, model_name, serve.capabilities)
                elif not custom:
                    chat_args(req, model_name)
            except (ValueError, RecursionError) as exc:
                return self._err(400, str(exc), "invalid_request_error")
            except TimeoutError:
                return self._err(408, "timed out reading request body", "invalid_request_error")
            stream = not custom and req.get("stream") is True
            started = False

            def on_chunk(obj: dict) -> None:
                nonlocal started
                if not started:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    started = True
                self.wfile.write(("data: " + json.dumps(obj) + "\n\n").encode())
                self.wfile.flush()

            def stream_error(message: str):
                log.warning("incomplete stream: %s", message)
                event = {"error": {"message": message, "type": "incomplete_stream"}}
                try:
                    self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
                    self.wfile.flush()
                except OSError:
                    pass

            def check_client():
                readable, _, _ = select.select([self.connection], [], [], 0)
                if readable and not self.connection.recv(1, socket.MSG_PEEK):
                    raise BrokenPipeError("HTTP client disconnected")

            try:
                if not custom:
                    req.setdefault("model", model_name)
                if not admission.acquire(blocking=False):
                    return self._err(503, "inference already running", "worker_busy")
                try:
                    options = {"custom": True} if custom else {}
                    if spec.passthrough:
                        options["timeout"] = serve.defaults.get("timeout", 600)
                    r = infer_chat(rt, port, req, on_chunk if stream else None, check_client=check_client, **options)
                finally:
                    admission.release()
            except (ConnectionLost, KernelGone) as e:
                if started:
                    return stream_error(str(e))
                return self._err(503, f"runtime unavailable: {e}", "kgl_connection_lost")
            except (BrokenPipeError, ConnectionResetError):
                return
            except (KglError, TimeoutError) as e:
                if started:
                    return stream_error(str(e))
                return self._err(504 if isinstance(e, TimeoutError) else 502, str(e))
            if stream and started:
                if r.status != 200:
                    return stream_error(r.error or "incomplete upstream response")
                try:
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except BrokenPipeError:
                    pass
                return
            if r.status == 200 and (custom or r.body is not None):
                return self._json(200, r.body)
            status = r.status if r.status in (400, 409, 413, 503, 504) else 502
            self._err(status, r.error or "upstream error")

    return Handler


def make_proxy_server(rt: Runtime, port: int, model_name: str, host: str = "127.0.0.1",
                      listen_port: int = 8001, api_key: str | None = None,
                      *, static: Path | str | None = None) -> ThreadingHTTPServer:
    host = validate_bind(host, api_key, static is not None)
    if static is not None:
        static = Path(static).resolve()
        if not static.is_dir():
            raise KglError("--static requires a directory")
    srv = ThreadingHTTPServer((host, listen_port),
                              make_proxy_handler(rt, port, model_name, api_key, host, static))
    srv.daemon_threads = True
    return srv
