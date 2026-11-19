"""Model serving: manage the VM-side server, run inference through Jupyter
messaging, and expose a local OpenAI-compatible shim (no public tunnel)."""
from __future__ import annotations

import base64
import json
import logging
import posixpath
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable

from .errors import ConnectionLost, KernelGone, KglError, ServeError
from .remote import (INFER, SERVE_HEALTH, SERVE_LOGS, SERVE_START, SERVE_STOP,
                     MarkerParser, render, resolve_path)
from .runtime import Runtime
from .files import upload_file

log = logging.getLogger("kgl.serving")
SERVER_SCRIPT = Path(__file__).with_name("remote_server.py")


class ServeManager:
    def __init__(self, rt: Runtime, port: int = 8000) -> None:
        self.rt, self.port = rt, port
        self.dir = posixpath.join(rt.settings.workdir, ".kgl")

    def start(self, model: str, backend: str = "transformers", dtype: str = "float16",
              wait: bool = True, wait_timeout: float = 900.0,
              progress: Callable[[str], None] | None = None) -> dict:
        say = progress or (lambda s: None)
        # Upload the server script (small; verified by sha256).
        upload_file(self.rt, SERVER_SCRIPT, posixpath.join(self.dir, "serve.py"))
        args = ["--model", model, "--backend", backend, "--dtype", dtype,
                "--port", str(self.port)]
        r = self.rt.call(SERVE_START, DIR=self.dir, ARGS=args)
        say("server process already running" if r.get("already_running")
            else f"server process started (pid {r['pid']})")
        if not wait:
            return r
        return self.wait_ready(wait_timeout, say)

    def health(self) -> dict:
        return self.rt.call(SERVE_HEALTH, DIR=self.dir, PORT=self.port)

    def wait_ready(self, timeout: float, say: Callable[[str], None] = lambda s: None) -> dict:
        t0, last = time.monotonic(), None
        while time.monotonic() - t0 < timeout:
            h = self.health()
            state = h["state"]
            if state != last:
                say(f"server state: {state}")
                last = state
            if state == "ready":
                return h
            if h.get("health", {}).get("status") == "error":
                raise ServeError("model failed to load: " + str(h["health"].get("error"))
                                 + "\nsee `kgl serve logs`")
            if state == "down" and time.monotonic() - t0 > 30:
                raise ServeError("server not responding:\n" + "\n".join(h.get("log_tail", [])))
            time.sleep(3)
        raise ServeError(f"timed out after {timeout:.0f}s waiting for model; see `kgl serve logs`")

    def stop(self) -> dict:
        return self.rt.call(SERVE_STOP, DIR=self.dir)

    def logs(self, n: int = 50) -> list[str]:
        return self.rt.call(SERVE_LOGS, DIR=self.dir, N=n).get("lines", [])


@dataclass
class InferResult:
    status: int = 0
    body: dict | None = None
    text: str = ""
    error: str | None = None
    latency_s: float = 0.0
    ttft_s: float | None = None  # time to first streamed token


def infer_chat(rt: Runtime, port: int, payload: dict,
               on_chunk: Callable[[dict], None] | None = None,
               timeout: float = 600.0) -> InferResult:
    """Send an OpenAI-style chat request to the VM server via the kernel."""
    res = InferResult()
    t0 = time.monotonic()
    pieces: list[str] = []

    def handle(m: dict) -> None:
        if "chunk" in m:
            if m["chunk"] == "[DONE]":
                return
            try:
                obj = json.loads(m["chunk"])
            except ValueError:
                return
            delta = (obj.get("choices") or [{}])[0].get("delta", {}).get("content")
            if delta:
                if res.ttft_s is None:
                    res.ttft_s = time.monotonic() - t0
                pieces.append(delta)
            if on_chunk:
                on_chunk(obj)
        if "status" in m:
            res.status = m["status"]
            res.body = m.get("body")
            res.error = m.get("error")

    parser = MarkerParser(handle)
    code = render(INFER, PAYLOAD=base64.b64encode(json.dumps(payload).encode()).decode(),
                  PORT=port, TIMEOUT=timeout)
    r = rt.run(code, lambda name, text: parser.feed(text), idempotent=True,
               total_timeout=timeout + 30)
    if r.error:
        raise KglError(f"remote error: {r.error['ename']}: {r.error['evalue']}")
    res.latency_s = time.monotonic() - t0
    if res.body:
        res.text = (res.body.get("choices") or [{}])[0].get("message", {}).get("content", "")
    else:
        res.text = "".join(pieces)
    if res.status == 0 and not res.error:
        res.error = "no response from remote server"
    return res


def make_proxy_handler(rt: Runtime, port: int, model_name: str, api_key: str | None):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            log.info("proxy: " + fmt, *args)

        def _json(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _err(self, code: int, msg: str, typ: str = "kgl_error") -> None:
            self._json(code, {"error": {"message": msg, "type": typ}})

        def _authorized(self) -> bool:
            if not api_key:
                return True
            if self.headers.get("Authorization") == f"Bearer {api_key}":
                return True
            self._err(401, "invalid api key", "invalid_request_error")
            return False

        def do_GET(self):
            if self.path == "/health":
                return self._json(200, {"status": "ok"})
            if not self._authorized():
                return
            if self.path in ("/v1/models", "/models"):
                return self._json(200, {"object": "list",
                                        "data": [{"id": model_name, "object": "model"}]})
            self._err(404, "not found")

        def do_POST(self):
            if not self._authorized():
                return
            if self.path not in ("/v1/chat/completions", "/chat/completions"):
                return self._err(404, "only /v1/chat/completions is supported")
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                assert isinstance(req.get("messages"), list)
            except Exception:
                return self._err(400, "invalid JSON body (need 'messages')", "invalid_request_error")
            stream = req.get("stream") is True
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

            try:
                r = infer_chat(rt, port, req, on_chunk if stream else None)
            except (ConnectionLost, KernelGone) as e:
                if started:
                    return
                return self._err(503, f"runtime unavailable: {e}", "kgl_connection_lost")
            except BrokenPipeError:
                return
            except (KglError, TimeoutError) as e:
                if started:
                    return
                return self._err(504 if isinstance(e, TimeoutError) else 502, str(e))
            if stream and started:
                try:
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except BrokenPipeError:
                    pass
                return
            if r.status == 200 and r.body is not None:
                return self._json(200, r.body)
            self._err(503 if r.status == 503 else 502, r.error or "upstream error")

    return Handler


def make_proxy_server(rt: Runtime, port: int, model_name: str, host: str = "127.0.0.1",
                      listen_port: int = 8001, api_key: str | None = None) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer((host, listen_port),
                              make_proxy_handler(rt, port, model_name, api_key))
    srv.daemon_threads = True
    return srv
