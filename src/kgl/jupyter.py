"""Jupyter Server REST + kernel channel client.

ASSUMPTION (unverified on Kaggle): the base URL points at a standard Jupyter
server, i.e. `{base}/api/kernels` and `{base}/api/kernels/<id>/channels`
exist. `kgl connect` / `kgl doctor` check this and fail loudly if not.
"""
from __future__ import annotations

import json
import logging
import os
import ssl
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable
from urllib.parse import parse_qsl, urlencode, urlsplit

from .errors import AuthError, ConnectionLost, KernelGone, KglError
from .state import redact_url
from .ws import WebSocket

log = logging.getLogger("kgl.jupyter")
OnOutput = Callable[[str, str], None]  # (stream_name, text)


class Endpoint:
    def __init__(self, base_url: str) -> None:
        sp = urlsplit(base_url.strip())
        if sp.scheme not in ("http", "https") or not sp.netloc:
            raise KglError("proxy URL must be a full http(s):// URL")
        self.scheme, self.netloc = sp.scheme, sp.netloc
        self.prefix, self.query = sp.path.rstrip("/"), sp.query
        self.redacted = redact_url(base_url)

    def _q(self, extra: dict[str, str] | None) -> str:
        pairs = parse_qsl(self.query) + list((extra or {}).items())
        return f"?{urlencode(pairs)}" if pairs else ""

    def http(self, path: str, extra: dict[str, str] | None = None) -> str:
        return f"{self.scheme}://{self.netloc}{self.prefix}{path}{self._q(extra)}"

    def ws(self, path: str, extra: dict[str, str] | None = None) -> str:
        scheme = "wss" if self.scheme == "https" else "ws"
        return f"{scheme}://{self.netloc}{self.prefix}{path}{self._q(extra)}"


@dataclass
class ExecResult:
    status: str = "ok"
    text: str = ""
    results: list[str] = field(default_factory=list)
    error: dict | None = None  # {"ename","evalue","traceback"}


class JupyterClient:
    def __init__(self, base_url: str, headers: dict[str, str] | None = None,
                 timeout: float = 20.0) -> None:
        self.ep = Endpoint(base_url)
        self.headers = dict(headers or {})
        self.timeout = timeout

    def _request(self, method: str, path: str, body: dict | None = None, timeout: float | None = None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            self.ep.http(path), data=data, method=method,
            headers={"Accept": "application/json", "Content-Type": "application/json",
                     **self.headers})
        try:
            ctx = (ssl.create_default_context(cafile=os.environ["KGL_CA_FILE"])
                   if os.environ.get("KGL_CA_FILE") else None)
            with urllib.request.urlopen(req, timeout=timeout if timeout is not None else self.timeout, context=ctx) as r:
                raw = r.read()
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise AuthError(f"HTTP {e.code} from {self.ep.redacted}: the proxy URL/token "
                                "is expired or not accepted. Start/refresh the notebook "
                                "session and run `kgl connect` again.") from e
            if e.code == 404 and path.startswith("/api/kernels/"):
                raise KernelGone("kernel not found on server") from e
            if e.code == 404:
                raise KglError(f"HTTP 404 for {path}: this URL does not look like a Jupyter "
                               "server base URL.") from e
            raise KglError(f"HTTP {e.code} for {method} {path}") from e
        except (urllib.error.URLError, OSError) as e:
            raise ConnectionLost(f"cannot reach {self.ep.redacted}: {e}") from e
        if not raw:
            return None
        try:
            return json.loads(raw)
        except ValueError as e:
            raise KglError("server returned non-JSON (login page? wrong URL?)") from e

    def list_kernels(self) -> list[dict]:
        return self._request("GET", "/api/kernels") or []

    def get_kernel(self, kernel_id: str, timeout: float | None = None) -> dict:
        return self._request("GET", f"/api/kernels/{kernel_id}", timeout=timeout)

    def create_kernel(self, name: str | None = None) -> dict:
        return self._request("POST", "/api/kernels", {"name": name} if name else {})

    def delete_kernel(self, kernel_id: str) -> None:
        self._request("DELETE", f"/api/kernels/{kernel_id}")

    def interrupt_kernel(self, kernel_id: str) -> None:
        self._request("POST", f"/api/kernels/{kernel_id}/interrupt")

    def restart_kernel(self, kernel_id: str) -> None:
        self._request("POST", f"/api/kernels/{kernel_id}/restart")

    def open_channel(self, kernel_id: str, connect_timeout: float = 15.0) -> "KernelChannel":
        session_id = uuid.uuid4().hex
        url = self.ep.ws(f"/api/kernels/{kernel_id}/channels", {"session_id": session_id})
        ws = WebSocket(url, headers=self.headers, connect_timeout=connect_timeout)
        return KernelChannel(ws, session_id)


class KernelChannel:
    def __init__(self, ws: WebSocket, session_id: str) -> None:
        self.ws, self.session_id = ws, session_id

    @property
    def closed(self) -> bool:
        return self.ws.closed

    def close(self) -> None:
        self.ws.close()

    def _request_msg(self, code: str) -> tuple[str, str]:
        msg_id = uuid.uuid4().hex
        msg = {
            "header": {"msg_id": msg_id, "username": "kgl", "session": self.session_id,
                       "date": datetime.now(timezone.utc).isoformat(),
                       "msg_type": "execute_request", "version": "5.3"},
            "parent_header": {}, "metadata": {},
            "content": {"code": code, "silent": False, "store_history": False,
                        "user_expressions": {}, "allow_stdin": False, "stop_on_error": True},
            "buffers": [], "channel": "shell",
        }
        return msg_id, json.dumps(msg)

    def execute(self, code: str, on_output: OnOutput | None = None,
                idle_timeout: float | None = None, total_timeout: float | None = None,
                ping_interval: float = 20.0) -> ExecResult:
        """Run code; stream output via on_output; return when kernel is idle.

        idle_timeout: max seconds without any message (None = unlimited).
        total_timeout: max seconds overall (None = unlimited).
        """
        msg_id, payload = self._request_msg(code)
        start = time.monotonic()
        self.ws.write_deadline = None if total_timeout is None else start + total_timeout
        self.ws.send(payload)
        res = ExecResult()
        parts: list[str] = []
        got_reply = got_idle = False
        idle_at: float | None = None
        last_msg = start
        while True:
            now = time.monotonic()
            if total_timeout is not None and now - start > total_timeout:
                raise TimeoutError(f"execution exceeded {total_timeout:.0f}s")
            if idle_timeout is not None and now - last_msg > idle_timeout:
                raise TimeoutError(f"no output for {idle_timeout:.0f}s")
            if got_idle and not got_reply and idle_at and now - idle_at > 5:
                raise ConnectionLost("missing execute_reply; execution outcome unknown", sent=True)
            wait = ping_interval
            if total_timeout is not None:
                wait = min(wait, max(0.05, total_timeout - (now - start)))
            if idle_timeout is not None:
                wait = min(wait, max(0.05, idle_timeout - (now - last_msg)))
            try:
                raw = self.ws.recv(timeout=wait)
            except TimeoutError:
                if total_timeout is not None and time.monotonic() - start >= total_timeout:
                    raise TimeoutError("execution deadline exceeded; outcome unknown")
                try:
                    self.ws.ping()
                except ConnectionLost as e:
                    raise ConnectionLost(str(e), sent=True) from e
                continue
            except ConnectionLost as e:
                raise ConnectionLost(str(e), sent=True) from e
            try:
                msg = json.loads(raw)
            except ValueError:
                log.debug("non-JSON frame ignored")
                continue
            if (msg.get("parent_header") or {}).get("msg_id") != msg_id:
                continue
            last_msg = time.monotonic()
            mtype = msg.get("msg_type") or (msg.get("header") or {}).get("msg_type")
            content = msg.get("content") or {}
            if mtype == "stream":
                text = content.get("text", "")
                parts.append(text)
                if on_output:
                    on_output(content.get("name", "stdout"), text)
            elif mtype in ("execute_result", "display_data"):
                plain = (content.get("data") or {}).get("text/plain")
                if plain:
                    res.results.append(plain)
                    if on_output:
                        on_output("result", plain + "\n")
            elif mtype == "error":
                res.error = {"ename": content.get("ename", "Error"),
                             "evalue": content.get("evalue", ""),
                             "traceback": content.get("traceback", [])}
            elif mtype == "execute_reply":
                got_reply = True
                res.status = content.get("status", "ok")
            elif mtype == "status" and content.get("execution_state") == "idle":
                got_idle, idle_at = True, time.monotonic()
            if got_reply and got_idle:
                break
        res.text = "".join(parts)
        if res.error:
            res.status = "error"
        return res
