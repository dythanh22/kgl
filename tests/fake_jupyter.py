"""A tiny fake Jupyter server for tests (stdlib only).

REST: /api/kernels (+ /<id>, /interrupt, /restart), WebSocket channels.
execute_request code is run for real in a `python -u -c` subprocess, so
snippets from kgl.remote behave as on a VM (state is NOT kept between cells).
Optional ?token=... enforcement and fault injection hooks.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import struct
import subprocess
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def _read_exact(rfile, n):
    data = b""
    while len(data) < n:
        chunk = rfile.read(n - len(data))
        if not chunk:
            raise EOFError
        data += chunk
    return data


def read_frame(rfile):
    b0, b1 = _read_exact(rfile, 2)
    op, n = b0 & 0x0F, b1 & 0x7F
    if n == 126:
        n = struct.unpack("!H", _read_exact(rfile, 2))[0]
    elif n == 127:
        n = struct.unpack("!Q", _read_exact(rfile, 8))[0]
    key = _read_exact(rfile, 4) if b1 & 0x80 else b""
    data = _read_exact(rfile, n)
    if key:
        data = bytes(b ^ key[i % 4] for i, b in enumerate(data))
    return op, data


def frame(op, payload: bytes) -> bytes:
    n = len(payload)
    if n < 126:
        h = struct.pack("!BB", 0x80 | op, n)
    elif n < 65536:
        h = struct.pack("!BBH", 0x80 | op, 126, n)
    else:
        h = struct.pack("!BBQ", 0x80 | op, 127, n)
    return h + payload


class FakeJupyter:
    def __init__(self, token: str | None = None):
        self.token = token
        self.kernels = {}
        self.fail_next_execute = False  # close socket on next execute (no reply)
        self.executed: list[str] = []
        self.interrupts = 0
        self._procs: dict[str, subprocess.Popen] = {}
        self.new_kernel()
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _auth_ok(self):
                if outer.token is None:
                    return True
                if f"token={outer.token}" in self.path:
                    return True
                self._json(403, {"message": "forbidden"})
                return False

            def _json(self, code, obj):
                b = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_GET(self):
                if not self._auth_ok():
                    return
                path = self.path.split("?")[0]
                if path == "/api/kernels":
                    return self._json(200, list(outer.kernels.values()))
                parts = path.strip("/").split("/")
                if len(parts) == 3 and parts[:2] == ["api", "kernels"]:
                    k = outer.kernels.get(parts[2])
                    return self._json(200, k) if k else self._json(404, {})
                if len(parts) == 4 and parts[3] == "channels" and \
                        self.headers.get("Upgrade", "").lower() == "websocket":
                    return self._ws(parts[2])
                self._json(404, {})

            def do_POST(self):
                if not self._auth_ok():
                    return
                path = self.path.split("?")[0]
                n = int(self.headers.get("Content-Length", "0"))
                if n:
                    self.rfile.read(n)
                if path == "/api/kernels":
                    return self._json(201, outer.new_kernel())
                parts = path.strip("/").split("/")
                if len(parts) == 4 and parts[3] == "interrupt":
                    outer.interrupts += 1
                    p = outer._procs.get(parts[2])
                    if p:
                        p.kill()
                    return self._json(204, {})
                self._json(404, {})

            def do_DELETE(self):
                parts = self.path.split("?")[0].strip("/").split("/")
                if len(parts) == 3 and outer.kernels.pop(parts[2], None):
                    return self._json(204, {})
                self._json(404, {})

            def _ws(self, kid):
                if kid not in outer.kernels:
                    return self._json(404, {})
                key = self.headers["Sec-WebSocket-Key"]
                acc = base64.b64encode(hashlib.sha1((key + GUID).encode()).digest()).decode()
                self.wfile.write(("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n"
                                  f"Connection: Upgrade\r\nSec-WebSocket-Accept: {acc}\r\n\r\n").encode())
                self.wfile.flush()
                self.close_connection = True
                try:
                    while True:
                        op, data = read_frame(self.rfile)
                        if op == 8:
                            return
                        if op == 9:
                            self.wfile.write(frame(10, data))
                            continue
                        if op == 1:
                            outer._handle(self, kid, json.loads(data))
                except (EOFError, OSError):
                    return

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}" + (f"?token={self.token}" if self.token else "")

    def new_kernel(self):
        k = {"id": uuid.uuid4().hex, "name": "python3", "execution_state": "idle"}
        self.kernels[k["id"]] = k
        return k

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def _send(self, h, parent, channel, mtype, content):
        msg = {"header": {"msg_id": uuid.uuid4().hex, "msg_type": mtype}, "msg_type": mtype,
               "parent_header": parent, "content": content, "channel": channel}
        h.wfile.write(frame(1, json.dumps(msg).encode()))
        h.wfile.flush()

    def _handle(self, h, kid, msg):
        if msg["header"]["msg_type"] != "execute_request":
            return
        hdr, code = msg["header"], msg["content"]["code"]
        self.executed.append(code)
        if self.fail_next_execute:
            self.fail_next_execute = False
            h.connection.close()
            raise EOFError
        self._send(h, hdr, "iopub", "status", {"execution_state": "busy"})
        p = subprocess.Popen([sys.executable, "-u", "-c", code], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        self._procs[kid] = p
        tail = b""
        while True:
            chunk = os.read(p.stdout.fileno(), 4096)
            if not chunk:
                break
            tail = (tail + chunk)[-2000:]
            self._send(h, hdr, "iopub", "stream", {"name": "stdout", "text": chunk.decode("utf-8", "replace")})
        rc = p.wait()
        self._procs.pop(kid, None)
        if rc != 0:
            last = tail.decode("utf-8", "replace").strip().splitlines()[-1:] or [""]
            ename, _, evalue = last[0].partition(": ")
            self._send(h, hdr, "iopub", "error", {"ename": ename or "Error", "evalue": evalue,
                                                  "traceback": [tail.decode("utf-8", "replace")]})
        self._send(h, hdr, "shell", "execute_reply", {"status": "error" if rc else "ok"})
        self._send(h, hdr, "iopub", "status", {"execution_state": "idle"})
