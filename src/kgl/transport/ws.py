"""Minimal RFC 6455 WebSocket client (stdlib only).

Only what Jupyter needs: text frames, fragmentation, ping/pong, close.
Reads keep an internal buffer, so a timeout never loses partial frames.
"""
from __future__ import annotations

import base64
import hashlib
import os
import socket
import ssl
import struct
import threading
import time
from urllib.parse import urlsplit

from ..core.errors import ConnectionLost, KglError

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
OP_CONT, OP_TEXT, OP_BIN, OP_CLOSE, OP_PING, OP_PONG = 0, 1, 2, 8, 9, 10


def _mask(data: bytes, key: bytes) -> bytes:
    if not data:
        return data
    k = (key * (len(data) // 4 + 1))[: len(data)]
    return (int.from_bytes(data, "big") ^ int.from_bytes(k, "big")).to_bytes(len(data), "big")


def encode_frame(opcode: int, payload: bytes, mask: bool = True, fin: bool = True) -> bytes:
    b0 = (0x80 if fin else 0) | opcode
    n = len(payload)
    mbit = 0x80 if mask else 0
    if n < 126:
        header = struct.pack("!BB", b0, mbit | n)
    elif n < 65536:
        header = struct.pack("!BBH", b0, mbit | 126, n)
    else:
        header = struct.pack("!BBQ", b0, mbit | 127, n)
    if mask:
        key = os.urandom(4)
        return header + key + _mask(payload, key)
    return header + payload


def parse_frame(buf: bytearray) -> tuple[bool, int, bytes, int] | None:
    """Return (fin, opcode, payload, bytes_used) or None if incomplete."""
    if len(buf) < 2:
        return None
    b0, b1 = buf[0], buf[1]
    fin, opcode = bool(b0 & 0x80), b0 & 0x0F
    masked, n, pos = bool(b1 & 0x80), b1 & 0x7F, 2
    if n == 126:
        if len(buf) < 4:
            return None
        n, pos = struct.unpack("!H", bytes(buf[2:4]))[0], 4
    elif n == 127:
        if len(buf) < 10:
            return None
        n, pos = struct.unpack("!Q", bytes(buf[2:10]))[0], 10
    key = b""
    if masked:
        if len(buf) < pos + 4:
            return None
        key, pos = bytes(buf[pos : pos + 4]), pos + 4
    if len(buf) < pos + n:
        return None
    payload = bytes(buf[pos : pos + n])
    if masked:
        payload = _mask(payload, key)
    return fin, opcode, payload, pos + n


class WebSocket:
    def __init__(self, url: str, headers: dict[str, str] | None = None,
                 connect_timeout: float = 15.0) -> None:
        sp = urlsplit(url)
        if sp.scheme not in ("ws", "wss") or not sp.hostname:
            raise KglError("WebSocket URL must start with ws:// or wss://")
        secure = sp.scheme == "wss"
        port = sp.port or (443 if secure else 80)
        path = (sp.path or "/") + (f"?{sp.query}" if sp.query else "")
        self.closed = False
        self._buf = bytearray()
        self._fragments: list[bytes] = []
        self.write_deadline: float | None = None
        self._send_lock = threading.Lock()
        self.pending_ping: bytes | None = None
        deadline = time.monotonic() + connect_timeout
        sock = None
        try:
            sock = socket.create_connection((sp.hostname, port), timeout=connect_timeout)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            if secure:
                ctx = ssl.create_default_context(cafile=os.environ.get("KGL_CA_FILE") or None)
                sock.settimeout(self._remaining(deadline))
                sock = ctx.wrap_socket(sock, server_hostname=sp.hostname)
        except OSError as e:
            if sock is not None:
                sock.close()
            raise ConnectionLost(f"cannot connect to {sp.hostname}:{port}: {e}") from e
        self._sock = sock
        try:
            self._handshake(sp.hostname, port, secure, path, headers or {}, deadline)
        except Exception:
            self._sock.close()
            raise

    @staticmethod
    def _remaining(deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("WebSocket deadline exceeded")
        return remaining

    def _handshake(self, host: str, port: int, secure: bool, path: str,
                   headers: dict[str, str], deadline: float) -> None:
        key = base64.b64encode(os.urandom(16)).decode()
        default = 443 if secure else 80
        lines = [
            f"GET {path} HTTP/1.1",
            f"Host: {host}" + ("" if port == default else f":{port}"),
            "Upgrade: websocket", "Connection: Upgrade",
            f"Sec-WebSocket-Key: {key}", "Sec-WebSocket-Version: 13",
        ]
        lines += [f"{k}: {v}" for k, v in headers.items()]
        try:
            self._sock.settimeout(self._remaining(deadline))
            self._sock.sendall(("\r\n".join(lines) + "\r\n\r\n").encode())
            while b"\r\n\r\n" not in self._buf:
                self._fill(deadline)
        except TimeoutError as e:
            raise ConnectionLost("WebSocket handshake timed out") from e
        except OSError as e:
            raise ConnectionLost(f"WebSocket handshake failed: {e}") from e
        head, _, rest = bytes(self._buf).partition(b"\r\n\r\n")
        self._buf = bytearray(rest)
        status_line, *hdr_lines = head.decode("latin-1").split("\r\n")
        parts = status_line.split(" ", 2)
        code = parts[1] if len(parts) > 1 else "?"
        if code != "101":
            raise ConnectionLost(f"WebSocket upgrade refused (HTTP {code})")
        hdrs = {h.split(":", 1)[0].strip().lower(): h.split(":", 1)[1].strip()
                for h in hdr_lines if ":" in h}
        expect = base64.b64encode(hashlib.sha1((key + GUID).encode()).digest()).decode()
        if hdrs.get("sec-websocket-accept") != expect:
            raise ConnectionLost("bad Sec-WebSocket-Accept from server")

    def _fill(self, deadline: float | None) -> None:
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
            raise TimeoutError("timed out")
        try:
            self._sock.settimeout(remaining)
            chunk = self._sock.recv(65536)
        except TimeoutError:
            raise
        except OSError as e:
            self.closed = True
            raise ConnectionLost(f"connection error: {e}") from e
        if not chunk:
            self.closed = True
            raise ConnectionLost("connection closed by peer (EOF)")
        self._buf += chunk

    def recv(self, timeout: float | None = None) -> str:
        """Return the next text message. Raises TimeoutError / ConnectionLost."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if deadline is not None:
                self._remaining(deadline)
            frame = parse_frame(self._buf)
            if frame is None:
                self._fill(deadline)
                continue
            fin, op, payload, used = frame
            del self._buf[:used]
            if op == OP_PING:
                self._send(OP_PONG, payload, timeout=30 if deadline is None else self._remaining(deadline))
                continue
            if op == OP_PONG:
                if payload == self.pending_ping:
                    self.pending_ping = None
                continue
            if op == OP_CLOSE:
                self.close()
                raise ConnectionLost("connection closed by peer")
            if op in (OP_TEXT, OP_BIN):
                self._fragments = [payload]
            elif op == OP_CONT:
                self._fragments.append(payload)
            if fin:
                message = b"".join(self._fragments)
                self._fragments.clear()
                return message.decode("utf-8", "replace")

    def _send(self, opcode: int, payload: bytes, *, timeout: float = 30) -> None:
        if self.closed:
            raise ConnectionLost("connection is closed")
        frame = encode_frame(opcode, payload)
        with self._send_lock:
            if self.write_deadline is not None:
                timeout = min(timeout, self.write_deadline - time.monotonic())
            if timeout <= 0:
                raise TimeoutError("write deadline exceeded before sending")
            try:
                self._sock.settimeout(timeout)
                self._sock.sendall(frame)
            except OSError as e:
                self.closed = True
                raise ConnectionLost(f"send failed, execution outcome unknown: {e}", sent=True) from e

    def send(self, text: str) -> None:
        self._send(OP_TEXT, text.encode("utf-8"))

    def ping(self, timeout: float = 30) -> None:
        self.pending_ping = os.urandom(8)
        self._send(OP_PING, self.pending_ping, timeout=timeout)

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            try:
                self._sock.settimeout(0.05)
                self._sock.sendall(encode_frame(OP_CLOSE, b""))
            except OSError:
                pass
        try:
            self._sock.close()
        except OSError:
            pass
