"""Exercise the gateway request parser and backend dispatch in memory."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from kgl.vm.backends import EchoBackend, PythonBackend
from kgl.vm.gateway import make_handler
from kgl.vm.state import State


class GatewayTests(unittest.TestCase):
    def request(self, backend, path, payload, limit=512):
        state = State()
        state.status = "ready"
        data = json.dumps(payload).encode()
        conn = Mock()
        conn.makefile.return_value = io.BytesIO(
            f"POST {path} HTTP/1.0\r\nContent-Length: {len(data)}\r\nX-Kgl-Instance: test\r\n\r\n".encode() + data)
        handler = make_handler(backend, "example", state, "test", request_bytes=limit)
        with patch.object(handler, "log_message"):
            handler(conn, ("127.0.0.1", 12345), Mock())
        response = b"".join(call.args[0] for call in conn.sendall.call_args_list)
        head, body = response.split(b"\r\n\r\n", 1)
        self.assertIsNone(state.active)
        return int(head.split()[1]), json.loads(body)

    def test_python_module_load_dispatch_and_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            module = Path(tmp) / "handler.py"
            module.write_text("def load(path, device): return path\n"
                              "def handle(model, req): return [model, req]\n")
            backend = PythonBackend("weights", str(module), False)
            with patch.dict("sys.modules", torch=None), patch("sys.path", []):
                backend.load()
            self.assertEqual(self.request(backend, "/internal/handle", [1]), (200, ["weights", [1]]))
            self.assertEqual(self.request(backend, "/internal/handle", "long", limit=3)[0], 413)
            self.assertEqual(self.request(backend, "/v1/chat/completions", {})[0], 404)

    def test_chat_dispatch_and_same_byte_limit(self):
        messages = {"messages": [{"role": "user", "content": "hello"}]}
        status, body = self.request(EchoBackend(), "/v1/chat/completions", messages)
        self.assertEqual(status, 200)
        self.assertEqual(body["choices"][0]["message"]["content"], "echo: hello")
        self.assertEqual(self.request(EchoBackend(), "/v1/chat/completions", messages, limit=3)[0], 413)
