import contextlib
import io
import os
from pathlib import Path
import socket
import threading
import time
import unittest
from unittest.mock import Mock, patch

from kgl import remote
from kgl.errors import ConnectionLost, ExecutionError
from kgl.serving import ServeManager, infer_chat
from kgl.ws import WebSocket, OP_TEXT, OP_CONT, OP_PING, encode_frame
from .support import FakeEnvTest


class TransportTests(unittest.TestCase):
    def test_fragment_survives_timeout_and_ping(self):
        ws = WebSocket.__new__(WebSocket)
        ws._buf = bytearray(encode_frame(OP_TEXT, b"a", fin=False))
        ws._fragments = []
        with patch.object(ws, "_fill", side_effect=TimeoutError), self.assertRaises(TimeoutError):
            ws.recv(.1)
        ws._buf.extend(encode_frame(OP_PING, b"") + encode_frame(OP_CONT, b"b"))
        with patch.object(ws, "_send"):
            self.assertEqual(ws.recv(1), "ab")

    def test_send_failure_after_write_is_uncertain(self):
        ws = WebSocket.__new__(WebSocket)
        ws.closed = False
        ws.write_deadline = None
        ws._send_lock = threading.Lock()
        ws._sock = Mock()
        ws._sock.sendall.side_effect = OSError("lost")
        with self.assertRaises(ConnectionLost) as caught:
            ws.send("code")
        self.assertTrue(caught.exception.sent)

    def test_eof_does_not_complete_stream(self):
        import base64
        payload = base64.b64encode(b'{"stream":true}').decode()
        for body in (b"", b'data: {"choices":[]}\n\n'):
            output = io.StringIO()
            with patch("urllib.request.urlopen", return_value=io.BytesIO(body)), contextlib.redirect_stdout(output):
                exec(remote.render(remote.INFER, PORT=8000, TIMEOUT=1, PAYLOAD=payload), {})
            result = remote.last_marker(output.getvalue())
            self.assertEqual(result["status"], 0)
            self.assertIn("before [DONE]", result["error"])


class RecoveryTests(FakeEnvTest):
    def tearDown(self):
        try:
            if hasattr(self, "mgr"):
                self.mgr.stop()
        finally:
            super().tearDown()

    def test_lost_stream_does_not_repeat(self):
        mgr = ServeManager(self.rt, 21000 + os.getpid() % 1000)
        mgr.start("echo", "echo", wait_timeout=15)
        self.mgr = mgr
        payload = {"messages": [{"role": "user", "content": "one two three"}], "stream": True}
        def disconnect(chunk):
            if chunk["choices"][0]["delta"].get("content"):
                self.rt._channel.ws._sock.shutdown(socket.SHUT_RDWR)
        before = len(self.server.executed)
        with self.assertRaises(ConnectionLost):
            infer_chat(self.rt, mgr.port, payload, disconnect, timeout=5)
        self.assertEqual(len(self.server.executed) - before, 1)
        self.assertEqual(infer_chat(self.rt, mgr.port, payload).text, "echo: one two three")

    def test_stale_pid_and_config_conflict(self):
        mgr = ServeManager(self.rt, 22000 + os.getpid() % 1000)
        directory = Path(mgr.dir)
        directory.mkdir()
        (directory / "serve.pid").write_text(str(os.getpid()))
        with self.assertRaisesRegex(ExecutionError, "stale PID"):
            mgr.stop()
        (directory / "serve.pid").unlink()
        health = mgr.start("echo", "echo", wait_timeout=15)["health"]
        self.mgr = mgr
        self.assertEqual(health["device"], "cpu")
        with self.assertRaisesRegex(ExecutionError, "different server instance"):
            mgr.stop("previous-instance")
        with self.assertRaisesRegex(ExecutionError, "configuration conflicts"):
            mgr.start("other", "transformers")
        self.assertTrue(mgr.stop(health["instance"])["stopped"])
        self.assertTrue(mgr.stop()["stopped"])

    def test_retry_only_before_submission(self):
        path = Path(self.tmp) / "writes"
        code = f'from pathlib import Path; p = Path({str(path)!r}); p.open("a").write("x")'
        channel = self.rt.client.open_channel(self.session.kernel_id)
        with patch.object(self.rt.client, "open_channel", side_effect=[ConnectionLost("unopened"), channel]):
            self.rt.run(code)
        self.assertEqual(path.read_text(), "x")
        original = channel.ws.send

        def delivered_then_lost(payload):
            original(payload)
            deadline = time.monotonic() + 2
            while path.read_text() == "x" and time.monotonic() < deadline:
                time.sleep(.01)
            raise ConnectionLost("ambiguous write", sent=True)

        with patch.object(channel.ws, "send", side_effect=delivered_then_lost):
            with self.assertRaisesRegex(ConnectionLost, "outcome unknown"):
                self.rt.run(code)
        self.assertEqual(path.read_text(), "xx")
        self.assertEqual(self.server.executed.count(code), 2)

    def test_queue_deadline(self):
        with self.rt._lock:
            errors = []
            def run():
                try:
                    self.rt.run("print(1)", total_timeout=.1)
                except TimeoutError as e:
                    errors.append(str(e))
            thread = threading.Thread(target=run)
            thread.start()
            thread.join(1)
            self.assertEqual(len(errors), 1)
            self.assertIn("kernel busy", errors[0])


class GenerationTests(unittest.TestCase):
    def test_worker_exception_reaches_caller_and_releases_lock(self):
        from queue import Queue
        from types import SimpleNamespace
        from kgl.remote_server import TransformersBackend

        class Streamer:
            def __init__(self, *args, **kwargs):
                self.queue = Queue()

            def end(self):
                self.queue.put(None)

            def __iter__(self):
                return iter(lambda: self.queue.get(timeout=1), None)

        backend = TransformersBackend("test", "float16")
        backend.model = Mock()
        backend.model.generate.side_effect = RuntimeError("test worker failed")
        backend.tok = Mock()
        backend.tok.return_value.to.return_value = {}
        with patch.dict("sys.modules", transformers=SimpleNamespace(TextIteratorStreamer=Streamer)):
            with self.assertRaisesRegex(RuntimeError, "test worker failed"):
                list(backend.generate([], 1, 0, 1))
            backend.model.generate.side_effect = lambda **opts: opts["streamer"].end()
            self.assertEqual(list(backend.generate([], 1, 0, 1)), [])
