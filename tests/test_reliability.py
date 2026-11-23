import contextlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import io
import os
from pathlib import Path
import socket
import threading
import time
import unittest
from unittest.mock import Mock, patch

from kgl import remote
from kgl.core.errors import ConnectionLost, ExecutionError, KernelGone
from kgl.client.files import upload_file
from kgl.transport.jupyter import KernelChannel
from kgl.client.serve_manager import ServeManager
from kgl.client.infer import infer_chat
from kgl.transport.ws import WebSocket, OP_TEXT, OP_CONT, OP_PING, encode_frame
from .support import FakeEnvTest


class TransportTests(unittest.TestCase):
    def test_expired_write_has_not_sent(self):
        ws = WebSocket.__new__(WebSocket)
        ws.closed = False
        ws.write_deadline = time.monotonic() - 1
        ws._send_lock = threading.Lock()
        ws._sock = Mock()
        with self.assertRaisesRegex(TimeoutError, "before sending"):
            ws.send("code")
        ws._sock.sendall.assert_not_called()

    def test_missing_pong_fails_with_uncertain_delivery(self):
        ws = Mock()
        ws.recv.side_effect = TimeoutError
        ws.ping.side_effect = lambda timeout: setattr(ws, "pending_ping", b"pending")
        channel = KernelChannel(ws, "test")
        with self.assertRaisesRegex(ConnectionLost, "pong deadline") as caught:
            channel.execute("print(1)", total_timeout=.5, ping_interval=.02)
        self.assertTrue(caught.exception.sent)
        ws.send.assert_called_once()

    def test_parentless_kernel_death_ends_execution(self):
        ws = Mock()
        ws.recv.return_value = '{"msg_type":"status","content":{"execution_state":"dead"}}'
        with self.assertRaises(KernelGone):
            KernelChannel(ws, "test").execute("print(1)", total_timeout=.5)

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
                exec(remote.render(remote.INFER, PORT=8000, TIMEOUT=1, PAYLOAD=payload, OPTIONS={}), {})
            result = remote.last_marker(output.getvalue())
            self.assertEqual(result["status"], 0)
            self.assertIn("before [DONE]", result["error"])


class RecoveryTests(FakeEnvTest):
    def test_streaming_without_retaining_output(self):
        code = "print('x' * 20000)"
        chunks = []
        result = self.rt.run(code, lambda name, text: chunks.append(text), capture_output=False)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.text, "")
        self.assertEqual("".join(chunks), "x" * 20000 + "\n")
        self.assertEqual(self.rt.run(code).text, "".join(chunks))

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

    def test_idle_channel_reopens_before_submitting(self):
        self.rt.run("print(1)")
        old = self.rt._channel
        old.ws._sock.shutdown(socket.SHUT_RDWR)
        self.rt._last_used = time.monotonic() - 21
        self.assertEqual(self.rt.run("print(2)").text.strip(), "2")
        self.assertIsNot(self.rt._channel, old)
        self.assertEqual(self.server.executed.count("print(2)"), 1)

    def test_failed_upload_preserves_destination(self):
        source = Path(self.tmp) / "source"
        source.write_bytes(b"A" * 3000)
        target = Path(self.workdir) / "target__DATA__"
        target.write_bytes(b"original")

        def disconnect_after_chunk(done, total):
            if done < total:
                raise ConnectionLost("test interrupted upload")

        with self.assertRaisesRegex(ConnectionLost, "interrupted upload"):
            upload_file(self.rt, source, target.name, disconnect_after_chunk)
        self.assertEqual(target.read_bytes(), b"original")
        self.assertEqual(list(target.parent.glob(".kgl-upload-*")), [])
        upload_file(self.rt, source, target.name)
        self.assertEqual(target.read_bytes(), source.read_bytes())

    def test_concurrent_uploads_and_repeated_chunk_keep_complete_destination(self):
        first, second = (Path(self.tmp) / name for name in ("first", "second"))
        first.write_bytes(b"A" * 2100)
        second.write_bytes(b"B" * 3100)
        target = Path(self.workdir) / "shared"
        barrier = threading.Barrier(2, timeout=10)
        original = self.rt.call

        def repeat_first_chunk(template, **values):
            reply = original(template, **values)
            if template == remote.UPLOAD_CHUNK and values["OFF"] == 0:
                self.assertEqual(original(template, **values), reply)
            return reply

        def sync(done, total):
            if done == 1024:
                barrier.wait()

        with patch.object(self.rt, "call", side_effect=repeat_first_chunk), ThreadPoolExecutor(2) as pool:
            pending = [pool.submit(upload_file, self.rt, source, target.name, sync)
                       for source in (first, second)]
            self.assertEqual([task.result()["size"] for task in pending], [2100, 3100])
        self.assertIn(target.read_bytes(), (first.read_bytes(), second.read_bytes()))
        self.assertEqual(list(target.parent.glob(".kgl-upload-*")), [])

    def test_cancellation_via_independent_kernel_releases_worker(self):
        self.mgr = ServeManager(self.rt, 23000 + os.getpid() % 1000)
        health = self.mgr.start("echo", "echo", wait_timeout=15)["health"]
        self.rt.settings = replace(self.settings, control_kernel_id=self.server.new_kernel()["id"])
        port, instance = self.mgr.port, health["instance"]
        payload = {"messages": [{"role": "user", "content": "word " * 900}], "stream": True}

        def disconnect(chunk):
            if chunk["choices"][0]["delta"].get("content"):
                raise BrokenPipeError("test client left")

        with self.assertLogs("kgl.client.infer", level="INFO") as logs, self.assertRaises(BrokenPipeError):
            infer_chat(self.rt, port, payload, disconnect, timeout=8, expected_instance=instance)
        self.assertIn("remote cancellation confirmed", "\n".join(logs.output))
        self.assertEqual(self.server.interrupts, 0)
        short = {"messages": [{"role": "user", "content": "next"}]}
        self.assertEqual(infer_chat(self.rt, port, short, expected_instance="replaced").status, 409)
        self.assertEqual(infer_chat(self.rt, port, short, expected_instance=instance).text, "echo: next")


class GenerationTests(unittest.TestCase):
    def test_balanced_placement_reserves_vram_and_requires_cuda_only(self):
        from types import SimpleNamespace
        from kgl.vm.backends import TransformersBackend

        gib = 1024**3
        cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 2,
                               mem_get_info=lambda _: (15 * gib, 15 * gib))
        torch = SimpleNamespace(cuda=cuda, float16="fp16", bfloat16="bf16", float32="fp32")
        model = Mock()
        model.get_input_embeddings.return_value.weight.device = "cuda:0"
        token_factory, model_factory = Mock(), Mock()
        model_factory.from_pretrained.return_value = model
        transformers = SimpleNamespace(AutoModelForCausalLM=model_factory, AutoTokenizer=token_factory)
        with patch.dict("sys.modules", torch=torch, transformers=transformers):
            backend = TransformersBackend("test", "float16", device_map="balanced")
            model.hf_device_map = {"first": 0, "second": "cuda:1"}
            backend.load()
            model_factory.from_pretrained.assert_called_once_with(
                "test", torch_dtype="fp16", device_map="balanced", max_memory={0: 14 * gib, 1: 14 * gib})
            model.to.assert_not_called()
            self.assertEqual((backend.device, backend.devices), ("cuda:0", ["cuda:0", "cuda:1"]))
            model.hf_device_map["tail"] = "cpu"
            with self.assertRaisesRegex(RuntimeError, "multiple CUDA GPUs"):
                TransformersBackend("test", "float16", device_map="balanced").load()
            placement = {"first": 0, "second": 1}
            model.hf_device_map = placement
            backend = TransformersBackend("test", "float16", device_map='{"first":0,"second":1}')
            backend.load()
            model_factory.from_pretrained.assert_called_with("test", torch_dtype="fp16", device_map=placement)
            with self.assertRaisesRegex(ValueError, "CUDA GPU indices"):
                TransformersBackend("test", "float16", device_map='{"first":0,"second":"disk"}')

    def test_worker_exception_reaches_caller_and_releases_lock(self):
        from queue import Queue
        from types import SimpleNamespace
        from kgl.vm.backends import TransformersBackend

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
            backend.prefill_chunk_size = 256
            inputs = {"input_ids": SimpleNamespace(shape=(1, 512))}
            self.assertEqual(list(backend.generate([], 1, 0, 1, inputs=inputs)), [])
            self.assertEqual(backend.model.generate.call_args.kwargs["prefill_chunk_size"], 256)
