from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import os
import threading
import unittest
from unittest.mock import patch
import urllib.request

from kgl.core.state import ServeRecord
from kgl.core.errors import KernelGone, KglError
from kgl.client.files import download_file, upload_file
from kgl.remote import SHELL, render
from kgl.vm.backends import PythonBackend
from kgl.vm.state import State
from kgl.vm.gateway import make_handler
from kgl.client.serve_manager import ServeManager
from kgl.client.infer import InferResult, infer_chat
from kgl.proxy.handler import make_proxy_server

from .support import FakeEnvTest


class FileTests(FakeEnvTest):
    def test_upload_download_roundtrip_multi_chunk(self):
        src = os.path.join(self.tmp, "src.bin")
        data = os.urandom(5000)  # chunk_kb=1 -> 5 chunks
        Path(src).write_bytes(data)
        r = upload_file(self.rt, src, "sub/dir/f.bin")
        self.assertEqual(r["size"], 5000)
        self.assertEqual((Path(self.workdir) / "sub/dir/f.bin").read_bytes(), data)
        dst = os.path.join(self.tmp, "out", "back.bin")
        download_file(self.rt, "sub/dir/f.bin", dst)
        self.assertEqual(Path(dst).read_bytes(), data)

    def test_empty_file_and_missing(self):
        src = os.path.join(self.tmp, "e")
        open(src, "wb").close()
        upload_file(self.rt, src, "e")
        with self.assertRaises(KglError):
            download_file(self.rt, "nope", os.path.join(self.tmp, "x"))

    def test_shell_mode_returns_rc(self):
        res = self.rt.run(render(SHELL, CWD=self.workdir, CMD="echo hi; exit 3"))
        self.assertIn("hi", res.text)
        self.assertIn('"rc": 3', res.text)


class ServeTests(FakeEnvTest):
    def setUp(self):
        super().setUp()
        self.port = 18000 + os.getpid() % 1000
        self.mgr = ServeManager(self.rt, self.port)

    def tearDown(self):
        try:
            self.mgr.stop()
        finally:
            super().tearDown()

    def test_start_infer_stream_and_nonstream_stop(self):
        self.mgr.start("echo", "echo", wait=True, wait_timeout=30)
        r = infer_chat(self.rt, self.port, {"messages": [{"role": "user", "content": "hello world"}]})
        self.assertEqual((r.status, r.text), (200, "echo: hello world"))
        chunks = []
        r = infer_chat(self.rt, self.port, {"messages": [{"role": "user", "content": "a b c"}],
                                            "stream": True}, chunks.append)
        self.assertEqual(r.text, "echo: a b c")
        self.assertGreater(len(chunks), 3)
        self.assertIsNotNone(r.ttft_s)
        # idempotent start
        self.assertTrue(self.rt.call(__import__("kgl.remote", fromlist=["x"]).SERVE_START,
                                     DIR=self.mgr.dir, ARGS=[])["already_running"])
        self.assertTrue(self.mgr.stop()["stopped"])
        self.assertEqual(self.mgr.health()["state"], "down")

    def test_bad_request_reports_400(self):
        self.mgr.start("echo", "echo", wait=True, wait_timeout=30)
        r = infer_chat(self.rt, self.port, {"nomessages": 1})
        self.assertEqual(r.status, 400)

    def test_down_server_gives_error_not_crash(self):
        r = infer_chat(self.rt, self.port, {"messages": [{"role": "user", "content": "x"}]})
        self.assertEqual(r.status, 0)
        self.assertTrue(r.error)

    def test_local_openai_proxy(self):
        self.mgr.start("echo", "echo", wait=True, wait_timeout=30)
        assets = Path(__file__).resolve().parents[1] / "examples/chat-demo"
        srv = make_proxy_server(self.rt, self.port, "echo", listen_port=0, api_key="k", static=assets)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        try:
            with urllib.request.urlopen(base + "/") as page:
                self.assertEqual(page.headers.get_content_type(), "text/html")
                self.assertEqual(page.read(), (assets / "demo.html").read_bytes())

            def post(body, key="k", headers=None):
                req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                             {"Content-Type": "application/json",
                                              "Authorization": f"Bearer {key}", **(headers or {})})
                return urllib.request.urlopen(req, timeout=30)

            msg = {"messages": [{"role": "user", "content": "ping"}]}
            self.assertEqual(json.loads(post(msg).read())["choices"][0]["message"]["content"], "echo: ping")
            lines = [l.decode().strip() for l in post({**msg, "stream": True}) if l.strip()]
            self.assertEqual(lines[-1], "data: [DONE]")
            text = "".join(json.loads(l[6:])["choices"][0]["delta"].get("content", "")
                           for l in lines[:-1])
            self.assertEqual(text, "echo: ping")
            with self.assertRaises(urllib.error.HTTPError) as cm:
                post(msg, key="wrong")
            self.assertEqual(cm.exception.code, 401)
            before = len(self.server.executed)
            invalid_headers = (
                ({"Host": "example.invalid"}, 400),
                ({"Origin": "http://example.invalid"}, 403),
                ({"Content-Type": "text/plain"}, 415),
            )
            for headers, status in invalid_headers:
                with self.subTest(headers=headers), self.assertRaises(urllib.error.HTTPError) as cm:
                    post(msg, headers=headers)
                self.assertEqual(cm.exception.code, status)
            with self.assertRaises(urllib.error.HTTPError) as cm:
                post({**msg, "max_completion_tokens": 10, "max_tokens": 10, "stream": True})
            self.assertEqual(cm.exception.code, 400)
            self.assertEqual(len(self.server.executed), before)
            ready = urllib.request.Request(base + "/ready", headers={"Authorization": "Bearer k"})
            with urllib.request.urlopen(ready) as reply:
                self.assertEqual(json.load(reply)["health"]["backend"], "echo")
            kernel = self.server.kernels[self.session.kernel_id]
            kernel["execution_state"] = "busy"
            try:
                with self.assertRaises(urllib.error.HTTPError) as cm:
                    urllib.request.urlopen(ready)
                busy = json.load(cm.exception)
                self.assertEqual((cm.exception.code, busy["state"]), (503, "busy"))
                self.assertEqual(cm.exception.headers["Retry-After"], "1")
                self.assertEqual(busy["last_health"]["backend"], "echo")
                self.assertGreaterEqual(busy["health_age_s"], 0)
            finally:
                kernel["execution_state"] = "idle"
            self.mgr.stop()
            with urllib.request.urlopen(base + "/health") as reply:
                self.assertEqual(reply.status, 200)
            with self.assertRaises(urllib.error.HTTPError) as cm:
                urllib.request.urlopen(ready)
            self.assertEqual(cm.exception.code, 503)
            models = json.loads(urllib.request.urlopen(urllib.request.Request(
                base + "/v1/models", headers={"Authorization": "Bearer k"})).read())
            self.assertEqual([model["id"] for model in models["data"]], ["echo"])
        finally:
            srv.shutdown()
            srv.server_close()

    def test_proxy_readiness_states_and_session_estimate(self):
        instance = "current"
        self.session.serve = ServeRecord(port=self.port, backend="echo", model="echo", instance=instance)
        health = {"model": "echo", "backend": "echo", "instance": instance}
        with make_proxy_server(self.rt, self.port, "echo", listen_port=0) as proxy:
            threading.Thread(target=proxy.serve_forever, daemon=True).start()
            url = f"http://127.0.0.1:{proxy.server_port}/ready"

            def get():
                try:
                    response = urllib.request.urlopen(url, timeout=5)
                except urllib.error.HTTPError as exc:
                    response = exc
                with response:
                    return response.status, json.load(response)

            try:
                with patch.object(self.rt.client, "get_kernel", return_value={"execution_state": "busy"}) as kernel, \
                        patch("kgl.proxy.handler.ServeManager.health") as probe:
                    code, body = get()
                    self.assertEqual((code, body["state"], body["approx_session_remaining_s"]), (503, "busy", None))
                    self.assertNotIn("last_health", body)
                    probe.assert_not_called()

                    kernel.return_value = {"execution_state": "idle"}
                    probe.side_effect = lambda **kwargs: {
                        "state": "loading", "health": health, "container_uptime_s": 60,
                    }
                    code, body = get()
                    remaining = self.rt.settings.approx_session_remaining(60)
                    self.assertEqual((code, body["state"], body["approx_session_remaining_s"]),
                                     (503, "loading", remaining))
                    self.assertEqual(body["health"], health)

                    kernel.return_value = {"execution_state": "busy"}
                    code, body = get()
                    self.assertEqual((code, body["state"], body["last_health"]), (503, "busy", health))
                    self.assertGreaterEqual(body["health_age_s"], 0)
                    self.assertLessEqual(body["approx_session_remaining_s"], remaining)
                    self.assertNotIn("health", body)

                    kernel.return_value = {"execution_state": "idle"}
                    probe.side_effect = lambda **kwargs: {"state": "ready", "health": health}
                    code, body = get()
                    self.assertEqual((code, body["state"], body["approx_session_remaining_s"]), (200, "ready", None))

                    probe.side_effect = KernelGone("kernel disconnected")
                    code, body = get()
                    self.assertEqual((code, body["state"], body["approx_session_remaining_s"]),
                                     (503, "connection_lost", None))
                    self.assertEqual(body["error"]["type"], "runtime_unavailable")

                    kernel.side_effect = KernelGone("kernel disconnected")
                    code, body = get()
                    self.assertEqual((code, body["state"]), (503, "connection_lost"))
                    self.assertEqual(body["error"]["type"], "kernel_rest_unavailable")
            finally:
                proxy.shutdown()

    def test_chat_proxy_busy_does_not_interrupt_active_stream(self):
        release = threading.Event()

        def infer(rt, port, payload, on_chunk, **kwargs):
            if on_chunk:
                on_chunk({"choices": [{"delta": {"content": "first"}}]})
                if not release.wait(10):
                    raise TimeoutError("test did not release stream")
                on_chunk({"choices": [{"delta": {}, "finish_reason": "stop"}]})
            return InferResult(status=200, body={"answer": "ok"})

        with make_proxy_server(self.rt, self.port, "echo", listen_port=0) as proxy:
            threading.Thread(target=proxy.serve_forever, daemon=True).start()

            def post(stream=False):
                url = f"http://127.0.0.1:{proxy.server_port}/v1/chat/completions"
                payload = {"messages": [{"role": "user", "content": "ping"}], "stream": stream}
                return urllib.request.urlopen(urllib.request.Request(
                    url, json.dumps(payload).encode(), {"Content-Type": "application/json"}), timeout=5)

            try:
                with patch("kgl.proxy.handler.infer_chat", side_effect=infer) as upstream:
                    with post(stream=True) as first:
                        try:
                            self.assertIn(b"first", first.readline())
                            with self.assertRaises(urllib.error.HTTPError) as caught:
                                post()
                            with caught.exception as busy:
                                self.assertEqual(busy.code, 503)
                                self.assertEqual(busy.headers["Retry-After"], "1")
                                self.assertEqual(json.load(busy)["error"]["type"], "worker_busy")
                            self.assertEqual(upstream.call_count, 1)
                        finally:
                            release.set()
                        tail = first.read()
                        self.assertIn(b'"finish_reason": "stop"', tail)
                        self.assertTrue(tail.endswith(b"data: [DONE]\n\n"))
                    with post() as third:
                        self.assertEqual(json.load(third)["answer"], "ok")
                    self.assertEqual(upstream.call_count, 2)
            finally:
                proxy.shutdown()

    def test_python_handler_validates_and_serializes_requests(self):
        backend = PythonBackend("classifier", None, False)
        backend.model = object()
        entered = threading.Event()
        release = threading.Event()

        def handle(model, request):
            if "color" not in request:
                raise ValueError("color is required")
            entered.set()
            if not release.wait(10):
                raise TimeoutError("test did not release worker")
            return {"predictions": [{"label": request["color"]}]}

        backend.invoke = handle
        state = State()
        state.status = "ready"
        self.session.serve = ServeRecord(port=self.port, backend=backend.name, model="classifier", instance="test-instance")
        worker = ThreadingHTTPServer(("127.0.0.1", 0),
                                     make_handler(backend, "classifier", state, "test-instance"))
        proxy = make_proxy_server(self.rt, worker.server_port, "classifier", listen_port=0, static=Path(__file__).resolve().parents[1] / "examples/chat-demo")
        for server in (worker, proxy):
            threading.Thread(target=server.serve_forever, daemon=True).start()
            self.addCleanup(server.server_close)
            self.addCleanup(server.shutdown)
        self.addCleanup(release.set)

        def post(body):
            req = urllib.request.Request(f"http://127.0.0.1:{proxy.server_port}/handle",
                                         json.dumps(body).encode(), {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=15) as reply:
                return json.load(reply)["predictions"]

        with self.assertRaises(urllib.error.HTTPError) as cm:
            post({})
        self.assertEqual(cm.exception.code, 400)

        with ThreadPoolExecutor(max_workers=1) as executor:
            first = executor.submit(post, {"color": "red"})
            try:
                self.assertTrue(entered.wait(5))
                with self.assertRaises(urllib.error.HTTPError) as cm:
                    post({"color": "blue"})
                self.assertEqual(cm.exception.code, 503)
            finally:
                release.set()
            self.assertEqual(first.result()[0]["label"], "red")
        self.assertEqual(post({"color": "blue"})[0]["label"], "blue")


if __name__ == "__main__":
    unittest.main()
