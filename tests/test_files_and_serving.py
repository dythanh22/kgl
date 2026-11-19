import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from http.server import ThreadingHTTPServer
from importlib.util import find_spec
import io
import json
from pathlib import Path
import os
import threading
import unittest
import urllib.request

from kgl.errors import KglError
from kgl.files import download_file, upload_file
from kgl.remote import SHELL, render
from kgl.remote_server import ImageBackend, State, make_handler
from kgl.serving import ServeManager, infer_chat, make_proxy_server

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
        srv = make_proxy_server(self.rt, self.port, "echo", listen_port=0, api_key="k", demo=True)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        try:
            with urllib.request.urlopen(base + "/") as page:
                self.assertEqual(page.headers.get_content_type(), "text/html")
                self.assertIn(b"Checking model", page.read())

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
            self.assertEqual(models["data"][0]["id"], "echo")
        finally:
            srv.shutdown()
            srv.server_close()

    @unittest.skipUnless(find_spec("PIL"), "Pillow is required to exercise image decoding")
    def test_image_proxy_checks_input_and_keeps_serving(self):
        from PIL import Image

        backend = ImageBackend("classifier", None, False)
        backend.image = Image
        backend.device = "cpu"
        backend.model = object()
        backend.inference_mode = nullcontext
        entered = threading.Event()
        release = threading.Event()

        def predict(model, image):
            self.assertIs(model, backend.model)
            self.assertEqual(image.mode, "RGB")
            entered.set()
            if not release.wait(10):
                raise TimeoutError("test did not release worker")
            return [{"label": "red" if image.getpixel((0, 0))[0] else "blue", "score": 1.0}]

        backend.predict = predict
        state = State()
        state.status = "ready"
        self.session.serve = {"backend": backend.name, "model": "classifier", "instance": "test-instance"}
        worker = ThreadingHTTPServer(("127.0.0.1", 0),
                                     make_handler(backend, "classifier", state, "test-instance"))
        proxy = make_proxy_server(self.rt, worker.server_port, "classifier", listen_port=0, demo=True)
        for server in (worker, proxy):
            threading.Thread(target=server.serve_forever, daemon=True).start()
            self.addCleanup(server.server_close)
            self.addCleanup(server.shutdown)
        self.addCleanup(release.set)

        def post(body):
            req = urllib.request.Request(f"http://127.0.0.1:{proxy.server_port}/predict",
                                         json.dumps(body).encode(), {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=15) as reply:
                return json.load(reply)["predictions"]

        def payload(color):
            stream = io.BytesIO()
            Image.new("RGB", (4, 4), color).save(stream, format="PNG")
            return {"image": base64.b64encode(stream.getvalue()).decode()}

        for bad in ("not base64!", "AAAA", base64.b64encode(b"x" * (350 * 1024 + 1)).decode()):
            with self.assertRaises(urllib.error.HTTPError) as cm:
                post({"image": bad})
            self.assertEqual(cm.exception.code, 400)

        with ThreadPoolExecutor(max_workers=1) as executor:
            first = executor.submit(post, payload("red"))
            try:
                self.assertTrue(entered.wait(5))
                with self.assertRaises(urllib.error.HTTPError) as cm:
                    post(payload("blue"))
                self.assertEqual(cm.exception.code, 503)
            finally:
                release.set()
            self.assertEqual(first.result()[0]["label"], "red")
        self.assertEqual(post(payload("blue"))[0]["label"], "blue")


if __name__ == "__main__":
    unittest.main()
