"""Exercise the llama gateway with an upstream subprocess and real HTTP/SSE."""
from contextlib import redirect_stderr, redirect_stdout
import hashlib
from importlib.util import find_spec
import io
import json
import os
from pathlib import Path
import socket
import struct
import threading
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request

from kgl.client.model_fetch import vm_operation
from kgl.cli import main
from kgl.client.serve_manager import ServeManager
from kgl.client.infer import infer_chat
from kgl.proxy.handler import make_proxy_server
from kgl.protocol import chat_args
from kgl.core.state import ServeRecord, SessionStore
from kgl.vm import llama_tools
from .support import FakeEnvTest

UPSTREAM = """
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import select
import socket
import threading
import time

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int)
args, _ = parser.parse_known_args()

busy = threading.Event()
started = time.monotonic()

class Handler(BaseHTTPRequestHandler):
    def reply(self, body, status=200):
        raw = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/health":
            self.reply({}, 503 if time.monotonic() - started < 0.3 else 200)
        elif self.path == "/slots?fail_on_no_slot=1":
            self.reply([{"is_processing": busy.is_set(), "n_ctx": 2048}], 503 if busy.is_set() else 200)

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = req["messages"][-1]["content"]
        finish = "length" if prompt == "short" else "stop"
        text = None if finish == "length" else "answer"
        if prompt == "echo-options":
            fields = ("chat_template_kwargs", "presence_penalty", "repeat_penalty", "response_format",
                      "temperature", "top_p", "top_k", "max_tokens")
            text = json.dumps({key: req[key] for key in fields if key in req})
        usage = {"prompt_tokens": 6, "completion_tokens": 2, "total_tokens": 8}
        timings = {"prompt_per_second": 140.0, "predicted_per_second": 17.0}
        if not req.get("stream"):
            self.reply({"model": "upstream", "choices": [{"message": {"content": text,
                        "reasoning_content": "thought"}, "finish_reason": finish}],
                        "usage": usage, "timings": timings})
            return
        busy.set()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()

            def event(value):
                encoded = json.dumps(value) if value != "[DONE]" else value
                self.wfile.write(("data: " + encoded + "\\n\\n").encode())
                self.wfile.flush()

            event({"model": "upstream", "choices": [{"delta": {"reasoning_content": "thought"}}]})
            if prompt == "crash":
                os._exit(9)
            if prompt == "incomplete":
                return
            if prompt == "cancel":
                if select.select([self.connection], [], [], 5)[0]:
                    self.connection.recv(1, socket.MSG_PEEK)
                return
            if text:
                event({"model": "upstream", "choices": [{"delta": {"content": text}}]})
            event({"choices": [{"delta": {}, "finish_reason": finish}], "timings": timings})
            event({"choices": [], "usage": usage})
            event("[DONE]")
        finally:
            busy.clear()

ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
"""


class LlamaTests(FakeEnvTest):
    def setUp(self):
        super().setUp()
        root = Path(self.workdir)
        model = root / "Qwen3.8-35B-A3B-Q4_K_M.gguf"
        root.mkdir(parents=True, exist_ok=True)
        model.write_bytes(b"GGUF")
        launcher = root / "llama-server"
        launcher.write_text(f"#!{os.sys.executable}\n" + UPSTREAM)
        launcher.chmod(0o700)
        manifest = root / "manifest.json"
        options = {key: "--" + key for key in
                   ("model", "host", "port", "ctx", "gpu_layers", "split_mode", "parallel")}
        manifest.write_text(json.dumps({"argv": ["llama-server"], "options": options}))
        self.model = str(model)
        self.settings_llama = {"llama_manifest": str(manifest), "ctx": 2048,
                               "settings": {"alias": "qwen-local", "quant": "Q4_K_M",
                                            "capabilities": {"reasoning": True, "top_k": True, "enable_thinking": True,
                                                             "presence_penalty": True, "repeat_penalty": True,
                                                             "response_format": True, "json_schema": True},
                                            "defaults": {"max_tokens": 10, "top_k": 20, "timeout": 10}}}
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            self.manager = ServeManager(self.rt, listener.getsockname()[1])
        self.addCleanup(self._stop_if_needed)
        response = self.manager.start(self.model, "llama", options=self.settings_llama, wait_timeout=10)
        self.health = response["health"]
        self.rt.session.serve = ServeRecord(model=self.model, backend="llama", port=self.manager.port,
                                 instance=self.health["instance"], **self.settings_llama["settings"])

    def tearDown(self):
        try:
            self._stop_if_needed()
        finally:
            super().tearDown()

    def _stop_if_needed(self):
        if self.manager is not None:
            self.manager.stop()
            self.manager = None

    def request(self, prompt, stream=True, **kwargs):
        return infer_chat(self.rt, self.manager.port, {
            "messages": [{"role": "user", "content": prompt}], "stream": stream,
        }, timeout=10, **kwargs)

    def test_idempotent_start_stream_and_json(self):
        again = self.manager.start(self.model, "llama", options=self.settings_llama, wait_timeout=10)
        self.assertEqual(again["health"]["instance"], self.health["instance"])
        self.assertEqual(again["health"]["n_ctx"], 2048)
        stream, plain = self.request("question"), self.request("question", stream=False)
        for result in (stream, plain):
            self.assertEqual((result.text, result.reasoning, result.finish_reason), ("answer", "thought", "stop"))
            self.assertEqual(result.usage["total_tokens"], 8)
            self.assertEqual(result.timings["predicted_per_second"], 17)
        self.assertLessEqual(stream.ttft_first_token_s, stream.ttft_first_content_s)
        self.assertIsNone(plain.ttft_first_token_s)
        short = self.request("short")
        self.assertEqual((short.text, short.reasoning, short.finish_reason), ("", "thought", "length"))
        self.assertIsNone(short.ttft_first_content_s)
        self.assertTrue(self.manager.stop(self.health["instance"])["stopped"])
        with self.assertRaises(ProcessLookupError):
            os.kill(self.health["pid"], 0)

    def test_cancel_with_dedicated_kernel_and_recover(self):
        self.rt.session.dedicated_kernel = True
        confirmed = []

        def interrupt(_):
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            self.request("cancel", on_chunk=interrupt, on_cancel=confirmed.append)
        self.assertEqual(confirmed, [True])
        self.assertEqual(self.server.interrupts, 1)
        self.assertEqual(self.request("question").text, "answer")

    def test_partial_stream_and_child_failure_are_reported(self):
        seen = []
        partial = self.request("incomplete", on_chunk=seen.append)
        self.assertEqual(partial.status, 0)
        self.assertIn("before [DONE]", partial.error)
        self.assertEqual(len(seen), 1)
        self.assertEqual(self.request("question").text, "answer")
        failed = self.request("crash")
        self.assertEqual(failed.status, 0)
        health = self.manager.health()["health"]
        self.assertEqual(health["status"], "error")
        self.assertIn("exited (9)", health["error"])

    def test_cli_reports_reasoning_and_rejects_truncation(self):
        SessionStore(Path(self.home)).save(self.rt.session)
        with patch.dict(os.environ, KGL_HOME=self.home, KGL_WORKDIR=self.workdir):
            with redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(main(["infer", "--json", "question"]), 0)
            result = json.loads(stdout.getvalue())
            self.assertEqual(result["reasoning"], "thought")
            self.assertEqual(result["finish_reason"], "stop")
            with redirect_stderr(io.StringIO()) as stderr:
                self.assertEqual(main(["infer", "--json", "short"]), 1)
            self.assertIn("increase --max-tokens", stderr.getvalue())

    def test_cli_start_uses_session_manifest(self):
        self.manager.stop(self.health["instance"])
        self.rt.session.serve = None
        self.rt.session.llama_manifest = self.settings_llama["llama_manifest"]
        store = SessionStore(Path(self.home))
        store.save(self.rt.session)
        with patch.dict(os.environ, KGL_HOME=self.home, KGL_WORKDIR=self.workdir):
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(main(["serve", "start", "--backend", "llama", "--model", self.model,
                                       "--ctx", "2048", "--port", str(self.manager.port), "--wait-timeout", "10"]), 0)
        self.assertEqual(store.load().serve.state, "ready")

    def test_verified_attached_quant_alias_and_same_basename_outside_preset(self):
        self.manager.stop(self.health["instance"])
        self.rt.session.serve = None
        self.rt.session.llama_manifest = self.settings_llama["llama_manifest"]
        store = SessionStore(Path(self.home))
        store.save(self.rt.session)
        path = Path(self.workdir) / "alternate" / "example-IQ4_XS.gguf"
        path.parent.mkdir()
        path.write_bytes(b"GGUF second quant")
        preset = Path(self.home) / "example.json"
        metadata = {"repo": "example/model", "revision": "main", "default_quant": "IQ4_XS",
                    "files": {"IQ4_XS": path.name}, "checksum_file": "SHA256SUMS",
                    "checksums": {"IQ4_XS": hashlib.sha256(path.read_bytes()).hexdigest()},
                    "inputs": {"IQ4_XS": str(path)}, "alias": "portable-local", "context": 4096}
        preset.write_text(json.dumps(metadata))
        args = ["serve", "start", "--preset", str(preset), "--ctx", "2048",
                "--port", str(self.manager.port), "--wait-timeout", "10"]
        with patch.dict(os.environ, KGL_HOME=self.home, KGL_WORKDIR=self.workdir), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as stderr:
            code = main([*args, "--model", str(path)])
            self.assertEqual(code, 0, stderr.getvalue())
            record = store.load().serve
            self.assertEqual((record.alias, record.quant, record.model), ("portable-local", "IQ4_XS", str(path)))
            health = self.manager.health()["health"]
            self.assertEqual((health["alias"], health["quant"], health["n_ctx"]), ("portable-local", "IQ4_XS", 2048))
            self.assertEqual(main([*args, "--model", str(path)]), 0)
            self.assertEqual(store.load().serve.instance, record.instance)
            self.assertEqual(main(["serve", "stop"]), 0)
            other = Path(self.workdir) / path.name
            other.write_bytes(b"GGUF wrong model")
            self.assertEqual(main([*args, "--model", str(other)]), 0)
            self.assertIsNone(store.load().serve.alias)
            self.assertEqual(main(["serve", "stop"]), 0)
            path.write_bytes(b"wrong model at the attached path")
            self.assertNotEqual(main([*args, "--model", str(path)]), 0)
            self.assertIsNone(store.load().serve)

    def test_proxy_preserves_reasoning_and_rejects_unverified_tools(self):
        proxy = make_proxy_server(self.rt, self.manager.port, self.model, listen_port=0)
        thread = threading.Thread(target=proxy.serve_forever, daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{proxy.server_port}/v1/chat/completions"
            message = {"messages": [{"role": "user", "content": "question"}]}
            headers = {"Content-Type": "application/json"}
            with urllib.request.urlopen(urllib.request.Request(base, json.dumps(message).encode(), headers)) as response:
                body = json.load(response)
            self.assertEqual(body["choices"][0]["message"]["reasoning_content"], "thought")
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(urllib.request.Request(base, json.dumps(dict(message, tools=[])).encode(), headers))
            self.assertEqual(caught.exception.code, 400)
            caught.exception.close()
        finally:
            proxy.shutdown()
            proxy.server_close()
            thread.join()

    def test_optional_llama_fields_reach_upstream_from_proxy_and_gateway(self):
        options = {"chat_template_kwargs": {"enable_thinking": False}, "presence_penalty": 0.5,
                   "repeat_penalty": 1.1, "response_format": {"type": "json_schema", "json_schema": {
                       "name": "sample", "schema": {"type": "object"}}},
                   "max_completion_tokens": 42, "temperature": 0.6, "top_p": 0.95}
        payload = {"messages": [{"role": "user", "content": "echo-options"}], **options}
        expected = {**options, "max_tokens": options["max_completion_tokens"], "top_k": 20}
        del expected["max_completion_tokens"]
        direct = infer_chat(self.rt, self.manager.port, payload, timeout=10)
        self.assertEqual(direct.status, 200)
        self.assertEqual(json.loads(direct.text), expected)
        with make_proxy_server(self.rt, self.manager.port, self.model, listen_port=0) as proxy:
            thread = threading.Thread(target=proxy.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{proxy.server_port}/v1/chat/completions"

            def post(body):
                request = urllib.request.Request(base, json.dumps(body).encode(),
                                                 {"Content-Type": "application/json"})
                return urllib.request.urlopen(request, timeout=10)

            try:
                with post(dict(payload, stream=True)) as response:
                    events = [line[6:].strip() for line in response if line.startswith(b"data: ")]
                self.assertEqual(events[-1], b"[DONE]")
                chunks = [json.loads(event) for event in events[:-1]]
                content = "".join(choice.get("delta", {}).get("content", "")
                                  for chunk in chunks for choice in chunk.get("choices", []))
                self.assertEqual(json.loads(content), expected)
                invalid = dict(payload, chat_template_kwargs={"enable_thinking": "false"})
                self.assertEqual(infer_chat(self.rt, self.manager.port, invalid, timeout=10).status, 400)
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    post(invalid)
                with caught.exception as error:
                    self.assertEqual(error.code, 400)
            finally:
                proxy.shutdown()
                thread.join()

    def test_proxy_qwen_alias_and_canonical_id(self):
        with make_proxy_server(self.rt, self.manager.port, self.model, listen_port=0) as proxy:
            threading.Thread(target=proxy.serve_forever, daemon=True).start()
            base = f"http://127.0.0.1:{proxy.server_port}/v1"

            def list_models():
                with urllib.request.urlopen(base + "/models", timeout=10) as response:
                    return [item["id"] for item in json.load(response)["data"]]

            def post(model, stream):
                payload = {"model": model, "stream": stream,
                           "messages": [{"role": "user", "content": "question"}]}
                request = urllib.request.Request(base + "/chat/completions", json.dumps(payload).encode(),
                                                 {"Content-Type": "application/json"})
                return urllib.request.urlopen(request, timeout=10)

            try:
                self.assertEqual(list_models(), ["qwen-local", self.model])
                self.rt.session.serve.instance = "different-instance"
                self.assertEqual(list_models(), [self.model])
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    post("qwen-local", False)
                with caught.exception as error:
                    self.assertEqual(error.code, 409)
                self.rt.session.serve.instance = self.health["instance"]
                for model in ("qwen-local", self.model):
                    for stream in (False, True):
                        with self.subTest(model=model, stream=stream), post(model, stream) as response:
                            if stream:
                                lines = [line.strip() for line in response if line.strip()]
                                self.assertEqual(lines[-1], b"data: [DONE]")
                                events = [json.loads(line[6:]) for line in lines[:-1]]
                                self.assertEqual(events[0]["model"], self.model)
                                self.assertTrue(any(e["choices"][0].get("finish_reason") == "stop"
                                                    for e in events if e.get("choices")))
                            else:
                                body = json.load(response)
                                self.assertEqual(body["model"], self.model)
                                self.assertEqual(body["choices"][0]["finish_reason"], "stop")
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    post("unknown", False)
                with caught.exception as error:
                    self.assertEqual(error.code, 400)
                self.manager.stop()
                self.assertEqual(list_models(), [self.model])
            finally:
                proxy.shutdown()

    def test_legacy_record_with_qwen_basename_has_canonical_id_only(self):
        self.rt.session.serve = ServeRecord(model=self.model, backend="llama", port=self.manager.port,
                                            instance=self.health["instance"])
        with make_proxy_server(self.rt, self.manager.port, self.model, listen_port=0) as proxy:
            thread = threading.Thread(target=proxy.serve_forever, daemon=True)
            thread.start()
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{proxy.server_port}/v1/models", timeout=10) as response:
                    self.assertEqual([item["id"] for item in json.load(response)["data"]], [self.model])
            finally:
                proxy.shutdown()
                thread.join()

    @unittest.skipUnless(find_spec("openai"), "OpenAI SDK is optional")
    def test_openai_sdk_stream_and_non_stream(self):
        from openai import OpenAI

        with make_proxy_server(self.rt, self.manager.port, self.model, listen_port=0) as proxy:
            thread = threading.Thread(target=proxy.serve_forever, daemon=True)
            thread.start()
            try:
                with OpenAI(api_key="test", base_url=f"http://127.0.0.1:{proxy.server_port}/v1",
                            timeout=10, max_retries=0) as client:
                    options = {"model": self.model, "messages": [{"role": "user", "content": "question"}]}
                    reply = client.chat.completions.create(**options)
                    self.assertEqual(reply.choices[0].message.reasoning_content, "thought")
                    self.assertEqual(reply.usage.total_tokens, 8)
                    with client.chat.completions.create(**options, stream=True,
                                                        stream_options={"include_usage": True}) as stream:
                        events = list(stream)
                    self.assertEqual(events[0].choices[0].delta.reasoning_content, "thought")
                    self.assertEqual(events[-1].usage.total_tokens, 8)
            finally:
                proxy.shutdown()
                thread.join()


class LlamaOptionTests(unittest.TestCase):
    def test_optional_fields_are_gated_and_validated(self):
        payload = {"messages": [{"role": "user", "content": "question"}]}
        capabilities = dict.fromkeys(("enable_thinking", "presence_penalty", "repeat_penalty",
                                      "response_format", "json_schema"), True)
        for field, value in (
                ("chat_template_kwargs", {"enable_thinking": 0}),
                ("chat_template_kwargs", {"enable_thinking": True, "other": False}),
                ("presence_penalty", float("nan")), ("presence_penalty", 2.1),
                ("repeat_penalty", float("inf")), ("repeat_penalty", False), ("repeat_penalty", 0),
                ("response_format", {"type": "json_object", "schema": {}}),
                ("response_format", {"type": "json_schema", "json_schema": {"name": "ref",
                   "schema": {"properties": {"item": {"$ref": "https://example.org/schema"}}}}})):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                chat_args(dict(payload, **{field: value}), "model", capabilities)
        for field, value in (
                ("chat_template_kwargs", {"enable_thinking": False}),
                ("presence_penalty", 0), ("repeat_penalty", 1),
                ("response_format", {"type": "json_object"})):
            request = dict(payload, **{field: value})
            chat_args(request, "model", capabilities)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "unsupported"):
                chat_args(request, "model", {})


class LlamaToolTests(FakeEnvTest):
    def test_vm_fetch_prefers_dataset_and_checks_digest(self):
        root = Path(self.workdir)
        input_dir = root / "input"
        input_dir.mkdir(parents=True)
        model = input_dir / "example-Q4_K_M.gguf"
        model.write_bytes(b"GGUF")
        checksum = hashlib.sha256(model.read_bytes()).hexdigest()
        (input_dir / "SHA256SUMS").write_text(f"{checksum}  {model.name}\n")
        result = vm_operation(self.rt, "fetch", {"repo": "repo/name", "quant": "Q4_K_M",
                                                "directory": str(root / "cache"), "input_root": str(input_dir)},
                              10, lambda *_: None)
        self.assertEqual(result["source"], "dataset")
        self.assertEqual(result["sha256"], checksum)

    def test_plan_separates_hybrid_memory(self):
        def gguf_string(value):
            encoded = value.encode()
            return struct.pack("<Q", len(encoded)) + encoded

        fields = {"block_count": 40, "full_attention_interval": 4, "attention.head_count_kv": 2,
                  "attention.key_length": 256, "attention.value_length": 256, "ssm.state_size": 128,
                  "ssm.inner_size": 4096, "ssm.conv_kernel": 4, "ssm.group_count": 16}
        model = Path(self.workdir) / "hybrid.gguf"
        name = "qwen35moe"
        model.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, len(fields) + 1)
                          + gguf_string("general.architecture") + struct.pack("<I", 8) + gguf_string(name)
                          + b"".join(gguf_string(f"{name}.{key}") + struct.pack("<II", 4, value)
                                     for key, value in fields.items()))
        with patch.object(llama_tools, "capture", return_value="0, 14000, 15360\n1, 14000, 15360"):
            result = llama_tools.plan(model, 32768)
        self.assertEqual(result["architecture"], name)
        self.assertEqual(result["memory_estimate"]["attention_layers"], 10)
        self.assertEqual(result["memory_estimate"]["recurrent_layers"], 30)
        self.assertEqual(result["memory_estimate"]["kv_bytes_per_token"], 20480)
        self.assertGreater(result["memory_estimate"]["recurrent_bytes"], 0)
        self.assertEqual(result["recommended_context"], 32768)
