#!/usr/bin/env python3
"""kgl model server. Runs ON the Kaggle VM, bound to 127.0.0.1 only.

It is reached from your machine through Jupyter messaging (kgl infer /
kgl serve proxy) - it is never exposed to the internet by kgl.
Stdlib only, except the 'transformers' backend (torch + transformers,
preinstalled on Kaggle images - unverified).
"""
import argparse
import json
from queue import Empty
import signal
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class EchoBackend:
    """No model/GPU: validates the transport end to end."""

    name = "echo"
    device = "cpu"
    dtype = None

    def load(self):
        pass

    def generate(self, messages, max_tokens, temperature, top_p):
        last = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
        for i, w in enumerate(f"echo: {last}".split(" ")):
            if i:
                time.sleep(0.005)
            yield w if i == 0 else " " + w


class TransformersBackend:
    name = "transformers"

    def __init__(self, model_id, dtype):
        self.model_id, self.dtype = model_id, dtype
        self.failure = None
        self.device = None
        self.lock = threading.Lock()

    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtypes = {"auto": "auto", "float16": torch.float16, "bfloat16": torch.bfloat16,
                  "float32": torch.float32}
        self.tok = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=dtypes[self.dtype])
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        self.device = str(self.model.device)
        self.dtype = str(self.model.dtype)

    def generate(self, messages, max_tokens, temperature, top_p):
        from transformers import TextIteratorStreamer

        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True, timeout=60)
        kw = dict(**inputs, max_new_tokens=max_tokens, streamer=streamer, do_sample=temperature > 0)
        if temperature > 0:
            kw.update(temperature=temperature, top_p=top_p)
        if not self.lock.acquire(timeout=1):
            raise RuntimeError("generation worker busy")
        errors = []

        def generate():
            try:
                self.model.generate(**kw)
            except Exception as exc:
                errors.append(exc)
                streamer.end()
            finally:
                self.lock.release()

        worker = threading.Thread(target=generate, daemon=True)
        worker.start()
        try:
            yield from streamer
        except Empty:
            self.failure = "generation worker timed out; restart server"
            raise TimeoutError(self.failure)
        worker.join(timeout=1)
        if errors:
            raise errors[0]


class State:
    status = "loading"  # loading | ready | error
    error = None


def make_handler(backend, model_id, state, instance=None):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._json(200, {"status": "error" if getattr(backend, "failure", None) else state.status,
                                 "model": model_id, "instance": instance,
                                 "device": backend.device, "dtype": backend.dtype,
                                 "backend": backend.name, "error": getattr(backend, "failure", None) or state.error})
            elif self.path == "/v1/models":
                self._json(200, {"object": "list", "data": [{"id": model_id, "object": "model"}]})
            else:
                self._json(404, {"error": {"message": "not found"}})

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                return self._json(404, {"error": {"message": "not found"}})
            if state.status != "ready" or getattr(backend, "failure", None):
                return self._json(503, {"error": {"message": f"model {state.status}: {state.error or ''}"}})
            try:
                n = int(self.headers.get("Content-Length", "0"))
                req = json.loads(self.rfile.read(n))
                messages = req["messages"]
                assert isinstance(messages, list) and messages
            except Exception as e:
                return self._json(400, {"error": {"message": f"bad request: {e!r}"}})
            max_tokens = int(req.get("max_tokens") or 256)
            temperature = float(req.get("temperature", 0.7))
            top_p = float(req.get("top_p", 0.95))
            cid, created = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time())

            def chunk(delta, finish=None):
                return {"id": cid, "object": "chat.completion.chunk", "created": created,
                        "model": model_id, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

            try:
                if req.get("stream") is True:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    w = self.wfile
                    w.write(("data: " + json.dumps(chunk({"role": "assistant"})) + "\n\n").encode())
                    for piece in backend.generate(messages, max_tokens, temperature, top_p):
                        w.write(("data: " + json.dumps(chunk({"content": piece})) + "\n\n").encode())
                        w.flush()
                    w.write(("data: " + json.dumps(chunk({}, "stop")) + "\n\ndata: [DONE]\n\n").encode())
                    w.flush()
                else:
                    text = "".join(backend.generate(messages, max_tokens, temperature, top_p))
                    self._json(200, {"id": cid, "object": "chat.completion", "created": created,
                                     "model": model_id,
                                     "choices": [{"index": 0, "finish_reason": "stop",
                                                  "message": {"role": "assistant", "content": text}}]})
            except BrokenPipeError:
                pass
            except Exception as e:  # generation failure
                sys.stderr.write(f"generation error: {e!r}\n")
                if req.get("stream") is True:
                    self.close_connection = True
                    return
                try:
                    self._json(500, {"error": {"message": repr(e)}})
                except Exception:
                    pass

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--backend", choices=["transformers", "echo"], default="transformers")
    ap.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="float16")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()

    backend = EchoBackend() if a.backend == "echo" else TransformersBackend(a.model, a.dtype)
    model_id = a.model if a.backend != "echo" else "echo"
    state = State()
    server = ThreadingHTTPServer((a.host, a.port), make_handler(backend, model_id, state, a.instance))
    signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=server.shutdown).start())

    def load():
        try:
            t0 = time.time()
            backend.load()
            state.status = "ready"
            print(f"model ready in {time.time() - t0:.1f}s", flush=True)
        except Exception as e:
            state.status, state.error = "error", repr(e)
            print("load failed:", repr(e), flush=True)

    threading.Thread(target=load, daemon=True).start()
    print(f"listening on {a.host}:{a.port} backend={backend.name}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
