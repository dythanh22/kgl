#!/usr/bin/env python3
"""kgl model server. Runs ON the Kaggle VM, bound to 127.0.0.1 only.

It is reached from your machine through Jupyter messaging (kgl infer /
kgl serve proxy) - it is never exposed to the internet by kgl.
The echo backend uses the standard library. Model backends load dependencies
from the notebook environment.
"""
import argparse
import json
import math
import signal
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from ..backends import OPTIONS, BACKENDS, DEFAULT_BACKEND, DEFAULT_MODEL, backend_spec
from ..protocol import INSTANCE_HEADER, MAX_REQUEST_BYTES, REQUEST_ID_HEADER, TIMEOUT_HEADER, server_health
from .backends import ADAPTERS
from .state import Generation, State


def make_handler(backend, model_id, state, instance=None, *, request_bytes=None):
    class Handler(BaseHTTPRequestHandler):
        request_limit = request_bytes or MAX_REQUEST_BYTES

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
                failure = backend.failure
                busy = state.active or (state.status == "ready" and not failure and backend.busy())
                status = "error" if failure else "busy" if busy else state.status
                self._json(200, server_health(backend, model_id, instance, status, failure or state.error))
            elif self.path == "/v1/models":
                self._json(200, {"object": "list", "data": [{"id": model_id, "object": "model"}]})
            else:
                self._json(404, {"error": {"message": "not found"}})

        def do_POST(self):
            if self.headers.get(INSTANCE_HEADER, instance) != instance:
                return self._json(409, {"error": {"message": "server instance changed"}})
            request_id = self.headers.get(REQUEST_ID_HEADER)
            if self.path == "/cancel":
                if not request_id or not instance or self.headers.get(INSTANCE_HEADER) != instance:
                    return self._json(400, {"error": {"message": "request and instance are required"}})
                with state.lock:
                    active = state.active
                    if active is None or active.id != request_id:
                        stopped = next((stopped for recent_id, stopped in state.recent
                                        if recent_id == request_id), None)
                    else:
                        stopped = None
                        active.cancelled.set()
                if active is None or active.id != request_id:
                    return self._json(404 if stopped is None else 200, {"cancelled": stopped is True})
                cancelled = active.finished.wait(2) and active.worker_stopped
                return self._json(200 if cancelled else 202, {"cancelled": cancelled})
            if self.path in ("/internal/chat/completions", "/internal/handle") and (
                    not instance or self.headers.get(INSTANCE_HEADER) != instance):
                return self._json(409, {"error": {"message": "server instance is required"}})
            if self.path not in backend.routes:
                return self._json(404, {"error": {"message": "not found"}})
            if state.status != "ready" or backend.failure:
                failure = backend.failure or state.error or state.status
                return self._json(503, {"error": {"message": f"model unavailable: {failure}"}})
            try:
                sizes = self.headers.get_all("Content-Length", [])
                if len(sizes) != 1 or self.headers.get("Transfer-Encoding"):
                    raise ValueError("a single Content-Length is required")
                n = int(sizes[0])
                if n > self.request_limit:
                    return self._json(413, {"error": {"message": "request exceeds byte limit"}})
                if n <= 0:
                    raise ValueError("invalid request size")
                req = json.loads(self.rfile.read(n))
                timeout = float(self.headers.get(TIMEOUT_HEADER, "600"))
                if not math.isfinite(timeout) or timeout <= 0:
                    raise ValueError("invalid inference timeout")
            except (ValueError, OverflowError, RecursionError) as e:
                return self._json(400, {"error": {"message": str(e)}})
            generation = Generation(request_id, timeout)
            with state.lock:
                if state.active is not None:
                    return self._json(503, {"error": {"message": "generation worker busy"}})
                state.active = generation
            try:
                backend.handle(self, req, generation, model_id)
            except (ValueError, TypeError) as exc:
                self._json(400, {"error": {"message": str(exc)}})
            finally:
                with state.lock:
                    state.active = None
                    if generation.id:
                        state.recent.append((generation.id, generation.worker_stopped))
                    generation.finished.set()

        def _complete(self, req, pieces, generation):
            cid, created = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time())

            def chunk(delta, finish=None):
                return {"id": cid, "object": "chat.completion.chunk", "created": created,
                        "model": model_id, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

            started = False
            try:
                if generation.should_stop():
                    return self._json(504, {"error": {"message": "generation cancelled or timed out"}})
                if req.get("stream") is True:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    started = True
                    w = self.wfile
                    w.write(("data: " + json.dumps(chunk({"role": "assistant"})) + "\n\n").encode())
                    for piece in pieces:
                        w.write(("data: " + json.dumps(chunk({"content": piece})) + "\n\n").encode())
                        w.flush()
                    if generation.should_stop():
                        raise TimeoutError("generation cancelled or timed out")
                    w.write(("data: " + json.dumps(chunk({}, generation.finish_reason)) + "\n\n").encode())
                    if (generation.usage is not None and "usage" in backend.spec.capabilities
                            and (req.get("stream_options") or {}).get("include_usage")):
                        usage = dict(chunk({}), choices=[], usage=generation.usage)
                        w.write(("data: " + json.dumps(usage) + "\n\n").encode())
                    w.write(b"data: [DONE]\n\n")
                    w.flush()
                else:
                    text = "".join(pieces)
                    if generation.should_stop():
                        raise TimeoutError("generation cancelled or timed out")
                    body = {"id": cid, "object": "chat.completion", "created": created,
                                     "model": model_id,
                                     "choices": [{"index": 0, "finish_reason": generation.finish_reason,
                                                  "message": {"role": "assistant", "content": text}}]}
                    if generation.usage is not None and "usage" in backend.spec.capabilities:
                        body["usage"] = generation.usage
                    self._json(200, body)
            except (BrokenPipeError, ConnectionResetError):
                generation.cancelled.set()
            except Exception as e:  # generation failure
                sys.stderr.write(f"generation error: {e!r}\n")
                if started:
                    self.close_connection = True
                    try:
                        event = {"error": {"message": str(e), "type": "incomplete_stream"}}
                        self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
                        self.wfile.flush()
                    except OSError:
                        pass
                    return
                try:
                    self._json(504 if isinstance(e, TimeoutError) else 500,
                               {"error": {"message": str(e)}})
                except OSError:
                    pass

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--backend", choices=BACKENDS, default=DEFAULT_BACKEND)
    ap.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="float16")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--require-gpu", action="store_true")
    ap.add_argument("--request-bytes", type=int)
    for name, option in OPTIONS.items():
        ap.add_argument(option.flag, dest=name, type=option.type)
    a = ap.parse_args()
    spec = backend_spec(a.backend)
    try:
        spec.validate_options({k: getattr(a, k) for k in OPTIONS if getattr(a, k) is not None}, a.require_gpu)
        if a.request_bytes is not None and a.request_bytes <= 0:
            raise ValueError("--request-bytes must be positive")
        backend = ADAPTERS[a.backend].from_args(a)
    except ValueError as exc:
        ap.error(str(exc))

    model_id = spec.model_id(a.model)
    state = State()
    server = ThreadingHTTPServer((a.host, a.port), make_handler(
        backend, model_id, state, a.instance, request_bytes=a.request_bytes))
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
    try:
        server.serve_forever()
    finally:
        backend.close()
        server.server_close()


if __name__ == "__main__":
    main()
