#!/usr/bin/env python3
"""kgl model server. Runs ON the Kaggle VM, bound to 127.0.0.1 only.

It is reached from your machine through Jupyter messaging (kgl infer /
kgl serve proxy) - it is never exposed to the internet by kgl.
Stdlib only, except the 'transformers' backend (torch + transformers,
preinstalled on Kaggle images - unverified).
"""
import argparse
from collections import deque
from contextlib import closing
import json
import math
from queue import Empty
import signal
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def parse_device_map(value):
    if value in (None, "balanced"):
        return value
    try:
        placement = json.loads(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("device map must be 'balanced' or a JSON object of module names and GPU indices") from exc
    if (not isinstance(placement, dict) or not placement
            or any(type(index) is not int or index < 0 for index in placement.values())
            or len(set(placement.values())) < 2):
        raise ValueError("device map must assign modules to at least two CUDA GPU indices")
    return placement


def chat_args(req, model_id):
    if not isinstance(req, dict):
        raise ValueError("request must be a JSON object")
    if req.get("model", model_id) != model_id:
        raise ValueError("requested model does not match the running model")
    for key in ("tools", "tool_choice", "functions", "function_call", "stop", "seed"):
        if req.get(key) is not None:
            raise ValueError(f"{key} is unsupported")
    if type(req.get("n", 1)) is not int or req.get("n", 1) != 1:
        raise ValueError("n must be 1")
    if type(req.get("stream", False)) is not bool:
        raise ValueError("stream must be a boolean")
    stream_options = req.get("stream_options")
    if stream_options is not None and (
            not req.get("stream") or not isinstance(stream_options, dict)
            or set(stream_options) - {"include_usage"}
            or type(stream_options.get("include_usage", False)) is not bool):
        raise ValueError("stream_options only supports include_usage for streaming requests")
    if "max_completion_tokens" in req and "max_tokens" in req:
        raise ValueError("supply only one of max_tokens and max_completion_tokens")
    maximum = req.get("max_completion_tokens", req.get("max_tokens", 256))
    if type(maximum) is not int or maximum <= 0:
        raise ValueError("completion token limit must be a positive integer")
    temperature, top_p = req.get("temperature", 0.7), req.get("top_p", 0.95)
    if type(temperature) not in (int, float) or not 0 <= temperature <= 2:
        raise ValueError("temperature must be between 0 and 2")
    if type(top_p) not in (int, float) or not 0 < top_p <= 1:
        raise ValueError("top_p must be greater than 0 and at most 1")
    messages = req.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    normalized = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in ("system", "user", "assistant"):
            raise ValueError("messages must have system, user or assistant roles")
        content = message.get("content")
        if isinstance(content, list):
            if not content or any(not isinstance(part, dict) or part.get("type") != "text"
                                  or not isinstance(part.get("text"), str) for part in content):
                raise ValueError("only text message parts are supported")
            content = "".join(part["text"] for part in content)
        if not isinstance(content, str):
            raise ValueError("message content must be text")
        normalized.append({"role": message["role"], "content": content})
    return normalized, maximum, temperature, top_p


class Generation:
    def __init__(self, request_id, timeout):
        self.id = request_id
        self.deadline = time.monotonic() + timeout
        self.cancelled = threading.Event()
        self.finished = threading.Event()
        self.worker_stopped = True
        self.finish_reason = "stop"
        self.usage = None

    def should_stop(self):
        return self.cancelled.is_set() or time.monotonic() >= self.deadline


class EchoBackend:
    """No model/GPU: validates the transport end to end."""

    name = "echo"
    device = "cpu"
    devices = ["cpu"]
    dtype = None

    def load(self):
        pass

    def generate(self, messages, max_tokens, temperature, top_p, *, generation=None):
        last = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
        for i, w in enumerate(f"echo: {last}".split(" ")):
            if generation and generation.should_stop():
                return
            if i:
                time.sleep(0.005)
            yield w if i == 0 else " " + w


class TransformersBackend:
    name = "transformers"

    def __init__(self, model_id, dtype, require_gpu=False, device_map=None, prefill_chunk_size=None):
        self.model_id, self.dtype = model_id, dtype
        self.require_gpu = require_gpu
        self.device_map = parse_device_map(device_map)
        self.prefill_chunk_size = prefill_chunk_size
        self.failure = None
        self.device = None
        self.devices = []
        self.lock = threading.Lock()

    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        use_cuda = torch.cuda.is_available()
        if self.require_gpu and not use_cuda:
            raise RuntimeError("CUDA is required but unavailable")
        if self.prefill_chunk_size:
            from transformers import GenerationConfig

            if not hasattr(GenerationConfig(), "prefill_chunk_size"):
                raise RuntimeError("installed Transformers does not support chunked prefill")

        dtypes = {"auto": "auto", "float16": torch.float16, "bfloat16": torch.bfloat16,
                  "float32": torch.float32}
        options = {"torch_dtype": dtypes[self.dtype]}
        if self.device_map is not None:
            count = torch.cuda.device_count()
            if count < 2:
                raise RuntimeError("multi-GPU placement requires at least two CUDA GPUs")
            options["device_map"] = self.device_map
            if self.device_map == "balanced":
                options["max_memory"] = {
                    index: max(0, torch.cuda.mem_get_info(index)[0] - 1024**3)
                    for index in range(count)
                }
            elif max(self.device_map.values()) >= count:
                raise RuntimeError(f"device map contains a GPU index outside the {count} visible CUDA GPUs")
        self.tok = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **options)
        if self.device_map is not None:
            placements = {f"cuda:{where}" if isinstance(where, int) else str(where)
                          for where in self.model.hf_device_map.values()}
            if len(placements) < 2 or any(not where.startswith("cuda:") for where in placements):
                raise RuntimeError(f"placement needs weights on multiple CUDA GPUs: {sorted(placements)}")
            self.devices = sorted(placements)
        else:
            self.model.to("cuda" if use_cuda else "cpu")
            self.devices = [str(self.model.device)]
        self.model.eval()
        self.device = str(self.model.get_input_embeddings().weight.device)
        self.dtype = str(self.model.dtype)

    def prepare(self, messages, max_tokens):
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(prompt, return_tensors="pt", add_special_tokens=False).to(
            self.model.get_input_embeddings().weight.device)
        context = getattr(self.model.config, "max_position_embeddings", None)
        if isinstance(context, int) and inputs["input_ids"].shape[-1] + max_tokens > context:
            raise ValueError(f"prompt plus completion exceeds model context ({context} tokens)")
        return inputs

    def generate(self, messages, max_tokens, temperature, top_p, *, generation=None, inputs=None):
        from transformers import TextIteratorStreamer

        if inputs is None:
            inputs = self.prepare(messages, max_tokens)
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True, timeout=60)
        kw = dict(**inputs, max_new_tokens=max_tokens, streamer=streamer, do_sample=temperature > 0)
        if self.prefill_chunk_size and inputs["input_ids"].shape[-1] > self.prefill_chunk_size:
            kw["prefill_chunk_size"] = self.prefill_chunk_size
        if temperature > 0:
            kw.update(temperature=temperature, top_p=top_p)
        if generation:
            from transformers import StoppingCriteria, StoppingCriteriaList

            class Cancelled(StoppingCriteria):
                def __call__(self, input_ids, scores, **kwargs):
                    return generation.should_stop()

            kw.update(stopping_criteria=StoppingCriteriaList([Cancelled()]), return_dict_in_generate=False)
        if not self.lock.acquire(timeout=1):
            raise RuntimeError("generation worker busy")
        errors = []

        def generate():
            try:
                output = self.model.generate(**kw)
                if generation:
                    prompt_tokens = inputs["input_ids"].shape[-1]
                    completion_tokens = output.shape[-1] - prompt_tokens
                    eos = self.model.generation_config.eos_token_id
                    eos_ids = eos if isinstance(eos, (list, tuple)) else (eos,)
                    if completion_tokens >= max_tokens and int(output[0, -1]) not in eos_ids:
                        generation.finish_reason = "length"
                    generation.usage = dict(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                                            total_tokens=prompt_tokens + completion_tokens)
            except Exception as exc:
                errors.append(exc)
                streamer.end()
            finally:
                self.lock.release()

        worker = threading.Thread(target=generate, daemon=True)
        try:
            worker.start()
        except Exception:
            self.lock.release()
            raise
        try:
            yield from streamer
        except Empty:
            self.failure = "generation worker timed out; restart server"
            raise TimeoutError(self.failure)
        finally:
            if generation and worker.is_alive():
                # A closed generator or a stalled streamer must release the worker.
                if sys.exc_info()[0] is not None:
                    generation.cancelled.set()
                worker.join(timeout=2)
                generation.worker_stopped = not worker.is_alive()
                if not generation.worker_stopped:
                    self.failure = "generation worker did not stop; restart server"
            else:
                worker.join(timeout=1)
        if errors:
            raise errors[0]


class State:
    def __init__(self):
        self.status = "loading"
        self.error = None
        self.lock = threading.Lock()
        self.active = None
        self.recent = deque(maxlen=128)


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
                status = ("error" if getattr(backend, "failure", None)
                          else "busy" if state.active else state.status)
                self._json(200, {"status": status,
                                 "model": model_id, "instance": instance,
                                 "device": backend.device, "devices": backend.devices, "dtype": backend.dtype,
                                 "backend": backend.name, "error": getattr(backend, "failure", None) or state.error})
            elif self.path == "/v1/models":
                self._json(200, {"object": "list", "data": [{"id": model_id, "object": "model"}]})
            else:
                self._json(404, {"error": {"message": "not found"}})

        def do_POST(self):
            if self.headers.get("X-Kgl-Instance", instance) != instance:
                return self._json(409, {"error": {"message": "server instance changed"}})
            request_id = self.headers.get("X-Kgl-Request-Id")
            if self.path == "/cancel":
                if not request_id or not instance or self.headers.get("X-Kgl-Instance") != instance:
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
            if self.path == "/internal/chat/completions" and (
                    not instance or self.headers.get("X-Kgl-Instance") != instance):
                return self._json(409, {"error": {"message": "server instance is required"}})
            if self.path not in ("/v1/chat/completions", "/internal/chat/completions"):
                return self._json(404, {"error": {"message": "not found"}})
            if state.status != "ready" or getattr(backend, "failure", None):
                failure = getattr(backend, "failure", None) or state.error or state.status
                return self._json(503, {"error": {"message": f"model unavailable: {failure}"}})
            try:
                n = int(self.headers.get("Content-Length", "0"))
                if not 0 < n <= 1024 * 1024:
                    raise ValueError("invalid request size")
                req = json.loads(self.rfile.read(n))
                args = chat_args(req, model_id)
                timeout = float(self.headers.get("X-Kgl-Timeout", "600"))
                if not math.isfinite(timeout) or timeout <= 0:
                    raise ValueError("invalid inference timeout")
            except (ValueError, OverflowError) as e:
                return self._json(400, {"error": {"message": str(e)}})
            generation = Generation(request_id, timeout)
            with state.lock:
                if state.active is not None:
                    return self._json(503, {"error": {"message": "generation worker busy"}})
                state.active = generation
            try:
                try:
                    options = ({"inputs": backend.prepare(args[0], args[1])}
                               if isinstance(backend, TransformersBackend) else {})
                except (ValueError, TypeError) as e:
                    return self._json(400, {"error": {"message": str(e)}})
                except Exception as e:
                    return self._json(500, {"error": {"message": f"cannot prepare prompt: {e}"}})
                with closing(backend.generate(*args, generation=generation, **options)) as pieces:
                    self._complete(req, pieces, generation)
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
                    if generation.usage is not None and (req.get("stream_options") or {}).get("include_usage"):
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
                    if generation.usage is not None:
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
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--backend", choices=["transformers", "echo"], default="transformers")
    ap.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="float16")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--require-gpu", action="store_true")
    ap.add_argument("--device-map")
    ap.add_argument("--prefill-chunk-size", type=int)
    a = ap.parse_args()
    if a.require_gpu and a.backend != "transformers":
        ap.error("--require-gpu requires the transformers backend")
    if a.device_map and a.backend != "transformers":
        ap.error("--device-map requires the transformers backend")
    if a.prefill_chunk_size is not None and (a.prefill_chunk_size <= 0 or a.backend != "transformers"):
        ap.error("--prefill-chunk-size requires a positive integer and the transformers backend")
    try:
        backend = (EchoBackend() if a.backend == "echo"
                   else TransformersBackend(a.model, a.dtype, a.require_gpu, a.device_map, a.prefill_chunk_size))
    except ValueError as exc:
        ap.error(str(exc))

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
