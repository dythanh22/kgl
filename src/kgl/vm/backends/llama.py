"""Supervise a pinned llama.cpp server and forward its OpenAI responses."""
import http.client
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import threading
import time

from ...protocol import chat_args, chat_defaults
from .base import BaseBackend


class LlamaBackend(BaseBackend):
    name = "llama"
    dtype = "gguf"
    device = None

    @classmethod
    def from_args(cls, args):
        if not args.llama_manifest:
            raise ValueError("llama requires --llama-manifest and a positive context")
        return cls(args.model, args.llama_manifest, args.ctx or 4096, args.tensor_split,
                   args.require_gpu, args.settings or {})

    def handle(self, handler, request, generation, model_id):
        req, args = self.parse_chat(request, model_id)
        self.complete(handler, dict(req, messages=args[0]), generation, model_id)

    def parse_chat(self, request, model_id):
        if not isinstance(request, dict):
            raise ValueError("request must be a JSON object")
        request = chat_defaults(request, self.settings.get("defaults", {}))
        return request, chat_args(request, model_id, self.settings.get("capabilities", {}))

    def busy(self):
        return self.slot_pending and not self.slot_available()

    def __init__(self, model, manifest, context, split, require_gpu, settings):
        self.model = model
        self.manifest = Path(manifest)
        self.context = context
        self.split = split
        self.require_gpu = require_gpu
        self.settings = settings
        self.process = None
        self.port = None
        self.log_path = None
        self.devices = []
        self.gpu_memory = []
        self.closing = threading.Event()
        self._process_lock = threading.Lock()
        self.slot_pending = False
        self.actual_context = None
        self.offloaded_layers = None
        self.last_timings = None

    @property
    def failure(self):
        if self.process and self.process.poll() is not None:
            tail = ""
            if self.log_path and self.log_path.is_file():
                with self.log_path.open("rb") as log:
                    log.seek(max(0, self.log_path.stat().st_size - 4096))
                    tail = log.read().decode(errors="replace").strip()
            return f"llama.cpp exited ({self.process.returncode})\n{tail}"
        return None

    def _read_log(self):
        max_bytes = 1024 * 1024
        with self.process.stdout as source, self.log_path.open("wb") as dest:
            for data in iter(lambda: source.read1(16384), b""):
                if dest.tell() + len(data) > max_bytes:
                    dest.seek(0)
                    dest.truncate()
                dest.write(data)
                dest.flush()

    def load(self):
        model = Path(self.model)
        with model.open("rb") as source:
            if source.read(4) != b"GGUF":
                raise ValueError("model does not have a GGUF header")
        manifest = json.loads(self.manifest.read_text())
        argv, options = manifest["argv"], manifest["options"]
        binary = self.manifest.parent / argv[0]
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            self.port = listener.getsockname()[1]
        args = [str(binary), *argv[1:]]
        values = {"model": str(model), "host": "127.0.0.1", "port": str(self.port),
                  "ctx": str(self.context), "gpu_layers": "all", "split_mode": "layer", "parallel": "1"}
        if self.split:
            values["tensor_split"] = self.split
        for key, value in values.items():
            args.extend((options[key], value))
        if "jinja" in options:
            args.append(options["jinja"])
        if self.settings.get("capabilities", {}).get("reasoning") and "reasoning_format" in options:
            args.extend((options["reasoning_format"], "deepseek"))
        self.log_path = Path.cwd() / f"llama-{os.getpid()}.log"
        with self._process_lock:
            if self.closing.is_set():
                return
            self.process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                            stdin=subprocess.DEVNULL)
        threading.Thread(target=self._read_log, daemon=True).start()
        try:
            while not self.closing.is_set():
                if self.failure:
                    raise RuntimeError(self.failure)
                conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=2)
                try:
                    conn.request("GET", "/health")
                    with conn.getresponse() as reply:
                        if reply.status == 200:
                            self.slot_available()
                            self._measure_gpu()
                            if self.require_gpu and not self.devices:
                                raise RuntimeError("llama.cpp did not allocate GPU memory")
                            if self.log_path.is_file():
                                output = self.log_path.read_text(errors="replace")
                                match = re.search(r"offloaded (\d+)/(\d+) layers", output)
                                if match:
                                    self.offloaded_layers = {"offloaded": int(match[1]), "total": int(match[2])}
                            return
                except (OSError, http.client.HTTPException):
                    pass
                finally:
                    conn.close()
                self.closing.wait(0.5)
        except BaseException:
            self.close()
            raise

    def _measure_gpu(self):
        def query(*args):
            return subprocess.run(["nvidia-smi", *args, "--format=csv,noheader,nounits"],
                                  check=True, text=True, capture_output=True, timeout=5).stdout.splitlines()

        try:
            cards = dict(line.split(", ", 1) for line in query("--query-gpu=uuid,index"))
            memory = []
            for line in query("--query-compute-apps=pid,gpu_uuid,used_gpu_memory"):
                pid, gpu, used = line.split(", ")
                try:
                    owned = os.getpgid(int(pid)) == os.getpgrp() if self.process.pid != int(pid) else True
                except ProcessLookupError:
                    continue
                if owned and gpu in cards and used.isdigit():
                    memory.append({"pid": int(pid), "gpu": int(cards[gpu]), "memory_mib": int(used)})
            self.gpu_memory = memory
            self.devices = [f"cuda:{index}" for index in sorted({sample["gpu"] for sample in memory})]
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
        self.device = self.devices[0] if self.devices else "cpu"

    def health(self):
        return {"capabilities": self.settings.get("capabilities", {}), "defaults": self.settings.get("defaults", {}),
                "alias": self.settings.get("alias"), "quant": self.settings.get("quant"),
                "n_ctx": self.actual_context, "requested_context": self.context, "gpu_memory": self.gpu_memory,
                "require_gpu": self.require_gpu,
                "offloaded_layers": self.offloaded_layers, "last_timings": self.last_timings,
                "llama_log": str(self.log_path) if self.log_path else None,
                "pid": self.process.pid if self.process else None}

    def close(self):
        with self._process_lock:
            self.closing.set()
            process = self.process
            if process and process.poll() is None:
                process.send_signal(signal.SIGTERM)
                try:
                    process.wait(timeout=4)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)

    def slot_available(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=0.35)
        try:
            conn.request("GET", "/slots?fail_on_no_slot=1")
            with conn.getresponse() as reply:
                if reply.status != 200:
                    return False
                slots = json.load(reply)
                available = next((slot for slot in slots if slot.get("is_processing") is False), None)
                if available is None:
                    return False
                self.actual_context = available.get("n_ctx")
                return True
        except (OSError, ValueError, TypeError, AttributeError, http.client.HTTPException):
            return False
        finally:
            conn.close()

    def complete(self, handler, request, generation, model_id):
        """Relay one completion and interrupt socket I/O on cancellation or deadline."""
        upstream = dict(request)
        upstream.pop("model", None)
        if "max_completion_tokens" in upstream:
            upstream["max_tokens"] = upstream.pop("max_completion_tokens")
        conn = http.client.HTTPConnection("127.0.0.1", self.port,
                                          timeout=max(0.01, generation.deadline - time.monotonic()))
        finished = threading.Event()
        stream_open = False
        upstream_done = False

        def watch(sock):
            while not finished.wait(0.05):
                if generation.should_stop():
                    try:
                        sock.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                    return

        try:
            if self.slot_pending and not self.slot_available():
                handler._json(503, {"error": {"message": "waiting for the llama.cpp slot to become idle"}})
                return
            self.slot_pending = True
            conn.connect()
            threading.Thread(target=watch, args=(conn.sock,), daemon=True).start()
            conn.request("POST", "/v1/chat/completions", body=json.dumps(upstream).encode(),
                         headers={"Content-Type": "application/json"})
            with conn.getresponse() as reply:
                if reply.status != 200 or not request.get("stream"):
                    body = json.load(reply)
                    if not isinstance(body, dict):
                        raise ValueError("llama.cpp returned a non-object JSON response")
                    upstream_done = True
                    if "model" in body:
                        body["model"] = model_id
                    if body.get("timings"):
                        self.last_timings = body["timings"]
                    if generation.should_stop():
                        raise TimeoutError("generation cancelled or timed out")
                    handler._json(reply.status, body)
                    return
                for raw in reply:
                    line = raw.strip()
                    if not line.startswith(b"data:"):
                        continue
                    if generation.should_stop():
                        raise TimeoutError("generation cancelled or timed out")
                    data = line[5:].strip()
                    if data != b"[DONE]":
                        event = json.loads(data)
                        if not isinstance(event, dict):
                            raise ValueError("llama.cpp returned a non-object SSE event")
                        if "model" in event:
                            event["model"] = model_id
                        if event.get("timings"):
                            self.last_timings = event["timings"]
                        data = json.dumps(event).encode()
                    if not stream_open:
                        handler.send_response(200)
                        handler.send_header("Content-Type", "text/event-stream")
                        handler.send_header("Cache-Control", "no-cache")
                        handler.end_headers()
                        stream_open = True
                    handler.wfile.write(b"data: " + data + b"\n\n")
                    handler.wfile.flush()
                    if data == b"[DONE]":
                        upstream_done = True
                        return
                raise OSError("llama.cpp closed the stream before [DONE]")
        except (BrokenPipeError, ConnectionResetError):
            generation.cancelled.set()
        except (OSError, ValueError, http.client.HTTPException) as exc:
            message = self.failure or str(exc)
            try:
                if stream_open:
                    event = {"error": {"message": message, "type": "incomplete_stream"}}
                    handler.wfile.write(b"data: " + json.dumps(event).encode() + b"\n\n")
                    handler.wfile.flush()
                else:
                    handler._json(504 if generation.should_stop() else 502, {"error": {"message": message}})
            except OSError:
                pass
        finally:
            finished.set()
            conn.close()
            if self.slot_pending and (upstream_done or self.failure):
                self.slot_pending = False
            elif self.slot_pending and generation.should_stop():
                deadline = time.monotonic() + 1.3
                while time.monotonic() < deadline:
                    if self.slot_available():
                        self.slot_pending = False
                        break
                    time.sleep(0.05)
            generation.worker_stopped = not self.slot_pending
