"""Manage the model gateway deployed to the notebook VM."""

from __future__ import annotations

from importlib.resources import files
import json
import logging
from pathlib import Path
import posixpath
import tempfile
import time
from typing import Callable
import zipfile

from ..backends import DEFAULT_BACKEND, OPTIONS, backend_spec
from ..core.errors import KglError, ServeError
from ..remote import SERVE_HEALTH, SERVE_LOGS, SERVE_START, SERVE_STOP
from ..transport.runtime import Runtime
from .files import upload_file

log = logging.getLogger(__name__)


def _write_gateway(path: Path) -> None:
    package = files("kgl")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("__main__.py", "from kgl.vm.gateway import main\nmain()\n")
        archive.writestr("kgl/__init__.py", "")
        for name in ("protocol.py", "backends.py"):
            archive.writestr(f"kgl/{name}", package.joinpath(name).read_bytes())
        vm = package.joinpath("vm")
        for name in ("__init__.py", "gateway.py", "state.py"):
            archive.writestr(f"kgl/vm/{name}", vm.joinpath(name).read_bytes())
        for item in sorted(vm.joinpath("backends").iterdir(), key=lambda resource: resource.name):
            if item.name.endswith(".py"):
                archive.writestr(f"kgl/vm/backends/{item.name}", item.read_bytes())


class ServeManager:
    def __init__(self, rt: Runtime, port: int = 8000) -> None:
        self.rt, self.port = rt, port
        self.dir = posixpath.join(rt.settings.workdir, ".kgl")

    def start(self, model: str, backend: str = DEFAULT_BACKEND, dtype: str = "float16",
              wait: bool = True, wait_timeout: float = 900.0,
              progress: Callable[[str], None] | None = None, require_gpu: bool = False,
              options: dict | None = None, on_start: Callable[[dict], None] | None = None) -> dict:
        say = progress or (lambda s: None)
        spec = backend_spec(backend)
        options = options or {}
        if spec.passthrough and (not posixpath.isabs(model) or not options.get("llama_manifest")):
            raise ServeError("llama requires absolute VM paths to the GGUF and build manifest")
        if spec.custom and not options.get("handler"):
            raise ServeError("python requires --handler")
        try:
            spec.validate_options(options, require_gpu)
        except ValueError as exc:
            raise ServeError(str(exc)) from exc
        with tempfile.TemporaryDirectory() as directory:
            gateway = Path(directory) / "serve.pyz"
            _write_gateway(gateway)
            upload_file(self.rt, gateway, posixpath.join(self.dir, gateway.name))
        args = ["--model", model, "--backend", backend, "--dtype", dtype,
                "--port", str(self.port)]
        args.extend(["--request-bytes", str(self.rt.settings.request_bytes)])
        if require_gpu:
            args.append("--require-gpu")
        for name in sorted(options):
            value = options[name]
            args.extend([OPTIONS[name].flag, json.dumps(value, sort_keys=True) if isinstance(value, dict) else str(value)])
        r = self.rt.call(SERVE_START, DIR=self.dir, ARGS=args)
        say("server process already running" if r.get("already_running")
            else f"server process started (pid {r['pid']})")
        try:
            if on_start:
                on_start(r)
            if not wait:
                return r
            result = self.wait_ready(wait_timeout, say, r["instance"], backend, spec.model_id(model))
        except (KglError, OSError, KeyboardInterrupt):
            if r.get("started"):
                try:
                    self.stop(r["instance"])
                except (KglError, TimeoutError) as exc:
                    log.warning("server cleanup was not confirmed: %s", exc)
            raise
        result["started"] = bool(r.get("started"))
        return result

    def health(self, timeout: float = 10.0) -> dict:
        control = self.rt.control_runtime()
        try:
            return (control or self.rt).call(SERVE_HEALTH, timeout=timeout, idempotent=True,
                                             DIR=self.dir, PORT=self.port)
        finally:
            if control:
                control.close()

    def wait_ready(self, timeout: float, say, instance: str, backend: str, model: str) -> dict:
        control = self.rt.control_runtime()
        try:
            return self._wait_ready(control or self.rt, timeout, say, instance, backend, model)
        finally:
            if control:
                control.close()

    def _wait_ready(self, runtime: Runtime, timeout: float, say, instance: str, backend: str, model: str) -> dict:
        t0, last = time.monotonic(), None
        while time.monotonic() - t0 < timeout:
            h = runtime.call(SERVE_HEALTH, timeout=max(0.01, timeout - (time.monotonic() - t0)),
                             idempotent=True, DIR=self.dir, PORT=self.port)
            state = h["state"]
            if state != last:
                say(f"server state: {state}")
                last = state
            if state == "ready":
                health = h["health"]
                if (health.get("instance"), health.get("backend"), health.get("model")) != (instance, backend, model):
                    raise ServeError("unexpected server identity, backend or model on requested port")
                return h
            if h.get("health", {}).get("status") == "error":
                raise ServeError("model failed to load: " + str(h["health"].get("error"))
                                 + "\nsee `kgl serve logs`")
            if state == "down" and time.monotonic() - t0 > 30:
                raise ServeError("server not responding:\n" + "\n".join(h.get("log_tail", [])))
            time.sleep(min(1.0, max(0, timeout - (time.monotonic() - t0))))
        raise ServeError(f"timed out after {timeout:.0f}s waiting for model; see `kgl serve logs`")

    def stop(self, instance: str | None = None) -> dict:
        control = self.rt.control_runtime() if instance else None
        try:
            return (control or self.rt).call(SERVE_STOP, timeout=20, DIR=self.dir, INSTANCE=instance)
        finally:
            if control:
                control.close()

    def logs(self, n: int = 50) -> list[str]:
        return self.rt.call(SERVE_LOGS, DIR=self.dir, N=n).get("lines", [])
