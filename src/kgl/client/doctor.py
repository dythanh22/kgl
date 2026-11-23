"""`kgl doctor`: runs the PoC spikes against a live runtime and reports facts.
Nothing here is assumed to work - every step records pass/fail + timing."""
from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass
from typing import Callable

from ..remote import PROBE
from ..transport.runtime import Runtime
from .infer import infer_chat
from .serve_manager import ServeManager
from ..backends import DOCTOR_BACKENDS, backend_spec
from ..protocol import gpu_ready


@dataclass
class Step:
    name: str
    ok: bool
    detail: str
    seconds: float


def _pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]


def run_doctor(rt: Runtime, backend: str, model: str, port: int, n: int,
               keep: bool, wait_timeout: float,
               say: Callable[[Step], None] = lambda s: None) -> list[Step]:
    steps: list[Step] = []

    def record(name: str, fn: Callable[[], tuple[bool, str]]) -> bool:
        t0 = time.monotonic()
        try:
            ok, detail = fn()
        except Exception as e:  # report, never crash the report
            ok, detail = False, f"{type(e).__name__}: {e}"
        s = Step(name, ok, detail, round(time.monotonic() - t0, 2))
        steps.append(s)
        say(s)
        return ok

    def s1():
        ks = rt.client.list_kernels()
        return True, f"{len(ks)} kernel(s): " + ", ".join(f"{k['id'][:8]}({k.get('execution_state', '?')})" for k in ks)

    def s2():
        r = rt.run("print(1+1)", total_timeout=30, check=True)
        return r.text.strip() == "2", f"output={r.text.strip()!r}"

    def s3():
        info = rt.call(PROBE, idempotent=True, WORKDIR=rt.settings.workdir)
        gpus = info.get("gpus") or []
        if gpus:
            return True, "; ".join(f"{g['name']} {g['mem_total_mb']}MB" for g in gpus)
        return False, f"no GPU visible ({info.get('gpu_error', 'unknown')}) - is the accelerator enabled?"

    mgr = ServeManager(rt, port)
    spec = backend_spec(backend)

    started_instance = None
    instance = None

    def s4():
        nonlocal started_instance, instance
        saved = rt.session.serve
        if saved and saved.port == port and saved.instance:
            expected_model = saved.model if spec.passthrough else spec.model_id(model)
            result = mgr.wait_ready(wait_timeout, lambda _: None, saved.instance,
                                    backend, expected_model)
        else:
            if spec.passthrough:
                return False, "start the llama backend on this port before running doctor"
            result = mgr.start(model, backend, wait_timeout=wait_timeout,
                               require_gpu=backend in DOCTOR_BACKENDS and spec.gpu_supported)
        health = result["health"]
        instance = health["instance"]
        if result.get("started"):
            started_instance = instance
        ok = not spec.gpu_supported or gpu_ready(health)
        return ok, str(health)

    lat: list[float] = []
    ttft: list[float] = []

    def s5():
        fails = 0
        for i in range(n):
            r = infer_chat(rt, port, {"messages": [{"role": "user", "content": f"say hi #{i}"}],
                                      "max_tokens": 32, "stream": True}, expected_instance=instance)
            if r.status == 200 and (r.text or (spec.passthrough and r.reasoning and r.finish_reason)):
                lat.append(r.latency_s)
                if r.ttft_s is not None:
                    ttft.append(r.ttft_s)
            else:
                fails += 1
        if not lat:
            return False, f"all {n} requests failed"
        d = (f"{len(lat)}/{n} ok; latency p50={_pct(lat, .5):.2f}s p95={_pct(lat, .95):.2f}s "
             f"mean={statistics.mean(lat):.2f}s")
        if ttft:
            d += f"; ttft p50={_pct(ttft, .5):.2f}s"
        return fails == 0, d

    def s6():
        rt.drop_channel()  # simulates a lost WebSocket (not a real network cut)
        r = infer_chat(rt, port, {"messages": [{"role": "user", "content": "after reconnect"}],
                                  "max_tokens": 16}, expected_instance=instance)
        has_output = bool(r.text or (spec.passthrough and r.reasoning and r.finish_reason))
        return r.status == 200 and has_output, f"status={r.status}"

    if record("1 rest_reachable", s1) and record("2 kernel_exec", s2):
        record("3 gpu_visible", s3)
        if record("4 serve_start", s4):
            if record("5 infer_x%d" % n, s5):
                record("6 reconnect", s6)
        if not keep and started_instance:
            def stop_server():
                result = mgr.stop(started_instance)
                return result.get("stopped") is True, str(result)
            record("7 serve_stop", stop_server)
    return steps


def steps_to_dict(steps: list[Step]) -> list[dict]:
    return [asdict(s) for s in steps]
