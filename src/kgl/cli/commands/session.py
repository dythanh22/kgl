"""Connect, inspect, and stop a notebook session."""

import getpass
import json
import os
import sys
import time

from ...client.serve_manager import ServeManager
from ...core.config import Settings
from ...core.errors import AuthError, ConnectionLost, KernelGone, KglError
from ...core.state import Session, SessionStore, redact_headers, redact_url
from ...backends import backend_spec
from ...remote import PROBE
from ...transport.jupyter import JupyterClient
from ...transport.runtime import Runtime
from ..common import EXIT_CONNECTION, EXIT_ERROR, EXIT_OK, open_runtime, eprint


def _probe(rt: Runtime, st: Settings) -> dict:
    return rt.call(PROBE, idempotent=True, WORKDIR=st.workdir)


def _fmt_probe(info: dict) -> list[str]:
    lines = [f"python      : {info.get('python', '?')}"]
    gpus = info.get("gpus") or []
    if gpus:
        for g in gpus:
            lines.append(f"gpu         : {g['name']} {g['mem_used_mb']}/{g['mem_total_mb']} MB "
                         f"util {g['util_pct']}%")
    else:
        lines.append(f"gpu         : NONE ({info.get('gpu_error', 'unknown')})")
    if "ram_total_mb" in info:
        lines.append(f"ram         : {info['ram_avail_mb']} MB free / {info['ram_total_mb']} MB")
    if "disk_free_gb" in info:
        lines.append(f"disk free   : {info['disk_free_gb']} GB")
    return lines


def cmd_auth(a, st: Settings, store: SessionStore) -> int:
    print("Open Run > Kaggle Jupyter Server > Manually Connect in the notebook. "
          "Copy the VS Code Compatible URL, run kgl connect and paste it at the hidden prompt.")
    return EXIT_OK


def cmd_connect(a, st: Settings, store: SessionStore) -> int:
    url = a.proxy_url or os.environ.get("KGL_PROXY_URL")
    if url == "-":
        url = sys.stdin.readline().strip()
    elif not url and sys.stdin.isatty():
        try:
            url = getpass.getpass("Kaggle: Run > Kaggle Jupyter Server > Manually Connect. "
                                  "Paste VS Code Compatible URL (input hidden), then press Enter: ").strip()
        except EOFError:
            url = ""
        if url:
            eprint("URL received. Connecting...")
    if not url:
        raise KglError("missing proxy URL: use --proxy-url URL, '-' (stdin) or $KGL_PROXY_URL")
    headers = {}
    for h in a.header or []:
        k, _, v = h.partition(":")
        if not v:
            raise KglError(f"bad --header {h!r}, expected 'Name: value'")
        headers[k.strip()] = v.strip()
    prior = store.load()
    dedicated = False
    client = JupyterClient(url, headers)
    kernels = client.list_kernels()
    if a.kernel_id:
        match = [k for k in kernels if k["id"].startswith(a.kernel_id)]
        if not match:
            raise KglError(f"kernel {a.kernel_id} not found; available: {[k['id'] for k in kernels]}")
        kernel = match[0]
    elif kernels and not a.new_kernel:
        kernel = kernels[0]
        if len(kernels) > 1:
            eprint(f"note: {len(kernels)} kernels found, using {kernel['id'][:8]} "
                   "(choose with --kernel-id)")
    else:
        eprint("no kernel found; creating one (may not be permitted on Kaggle)")
        kernel = client.create_kernel()
        dedicated = True
    sess = Session(base_url=url, kernel_id=kernel["id"], headers=headers,
                   control_kernel_id=st.control_kernel_id, dedicated_kernel=dedicated)
    print(f"selected: {redact_url(url)} kernel={kernel['id'][:8]} "
          f"state={kernel.get('execution_state', '?')}")
    rt = Runtime(sess, st)
    try:
        lines = _fmt_probe(_probe(rt, st))
        store.save(sess, expected=prior)
        print("connected")
        for line in lines:
            print(line)
    finally:
        rt.close()
    return EXIT_OK


def cmd_status(a, st: Settings, store: SessionStore) -> int:
    rt, s = open_runtime(st, store)
    out: dict = {"url": redact_url(s.base_url), "kernel_id": s.kernel_id,
                 "headers": redact_headers(s.headers),
                 "connected_for_s": int(time.time() - s.connected_at),
                 "quota": "unavailable (no official API; not implemented)"}
    try:
        out["kernel"] = rt.client.get_kernel(s.kernel_id).get("execution_state", "?")
        out["connection"] = "ok"
        info = {} if out["kernel"] == "busy" else _probe(rt, st)
        if info:
            out["probe"] = info
        remaining = st.approx_session_remaining(info.get("container_uptime_s"))
        if remaining is not None:
            out["approx_session_remaining_s"] = remaining
        if s.serve:
            out["serve"] = ({"state": "busy", "instance": s.serve.instance}
                            if out["kernel"] == "busy" else ServeManager(rt, s.serve.port).health())
    except (ConnectionLost, AuthError, KernelGone) as e:
        out["connection"] = f"LOST: {e}"
    finally:
        rt.close()
    if a.json:
        print(json.dumps(out, indent=2))
    else:
        print(f"url         : {out['url']}\nkernel      : {s.kernel_id[:8]} "
              f"({out.get('kernel', 'unknown')})\nconnection  : {out['connection']}")
        if "probe" in out:
            print("\n".join(_fmt_probe(out["probe"])))
        if "approx_session_remaining_s" in out:
            rem = out["approx_session_remaining_s"]
            print(f"session     : ~{rem // 3600}h{rem % 3600 // 60:02d}m left of "
                  f"{st.session_limit_hours:g}h (approx; container uptime, limit unverified)")
        if "serve" in out:
            print(f"model server: {out['serve'].get('state')}")
            health = out["serve"].get("health", {})
            if backend_spec(health.get("backend")).passthrough:
                print(f"context     : {health.get('n_ctx') or '?'}")
                layers = health.get("offloaded_layers")
                if layers:
                    print(f"GPU layers  : {layers['offloaded']}/{layers['total']}")
                for gpu in health.get("gpu_memory", []):
                    print(f"worker VRAM : GPU {gpu['gpu']} PID {gpu['pid']} {gpu['memory_mib']} MiB")
                timings = health.get("last_timings") or {}
                for key, label in (("prompt_per_second", "prompt"), ("predicted_per_second", "generation")):
                    if isinstance(timings.get(key), (int, float)):
                        print(f"{label:12}: {timings[key]:.2f} tok/s")
        print(f"quota       : {out['quota']}")
    return EXIT_OK if out["connection"] == "ok" else EXIT_CONNECTION


def cmd_stop(a, st: Settings, store: SessionStore) -> int:
    rt, s = open_runtime(st, store)
    try:
        if s.serve and not a.keep_server:
            try:
                print("model server:", json.dumps(ServeManager(rt, s.serve.port).stop(s.serve.instance)))
            except KglError as e:
                eprint(f"could not stop model server: {e}")
                return EXIT_ERROR
            if a.kill_kernel:
                s.serve = None
                store.save(s, expected=s)
        if a.kill_kernel:
            try:
                rt.client.delete_kernel(s.kernel_id)
                print("kernel deleted")
            except KernelGone:
                print("kernel already absent")
    finally:
        rt.close()
    store.clear(expected=s)
    print("local session cleared. NOTE: the Kaggle notebook session itself is still running and "
          "counting against your quota - stop it from the Kaggle UI.")
    return EXIT_OK
