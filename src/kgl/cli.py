"""kgl command line interface (argparse, no third-party deps)."""
from __future__ import annotations

import argparse
import getpass
import json
import logging
import os
import re
import stat
import sys
import time
from pathlib import Path

from . import __version__
from .config import Settings
from .doctor import run_doctor, steps_to_dict
from .errors import (AuthError, ConnectionLost, ExecutionError, KernelGone, KglError,
                     NoSession, ServeError)
from .files import download_file, upload_file
from .jupyter import JupyterClient
from .remote import PROBE, SHELL, last_marker, render
from .runtime import Runtime
from .serving import ServeManager, infer_chat, make_proxy_server
from .state import Session, SessionStore, redact_headers, redact_url

EXIT_OK, EXIT_ERROR, EXIT_USAGE, EXIT_CONNECTION, EXIT_REMOTE = 0, 1, 2, 3, 4
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
log = logging.getLogger("kgl")


def eprint(*a: object) -> None:
    print(*a, file=sys.stderr)


def _require_session(store: SessionStore) -> Session:
    s = store.load()
    if s is None:
        raise NoSession("no active session. Run `kgl connect --proxy-url ...` first.")
    return s


def _runtime(st: Settings, store: SessionStore) -> tuple[Runtime, Session]:
    s = _require_session(store)
    return Runtime(s, st), s


def _write_stream(name: str, text: str) -> None:
    out = sys.stderr if name == "stderr" else sys.stdout
    out.write(text)
    out.flush()


def _probe(rt: Runtime, st: Settings) -> dict:
    return rt.call(PROBE, WORKDIR=st.workdir)


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


# ---------------------------------------------------------------- commands
def cmd_auth(a, st: Settings, store: SessionStore) -> int:
    """Report which *official* Kaggle credentials exist (values are never shown)."""
    found = []
    if os.environ.get("KAGGLE_API_TOKEN"):
        found.append("env KAGGLE_API_TOKEN")
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        found.append("env KAGGLE_USERNAME + KAGGLE_KEY")
    for p in (Path.home() / ".kaggle" / "access_token", Path.home() / ".kaggle" / "kaggle.json"):
        if p.exists():
            mode = stat.S_IMODE(p.stat().st_mode)
            warn = "  (WARNING: readable by others, chmod 600)" if mode & 0o077 else ""
            found.append(f"file {p}{warn}")
    print("Official Kaggle API credentials (used only for dataset/kernel CLI features):")
    print("  " + ("\n  ".join(found) if found else "none found"))
    if not found:
        print("  Create a token in Kaggle > Settings > API, then set KAGGLE_API_TOKEN or place it "
              "in ~/.kaggle/.")
    s = store.load()
    print(f"\nRuntime session: {'saved (' + redact_url(s.base_url) + ')' if s else 'none'}")
    print("Interactive runtime access needs the notebook's Jupyter proxy URL; there is no "
          "official API for it.\nSee docs/TESTING.md ('Getting the proxy URL').")
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
    sess = Session(base_url=url, kernel_id=kernel["id"], headers=headers)
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
    rt, s = _runtime(st, store)
    out: dict = {"url": redact_url(s.base_url), "kernel_id": s.kernel_id,
                 "headers": redact_headers(s.headers),
                 "connected_for_s": int(time.time() - s.connected_at),
                 "quota": "unavailable (no official API; not implemented)"}
    try:
        out["kernel"] = rt.client.get_kernel(s.kernel_id).get("execution_state", "?")
        out["connection"] = "ok"
        info = _probe(rt, st)
        out["probe"] = info
        if "container_uptime_s" in info:
            limit = st.session_limit_hours * 3600
            out["approx_session_remaining_s"] = int(limit - info["container_uptime_s"])
        if s.serve:
            out["serve"] = ServeManager(rt, s.serve["port"]).health()
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
        print(f"quota       : {out['quota']}")
    return EXIT_OK if out["connection"] == "ok" else EXIT_CONNECTION


def _read_code(a) -> str:
    if a.file:
        return Path(a.file).read_text()
    if a.code:
        return " ".join(a.code)
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise KglError("no code given: pass code, -f FILE, or pipe via stdin")


def cmd_exec(a, st: Settings, store: SessionStore) -> int:
    rt, s = _runtime(st, store)
    code = _read_code(a)
    if a.shell:
        code = render(SHELL, CWD=st.workdir, CMD=code)
    try:
        res, markers = rt.run_markers(code, _write_stream, idle_timeout=a.idle_timeout or None,
                                      total_timeout=a.timeout or None)
    except KeyboardInterrupt:
        eprint("\ninterrupting remote kernel...")
        try:
            rt.client.interrupt_kernel(s.kernel_id)
        except KglError as e:
            eprint(f"interrupt failed: {e}")
        return 130
    finally:
        rt.close()
    if res.error:
        eprint(ANSI.sub("", "\n".join(res.error["traceback"])) or
               f"{res.error['ename']}: {res.error['evalue']}")
        return EXIT_REMOTE
    if a.shell and markers and "rc" in markers[-1]:
        return max(0, min(int(markers[-1]["rc"]), 125))
    return EXIT_OK


def _progress(label: str):
    if not sys.stderr.isatty():
        return None

    def p(done: int, total: int) -> None:
        pct = 100 if total == 0 else done * 100 // total
        sys.stderr.write(f"\r{label}: {pct:3d}% ({done}/{total} bytes)")
        if done >= total:
            sys.stderr.write("\n")
        sys.stderr.flush()
    return p


def cmd_upload(a, st: Settings, store: SessionStore) -> int:
    rt, _ = _runtime(st, store)
    try:
        r = upload_file(rt, Path(a.local), a.remote, _progress("upload"))
    finally:
        rt.close()
    print(f"uploaded {r['size']} bytes -> {r['remote']} (sha256 verified)")
    return EXIT_OK


def cmd_download(a, st: Settings, store: SessionStore) -> int:
    rt, _ = _runtime(st, store)
    try:
        r = download_file(rt, a.remote, Path(a.local), _progress("download"))
    finally:
        rt.close()
    print(f"downloaded {r['size']} bytes -> {r['local']} (sha256 verified)")
    return EXIT_OK


def cmd_serve(a, st: Settings, store: SessionStore) -> int:
    rt, s = _runtime(st, store)
    try:
        if a.serve_cmd == "start":
            mgr = ServeManager(rt, a.port)
            h = mgr.start(a.model, a.backend, a.dtype, wait=not a.no_wait,
                          wait_timeout=a.wait_timeout, progress=eprint)
            s.serve = {"port": a.port, "model": a.model if a.backend != "echo" else "echo",
                       "backend": a.backend, "instance": h.get("health", h)["instance"]}
            store.save(s, expected=s)
            print("model server ready" if not a.no_wait else "model server launching")
            return EXIT_OK
        if a.serve_cmd == "logs":
            print("\n".join(ServeManager(rt).logs(a.lines)))
            return EXIT_OK
        if not s.serve and a.serve_cmd == "stop":
            print(json.dumps(ServeManager(rt).stop()))
            return EXIT_OK
        if not s.serve:
            raise KglError("no model server recorded; run `kgl serve start` first")
        mgr = ServeManager(rt, s.serve["port"])
        if a.serve_cmd == "status":
            print(json.dumps(mgr.health(), indent=2))
        elif a.serve_cmd == "stop":
            print(json.dumps(mgr.stop(s.serve.get("instance"))))
            s.serve = None
            store.save(s, expected=s)
        elif a.serve_cmd == "proxy":
            host, _, port = a.listen.rpartition(":")
            key = a.api_key or os.environ.get("KGL_PROXY_KEY")
            srv = make_proxy_server(rt, s.serve["port"], s.serve["model"], host or "127.0.0.1",
                                    int(port), key)
            eprint(f"OpenAI-compatible endpoint: http://{host or '127.0.0.1'}:{port}/v1 "
                   f"(model={s.serve['model']}, auth={'on' if key else 'off'}); Ctrl-C to stop.\n"
                   "Keep this terminal open and send requests from a second terminal.")
            try:
                srv.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                srv.server_close()
        return EXIT_OK
    finally:
        rt.close()


def cmd_infer(a, st: Settings, store: SessionStore) -> int:
    rt, s = _runtime(st, store)
    if not s.serve:
        raise KglError("no model server recorded; run `kgl serve start` first")
    prompt = " ".join(a.prompt) if a.prompt else sys.stdin.read()
    msgs = ([{"role": "system", "content": a.system}] if a.system else []) + \
           [{"role": "user", "content": prompt}]
    payload = {"messages": msgs, "max_tokens": a.max_tokens, "temperature": a.temperature,
               "stream": not a.no_stream}

    def on_chunk(obj: dict) -> None:
        d = (obj.get("choices") or [{}])[0].get("delta", {}).get("content")
        if d and not a.json:
            sys.stdout.write(d)
            sys.stdout.flush()

    try:
        r = infer_chat(rt, s.serve["port"], payload, on_chunk if payload["stream"] else None,
                       timeout=a.timeout)
    finally:
        rt.close()
    if r.status != 200:
        raise ServeError(f"inference failed (status {r.status}): {r.error}")
    if a.json:
        print(json.dumps({"text": r.text, "latency_s": round(r.latency_s, 3),
                          "ttft_s": r.ttft_s and round(r.ttft_s, 3)}))
    else:
        if not payload["stream"]:
            print(r.text)
        else:
            print()
        if a.timing:
            eprint(f"[latency {r.latency_s:.2f}s" +
                   (f", ttft {r.ttft_s:.2f}s]" if r.ttft_s is not None else "]"))
    return EXIT_OK


def cmd_stop(a, st: Settings, store: SessionStore) -> int:
    rt, s = _runtime(st, store)
    try:
        if s.serve and not a.keep_server:
            try:
                print("model server:", json.dumps(ServeManager(rt, s.serve["port"]).stop(s.serve.get("instance"))))
            except KglError as e:
                eprint(f"could not stop model server: {e}")
                return EXIT_ERROR
        if a.kill_kernel:
            try:
                rt.client.delete_kernel(s.kernel_id)
                print("kernel deleted")
            except KglError as e:
                eprint(f"could not delete kernel: {e}")
    finally:
        rt.close()
    store.clear(expected=s)
    print("local session cleared. NOTE: the Kaggle notebook session itself is still running and "
          "counting against your quota - stop it from the Kaggle UI.")
    return EXIT_OK


def cmd_doctor(a, st: Settings, store: SessionStore) -> int:
    rt, s = _runtime(st, store)

    def say(step) -> None:
        print(f"[{'PASS' if step.ok else 'FAIL'}] {step.name:<16} {step.seconds:>7.2f}s  {step.detail}")

    try:
        steps = run_doctor(rt, a.backend, a.model, a.port, a.n, a.keep, a.wait_timeout, say)
    finally:
        rt.close()
    if a.report:
        Path(a.report).write_text(json.dumps(steps_to_dict(steps), indent=2))
        print(f"report written to {a.report}")
    ok = bool(steps) and all(x.ok for x in steps)
    if ok and a.backend == "echo":
        print("\nTransport verified with the echo backend. Re-run with --backend transformers "
              "to verify a real model on the GPU.")
    return EXIT_OK if ok else EXIT_ERROR


# ------------------------------------------------------------------ parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kgl", description="Use a Kaggle GPU notebook from your terminal.")
    p.add_argument("-V", "--version", action="version", version=f"kgl {__version__}")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging to stderr")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("auth", help="check official Kaggle credentials / session").set_defaults(fn=cmd_auth)

    c = sub.add_parser("connect", help="attach using the VS Code Compatible URL from Run > Kaggle Jupyter Server")
    c.add_argument("--proxy-url", help="Jupyter base URL ('-' reads stdin; or $KGL_PROXY_URL)")
    c.add_argument("--header", action="append", help="extra HTTP header 'Name: value' (repeatable)")
    c.add_argument("--kernel-id", help="kernel id (prefix ok)")
    c.add_argument("--new-kernel", action="store_true", help="create a kernel instead of reusing")
    c.set_defaults(fn=cmd_connect)

    s = sub.add_parser("status", help="connection, GPU/RAM, session time, model server")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_status)

    e = sub.add_parser("exec", help="run Python (or --shell) on the VM; code from args, -f or stdin")
    e.add_argument("code", nargs="*")
    e.add_argument("-f", "--file")
    e.add_argument("--shell", action="store_true", help="treat input as a bash command")
    e.add_argument("--timeout", type=float, default=0, help="total seconds (0 = unlimited)")
    e.add_argument("--idle-timeout", type=float, default=0, help="seconds without output (0 = unlimited)")
    e.set_defaults(fn=cmd_exec)

    u = sub.add_parser("upload", help="upload LOCAL to REMOTE (relative -> /kaggle/working)")
    u.add_argument("local")
    u.add_argument("remote")
    u.set_defaults(fn=cmd_upload)
    d = sub.add_parser("download", help="download REMOTE to LOCAL")
    d.add_argument("remote")
    d.add_argument("local")
    d.set_defaults(fn=cmd_download)

    sv = sub.add_parser("serve", help="manage the model server on the VM")
    ss = sv.add_subparsers(dest="serve_cmd", required=True)
    st_ = ss.add_parser("start")
    st_.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    st_.add_argument("--backend", choices=["transformers", "echo"], default="transformers")
    st_.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="float16")
    st_.add_argument("--port", type=int, default=8000, help="VM-local port")
    st_.add_argument("--no-wait", action="store_true")
    st_.add_argument("--wait-timeout", type=float, default=900)
    ss.add_parser("stop")
    ss.add_parser("status")
    lg = ss.add_parser("logs")
    lg.add_argument("-n", "--lines", type=int, default=50)
    px = ss.add_parser("proxy", help="local OpenAI-compatible endpoint backed by the VM server")
    px.add_argument("--listen", default="127.0.0.1:8001")
    px.add_argument("--api-key", help="require this Bearer key (or $KGL_PROXY_KEY)")
    sv.set_defaults(fn=cmd_serve)

    i = sub.add_parser("infer", help="send one chat prompt to the model on the VM")
    i.add_argument("prompt", nargs="*", help="prompt (or stdin)")
    i.add_argument("--system")
    i.add_argument("--max-tokens", type=int, default=256)
    i.add_argument("--temperature", type=float, default=0.7)
    i.add_argument("--no-stream", action="store_true")
    i.add_argument("--json", action="store_true")
    i.add_argument("--timing", action="store_true")
    i.add_argument("--timeout", type=float, default=600)
    i.set_defaults(fn=cmd_infer)

    k = sub.add_parser("stop", help="stop model server, forget the local session")
    k.add_argument("--keep-server", action="store_true")
    k.add_argument("--kill-kernel", action="store_true", help="also DELETE the kernel (may break the notebook)")
    k.set_defaults(fn=cmd_stop)

    dr = sub.add_parser("doctor", help="run the PoC spikes and print a pass/fail report")
    dr.add_argument("--backend", choices=["echo", "transformers"], default="echo")
    dr.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    dr.add_argument("--port", type=int, default=8000)
    dr.add_argument("-n", type=int, default=20, help="number of inference requests")
    dr.add_argument("--keep", action="store_true", help="leave the server running")
    dr.add_argument("--wait-timeout", type=float, default=900)
    dr.add_argument("--report", help="write JSON report to this path")
    dr.set_defaults(fn=cmd_doctor)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    st = Settings.from_env()
    store = SessionStore(st.home)
    try:
        return args.fn(args, st, store)
    except KeyboardInterrupt:
        return 130
    except ExecutionError as e:
        eprint(f"remote error: {e}")
        return EXIT_REMOTE
    except (ConnectionLost, AuthError, KernelGone, TimeoutError) as e:
        eprint(f"connection error: {e}")
        return EXIT_CONNECTION
    except NoSession as e:
        eprint(f"error: {e}")
        return EXIT_CONNECTION
    except KglError as e:
        eprint(f"error: {e}")
        return EXIT_ERROR
