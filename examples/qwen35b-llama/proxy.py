"""Manage the local Qwen chat proxy on Linux after ready.py succeeds."""
import ctypes
import fcntl
import json
import os
from pathlib import Path
import re
import select
import signal
import subprocess
import sys
import time
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

from kgl.core.config import Settings
from kgl.core.state import SessionStore
from ready import MODEL

STATIC = Path(__file__).resolve().parents[1] / "chat-demo"
PORT = 8001
URL = f"http://127.0.0.1:{PORT}"


def open_pidfd(pid):
    if hasattr(os, "pidfd_open"):
        return os.pidfd_open(pid)
    # Some conda Python builds omit the wrapper even on modern Linux hosts.
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.pidfd_open(pid, 0)
    if fd < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return fd


def listener():
    lines = subprocess.check_output(["ss", "-Hltnp", f"sport = :{PORT}"], text=True).splitlines()
    if not lines:
        return None
    if len(lines) == 1 and lines[0].split()[3] == f"127.0.0.1:{PORT}":
        pids = re.findall(r"pid=(\d+)", lines[0])
        if len(pids) == 1:
            return int(pids[0])
    raise RuntimeError("Port 8001 is occupied by another or unidentified listener")


def check_owner(pid, home):
    proc = Path("/proc") / str(pid)
    args = (proc / "cmdline").read_bytes().rstrip(b"\0").decode().split("\0")
    cwd = (proc / "cwd").resolve(strict=True)
    if (proc.stat().st_uid != os.getuid() or not (proc / "exe").samefile(sys.executable)
            or len(args) != 7 or args[1:6] != ["-m", "kgl", "serve", "proxy", "--static"]
            or (cwd / args[6]).resolve() != STATIC):
        raise RuntimeError(f"Port 8001 belongs to another application (PID {pid})")
    env = dict(entry.split(b"=", 1) for entry in (proc / "environ").read_bytes().split(b"\0") if entry)
    owner_home = cwd / os.fsdecode(env.get(b"KGL_HOME") or Path.home() / ".config/kgl")
    if owner_home.resolve() != home.resolve():
        raise RuntimeError(f"Port 8001 belongs to another KGL_HOME (PID {pid})")
    if env.get(b"KGL_PROXY_KEY", b"") != os.environ.get("KGL_PROXY_KEY", "").encode():
        raise RuntimeError("Port 8001 proxy uses a different KGL_PROXY_KEY")


def probe():
    request = Request(URL + "/ready")
    key = os.environ.get("KGL_PROXY_KEY")
    if key:
        request.add_header("Authorization", "Bearer " + key)
    try:
        response = build_opener(ProxyHandler({})).open(request, timeout=10)
    except HTTPError as exc:
        if exc.code != 503:
            raise
        response = exc
    with response:
        return response.status, json.load(response)


def matches(result, instance):
    status, data = result
    health = data.get("health", {})
    return (status == 200 and data.get("state") == "ready"
            and health.get("instance") == instance and health.get("model") == MODEL)


def target_ready(instance):
    raw = subprocess.check_output([sys.executable, "-m", "kgl", "serve", "status"], text=True)
    if not matches((200, json.loads(raw)), instance):
        raise RuntimeError("Target worker is not ready. Run ready.py first.")


def manage(home, instance):
    pid = listener()
    if pid is not None:
        fd = open_pidfd(pid)
        try:
            check_owner(pid, home)
            result = probe()
            if matches(result, instance):
                return "reused", pid
            data = result[1]
            if data.get("state") == "busy":
                raise RuntimeError("Proxy worker busy. Retry after the active request.")
            if data.get("state") != "wrong_instance" and data.get("health", {}).get("instance") == instance:
                raise RuntimeError("Expected worker is not ready. Retry after checking ready.py.")
            target_ready(instance)
            if listener() != pid:
                raise RuntimeError("Port 8001 changed owner during readiness check")
            signal.pidfd_send_signal(fd, signal.SIGTERM)
            poll = select.poll()
            poll.register(fd, select.POLLIN)
            if not poll.poll(5000):
                raise RuntimeError("Previous proxy has not exited")
        finally:
            os.close(fd)
    else:
        target_ready(instance)

    with (home / "qwen-proxy.log").open("ab") as log:
        child = subprocess.Popen([sys.executable, "-m", "kgl", "serve", "proxy",
                                  "--static", str(STATIC)], stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=log, start_new_session=True)
    try:
        for _ in range(50):
            if child.poll() is not None:
                raise RuntimeError(f"Proxy startup failed. See {log.name}")
            if listener() == child.pid:
                if matches(probe(), instance):
                    return ("switched" if pid else "started"), child.pid
                raise RuntimeError("New proxy did not confirm the expected worker instance")
            time.sleep(0.1)
        raise RuntimeError("Proxy did not bind port 8001 in time")
    except BaseException:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=5)
        raise


def main():
    home = Settings.from_env().home
    if not home.is_dir():
        raise RuntimeError("Connect to Kaggle and run ready.py first")
    with (home / "qwen-proxy.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        session = SessionStore(home).load()
        worker = session.serve if session else None
        if not worker or worker.model != MODEL or not worker.instance or worker.backend != "llama":
            raise RuntimeError("Run ready.py first to select the Qwen worker")
        action, pid = manage(home, worker.instance)
        print(f"Qwen proxy {action}: PID {pid}, instance {worker.instance}\nChat {URL}/\nAPI {URL}/v1")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as exc:
        sys.exit(str(exc))
