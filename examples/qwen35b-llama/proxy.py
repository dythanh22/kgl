"""Manage the local Qwen chat proxy on Linux after ready.py succeeds."""
import argparse
from contextlib import ExitStack
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

from kgl.client.model_fetch import read_preset
from kgl.core.config import Settings
from kgl.core.errors import KglError
from kgl.core.state import SessionStore
from ready import PRESET

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


def check_owner(pid, home, from_home=None):
    proc = Path("/proc") / str(pid)
    args = (proc / "cmdline").read_bytes().rstrip(b"\0").decode().split("\0")
    cwd = (proc / "cwd").resolve(strict=True)
    if (proc.stat().st_uid != os.getuid() or not (proc / "exe").samefile(sys.executable)
            or len(args) != 7 or args[1:6] != ["-m", "kgl", "serve", "proxy", "--static"]
            or (cwd / args[6]).resolve() != STATIC):
        raise RuntimeError(f"Port 8001 belongs to another application (PID {pid})")
    env = dict(entry.split(b"=", 1) for entry in (proc / "environ").read_bytes().split(b"\0") if entry)
    owner_home = cwd / os.fsdecode(env.get(b"KGL_HOME") or Path.home() / ".config/kgl")
    owner_home = owner_home.resolve()
    if owner_home not in (home.resolve(), from_home):
        raise RuntimeError(f"Port 8001 belongs to another KGL_HOME (PID {pid})")
    if env.get(b"KGL_PROXY_KEY", b"") != os.environ.get("KGL_PROXY_KEY", "").encode():
        raise RuntimeError("Port 8001 proxy uses a different KGL_PROXY_KEY")
    return owner_home


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


def matches(result, worker):
    status, data = result
    health = data.get("health", {})
    return (status == 200 and data.get("state") == "ready"
            and health.get("instance") == worker.instance and health.get("model") == worker.model
            and health.get("backend") == worker.backend
            and (not worker.alias or (health.get("alias"), health.get("quant")) == (worker.alias, worker.quant)))


def target_ready(worker):
    raw = subprocess.check_output([sys.executable, "-m", "kgl", "serve", "status"], text=True)
    if not matches((200, json.loads(raw)), worker):
        raise RuntimeError("Target worker is not ready. Run ready.py first.")


def manage(home, worker, from_home=None):
    pid = listener()
    if pid is not None:
        fd = open_pidfd(pid)
        try:
            owner = check_owner(pid, home, from_home)
            handoff = from_home is not None and owner == from_home
            result = probe()
            if not handoff and matches(result, worker):
                return "reused", pid
            data = result[1]
            if data.get("state") == "busy":
                raise RuntimeError("Proxy worker busy. Retry after the active request.")
            if handoff and (result[0], data.get("state")) != (503, "connection_lost"):
                raise RuntimeError("Previous session proxy is still serving or has not lost its connection")
            if data.get("state") != "wrong_instance" and data.get("health", {}).get("instance") == worker.instance:
                raise RuntimeError("Expected worker is not ready. Retry after checking ready.py.")
            target_ready(worker)
            if handoff:
                status, data = probe()
                if (status, data.get("state")) != (503, "connection_lost"):
                    raise RuntimeError("Previous session proxy recovered during the handoff")
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
        target_ready(worker)

    with (home / "qwen-proxy.log").open("ab") as log:
        child = subprocess.Popen([sys.executable, "-m", "kgl", "serve", "proxy",
                                  "--static", str(STATIC)], stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=log, start_new_session=True)
    try:
        for _ in range(50):
            if child.poll() is not None:
                raise RuntimeError(f"Proxy startup failed. See {log.name}")
            if listener() == child.pid:
                if matches(probe(), worker):
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-home", type=Path, help="previous KGL_HOME for a disconnected proxy on port 8001")
    args = parser.parse_args()
    home = Settings.from_env().home.resolve()
    from_home = args.from_home.expanduser().resolve() if args.from_home else None
    if from_home == home:
        parser.error("--from-home must differ from the current KGL_HOME")
    homes = {home, from_home} if from_home else {home}
    if not all(item.is_dir() for item in homes):
        raise RuntimeError("Check both session directories and run ready.py for the target")
    preset = read_preset(PRESET)
    with ExitStack() as locks:
        # Acquire both locks in the same order for handoffs in either direction.
        for item in sorted(homes):
            lock = locks.enter_context((item / "qwen-proxy.lock").open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX)
        session = SessionStore(home).load()
        worker = session.serve if session else None
        if (not worker or not worker.instance or worker.backend != "llama"
                or worker.model not in preset["inputs"].values()
                or (worker.alias is not None and (
                    worker.alias != preset["alias"] or preset["inputs"].get(worker.quant) != worker.model))):
            raise RuntimeError("Run ready.py first to select the Qwen worker")
        action, pid = manage(home, worker, from_home)
        print(f"Qwen proxy {action}: PID {pid}, instance {worker.instance}\nChat {URL}/\nAPI {URL}/v1")


if __name__ == "__main__":
    try:
        main()
    except (KglError, RuntimeError, OSError, ValueError, subprocess.SubprocessError) as exc:
        sys.exit(str(exc))
