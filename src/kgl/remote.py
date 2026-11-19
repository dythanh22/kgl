"""Code snippets executed *on the Kaggle VM* through the kernel, plus the
marker protocol used to return structured data inside normal stdout."""
from __future__ import annotations

import json
import posixpath
import re
import uuid
from textwrap import indent
from typing import Callable

MARKER = "@@KGL@@"

PRELUDE = (
    "import json as _j, sys as _s\n"
    "def _emit(o):\n"
    f"    _s.stdout.write({MARKER!r} + _j.dumps(o) + '\\n'); _s.stdout.flush()\n"
)


def new_marker() -> str:
    return MARKER + uuid.uuid4().hex + "@@"


def render(template: str, *, marker: str = MARKER, **values: object) -> str:
    """Substitute __NAME__ placeholders with repr() of values (safe quoting)."""
    out = re.sub(r"__([A-Z][A-Z0-9_]*?)__",
                 lambda match: repr(values[match[1]]) if match[1] in values else match[0],
                 template)
    return f"exec({(PRELUDE.replace(MARKER, marker) + out)!r}, {{}})"


class MarkerParser:
    """Splits a text stream into passthrough text and `@@KGL@@{json}` events."""

    def __init__(self, on_marker: Callable[[dict], None] | None = None,
                 marker: str = MARKER) -> None:
        self.markers: list[dict] = []
        self._on_marker = on_marker
        self._buf = ""
        self._marker = marker
        self._line_start = True

    def feed(self, text: str) -> str:
        self._buf += text
        out: list[str] = []
        while (i := self._buf.find("\n")) != -1:
            line, self._buf = self._buf[:i], self._buf[i + 1:]
            at_start, self._line_start = self._line_start, True
            if at_start and line.startswith(self._marker):
                try:
                    obj = json.loads(line[len(self._marker):])
                    if not isinstance(obj, dict):
                        raise ValueError("marker must contain an object")
                except ValueError:
                    out.append(line + "\n")
                    continue
                self.markers.append(obj)
                if self._on_marker:
                    self._on_marker(obj)
            else:
                out.append(line + "\n")
        if self._buf and (not self._line_start
                          or not self._marker.startswith(self._buf[: len(self._marker)])):
            out.append(self._buf)
            self._buf = ""
            self._line_start = False
        return "".join(out)

    def flush(self) -> str:
        rest, self._buf = self._buf, ""
        return rest


def last_marker(text: str, marker: str = MARKER) -> dict | None:
    p = MarkerParser(marker=marker)
    p.feed(text if text.endswith("\n") else text + "\n")
    return p.markers[-1] if p.markers else None


def resolve_path(path: str, workdir: str) -> str:
    return path if path.startswith("/") else posixpath.join(workdir, path)


PROBE = '''
import os, subprocess, shutil, time, platform
info = {"python": platform.python_version(), "gpus": []}
try:
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15)
    if out.returncode == 0:
        for line in out.stdout.strip().splitlines():
            n, u, t, g = [x.strip() for x in line.split(",")]
            info["gpus"].append({"name": n, "mem_used_mb": int(u), "mem_total_mb": int(t), "util_pct": int(g)})
    else:
        info["gpu_error"] = out.stderr.strip()[:200]
except FileNotFoundError:
    info["gpu_error"] = "nvidia-smi not found"
except Exception as e:
    info["gpu_error"] = repr(e)[:200]
try:
    mem = {}
    for l in open("/proc/meminfo"):
        k, v = l.split(":", 1); mem[k] = int(v.split()[0])
    info["ram_total_mb"] = mem["MemTotal"] // 1024
    info["ram_avail_mb"] = mem.get("MemAvailable", 0) // 1024
except Exception:
    pass
try:
    wd = __WORKDIR__
    du = shutil.disk_usage(wd if os.path.isdir(wd) else "/")
    info["disk_free_gb"] = round(du.free / 1e9, 1)
except Exception:
    pass
try:
    clk = os.sysconf("SC_CLK_TCK")
    start_ticks = int(open("/proc/1/stat").read().rsplit(")", 1)[1].split()[19])
    btime = [int(l.split()[1]) for l in open("/proc/stat") if l.startswith("btime")][0]
    info["container_uptime_s"] = int(time.time() - (btime + start_ticks / clk))
except Exception:
    pass
_emit(info)
'''

UPLOAD_CHUNK = '''
import base64, os
p = __PATH__
os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
with os.fdopen(os.open(p, os.O_WRONLY | os.O_CREAT, 0o600), "wb") as f:
    f.seek(__OFF__)
    _emit({"written": f.write(base64.b64decode(__DATA__))})
'''

COMMIT_UPLOAD = '''
import hashlib, os
p = __PATH__
h = hashlib.sha256()
with open(p, "rb") as f:
    for b in iter(lambda: f.read(1 << 20), b""):
        h.update(b)
    size = f.tell()
if size != __SIZE__ or h.hexdigest() != __HASH__:
    raise ValueError("upload verification failed (size/sha256 mismatch)")
os.replace(p, __TARGET__)
_emit({"size": size, "sha256": h.hexdigest()})
'''

DISCARD_UPLOAD = '''
from pathlib import Path
Path(__PATH__).unlink(missing_ok=True)
_emit({"ok": True})
'''

STAT = '''
import hashlib, os
p = __PATH__
if os.path.isfile(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    _emit({"exists": True, "size": os.path.getsize(p), "sha256": h.hexdigest()})
else:
    _emit({"exists": False})
'''

DOWNLOAD_CHUNK = '''
import base64, os
p, off, n = __PATH__, __OFF__, __N__
size = os.path.getsize(p)
with open(p, "rb") as f:
    f.seek(off); data = f.read(n)
_emit({"size": size, "offset": off, "data": base64.b64encode(data).decode()})
'''

SHELL = '''
import codecs, subprocess, sys, os
cwd = __CWD__
p = subprocess.Popen(__CMD__, shell=True, executable="/bin/bash", stdout=subprocess.PIPE,
                     stdin=subprocess.DEVNULL, stderr=subprocess.STDOUT,
                     cwd=cwd if os.path.isdir(cwd) else None)
decode = codecs.getincrementaldecoder("utf-8")("replace")
for chunk in iter(lambda: p.stdout.read1(4096), b""):
    sys.stdout.write(decode.decode(chunk)); sys.stdout.flush()
sys.stdout.write(decode.decode(b"", final=True))
p.wait()
sys.stdout.write("\\n"); _emit({"rc": p.returncode})
'''

_PROCESS = '''
import json, os, signal, subprocess, sys, time, uuid
from pathlib import Path
import fcntl
d = Path(__DIR__)
d.mkdir(parents=True, exist_ok=True)
pidf, metadata = d / "serve.pid", d / "serve.json"

def inspect_process():
    saved = json.loads(metadata.read_text()) if metadata.exists() else {}
    if not saved and not pidf.exists():
        return None
    pid = saved["pid"] if saved else int(pidf.read_text())
    proc = Path("/proc") / str(pid)
    try:
        fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        argv = (proc / "cmdline").read_bytes().decode().split(chr(0))
        if len(argv) < 2 or argv[1] != str(d / "serve.py") or os.getpgid(pid) != pid:
            raise RuntimeError("stale PID belongs to another process")
        if saved and (saved["pid"] != pid or saved["start_ticks"] != fields[19]):
            raise RuntimeError("server PID identity changed")
        return pid, argv[2:-1], saved
    except (FileNotFoundError, ProcessLookupError):
        return None
'''

def _locked_process_operation(operation: str) -> str:
    return (_PROCESS + 'with (d / "serve.lock").open("a") as lock:\n'
            + '    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)\n'
            + indent(operation, "    "))


SERVE_START = _locked_process_operation('''
running = inspect_process()
args = __ARGS__
if running:
    pid, argv, saved = running
    if saved.get("protocol") != 2:
        raise RuntimeError("legacy server detected; stop it before upgrading")
    if args and argv != args + ["--instance", saved["instance"]]:
        raise RuntimeError("running server configuration conflicts with requested configuration")
    _emit(dict(saved, already_running=True))
else:
    instance = uuid.uuid4().hex
    cmd = [sys.executable, str(d / "serve.py")] + args + ["--instance", instance]
    with (d / "serve.log").open("ab") as lf:
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             start_new_session=True, cwd=d)
    pending = d / f".serve-{instance}.json"
    try:
        fields = Path(f"/proc/{p.pid}/stat").read_text().rsplit(")", 1)[1].split()
        saved = dict(pid=p.pid, start_ticks=fields[19], instance=instance, args=args, protocol=2)
        pending.write_text(json.dumps(saved))
        pending.replace(metadata)
    except BaseException:
        try:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        raise
    finally:
        pending.unlink(missing_ok=True)
    _emit(dict(saved, started=True))
''')

SERVE_HEALTH = '''
import json, os, urllib.request
from collections import deque
d, port = __DIR__, __PORT__
try:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
        h = json.loads(r.read().decode())
    _emit({"state": h.get("status", "unknown"), "health": h})
except Exception as e:
    tail = []
    try:
        with open(d + "/serve.log", errors="replace") as f:
            tail = [line.rstrip("\\n") for line in deque(f, maxlen=15)]
    except Exception:
        pass
    _emit({"state": "down", "error": repr(e)[:200], "log_tail": tail})
'''

SERVE_STOP = _locked_process_operation('''
running = inspect_process()
if running:
    pid, _, saved = running
    if __INSTANCE__ and saved.get("instance") != __INSTANCE__:
        raise RuntimeError("refusing to stop a different server instance")
    os.killpg(pid, signal.SIGTERM)
    for attempt in range(60):
        if inspect_process() is None:
            break
        if attempt == 50:
            os.killpg(pid, signal.SIGKILL)
        time.sleep(0.2)
    else:
        raise RuntimeError("server did not stop")
pidf.unlink(missing_ok=True)
metadata.unlink(missing_ok=True)
_emit({"stopped": True, "pid": running[0] if running else None})
''')


SERVE_LOGS = '''
from collections import deque
d, n = __DIR__, __N__
try:
    with open(d + "/serve.log", errors="replace") as f:
        _emit({"lines": [line.rstrip("\\n") for line in deque(f, maxlen=n)]})
except Exception as e:
    _emit({"lines": [], "error": repr(e)[:200]})
'''

INFER = '''
import base64, json, urllib.request, urllib.error
payload = base64.b64decode(__PAYLOAD__)
options = __OPTIONS__
stream = not options.get("predict") and json.loads(payload).get("stream") is True
headers = {"Content-Type": "application/json", "X-Kgl-Timeout": str(__TIMEOUT__)}
if options.get("request_id"):
    headers["X-Kgl-Request-Id"] = options["request_id"]
instance = options.get("instance")
route = "/v1/chat/completions"
if instance:
    headers["X-Kgl-Instance"] = instance
    route = "/internal/predict" if options.get("predict") else "/internal/chat/completions"
req = urllib.request.Request(f"http://127.0.0.1:{__PORT__}" + route, data=payload, headers=headers)
try:
    with urllib.request.urlopen(req, timeout=__TIMEOUT__) as r:
        if stream:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if line.startswith("data:"):
                    chunk = line[5:].strip()
                    if chunk == "[DONE]":
                        _emit({"status": 200, "done": True})
                        break
                    obj = json.loads(chunk)
                    if "error" in obj:
                        _emit({"status": 0, "error": obj["error"]["message"]})
                        break
                    _emit({"chunk": chunk})
            else:
                raise ValueError("incomplete stream: upstream closed before [DONE]")
        else:
            _emit({"status": 200, "body": json.loads(r.read().decode())})
except urllib.error.HTTPError as e:
    raw = e.read(2000).decode("utf-8", "replace")
    try:
        raw = json.loads(raw)["error"]["message"]
    except (ValueError, KeyError, TypeError):
        pass
    if instance and e.code == 404:
        raw = "running server needs upgrading; run kgl serve stop and kgl serve start"
    _emit({"status": e.code, "error": raw})
except Exception as e:
    _emit({"status": 0, "error": repr(e)[:500]})
'''

CANCEL_INFER = '''
import json, urllib.error, urllib.request
req = urllib.request.Request(f"http://127.0.0.1:{__PORT__}/cancel", data=b"",
    headers={"X-Kgl-Instance": __INSTANCE__, "X-Kgl-Request-Id": __REQUEST_ID__})
try:
    with urllib.request.urlopen(req, timeout=2.5) as reply:
        _emit(json.load(reply))
except (OSError, ValueError):
    _emit({"cancelled": False})
'''
