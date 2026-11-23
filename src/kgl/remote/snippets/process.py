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
        if (len(argv) < 2 or argv[1] not in (str(d / "serve.py"), str(d / "serve.pyz"))
                or os.getpgid(pid) != pid):
            raise RuntimeError("stale PID belongs to another process")
        if saved and (saved["pid"] != pid or saved["start_ticks"] != fields[19]):
            raise RuntimeError("server PID identity changed")
        return pid, argv[2:-1], saved
    except (FileNotFoundError, ProcessLookupError):
        return None
