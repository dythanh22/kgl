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
            tail = [line.rstrip("\n") for line in deque(f, maxlen=15)]
    except Exception:
        pass
    state = "down"
    try:
        from pathlib import Path
        saved = json.loads((Path(d) / "serve.json").read_text())
        fields = Path(f"/proc/{saved['pid']}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] != "Z" and saved["start_ticks"] == fields[19]:
            state = "starting"
    except (OSError, ValueError, KeyError, IndexError):
        pass
    _emit({"state": state, "error": repr(e)[:200], "log_tail": tail})
