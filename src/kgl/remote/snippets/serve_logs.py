from collections import deque
d, n = __DIR__, __N__
try:
    with open(d + "/serve.log", errors="replace") as f:
        _emit({"lines": [line.rstrip("\n") for line in deque(f, maxlen=n)]})
except Exception as e:
    _emit({"lines": [], "error": repr(e)[:200]})
