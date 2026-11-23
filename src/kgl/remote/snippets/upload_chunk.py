import base64, os
p = __PATH__
os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
with os.fdopen(os.open(p, os.O_WRONLY | os.O_CREAT, 0o600), "wb") as f:
    f.seek(__OFF__)
    _emit({"written": f.write(base64.b64decode(__DATA__))})
