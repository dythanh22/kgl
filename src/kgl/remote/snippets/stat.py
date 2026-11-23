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
