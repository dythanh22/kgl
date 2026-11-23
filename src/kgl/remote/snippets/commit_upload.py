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
