import base64, os
p, off, n = __PATH__, __OFF__, __N__
size = os.path.getsize(p)
with open(p, "rb") as f:
    f.seek(off); data = f.read(n)
_emit({"size": size, "offset": off, "data": base64.b64encode(data).decode()})
