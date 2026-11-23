import os
p = __PATH__
_emit({"exists": os.path.isfile(p), "size": os.path.getsize(p) if os.path.isfile(p) else 0})
