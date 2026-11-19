"""Upload/download through the kernel (chunked base64). Works on any Jupyter
kernel; does not depend on the Contents API (unverified on Kaggle)."""
from __future__ import annotations

import base64
import hashlib
import os
import posixpath
import tempfile
import uuid
from pathlib import Path
from typing import Callable

from .errors import KglError
from .remote import COMMIT_UPLOAD, DISCARD_UPLOAD, DOWNLOAD_CHUNK, STAT, UPLOAD_CHUNK, resolve_path
from .runtime import Runtime

Progress = Callable[[int, int], None]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def upload_file(rt: Runtime, local: Path, remote: str, progress: Progress | None = None) -> dict:
    local = Path(local)
    if not local.is_file():
        raise KglError(f"local file not found: {local}")
    rpath = resolve_path(remote, rt.settings.workdir)
    if remote.endswith("/"):
        rpath = rpath + local.name
    size, chunk = local.stat().st_size, rt.settings.chunk_kb * 1024
    sent = 0
    tmp = posixpath.join(posixpath.dirname(rpath), f".kgl-upload-{uuid.uuid4().hex}")
    digest = hashlib.sha256()
    try:
        with open(local, "rb") as f:
            while True:
                data = f.read(chunk)
                if not data and sent:
                    break
                reply = rt.call(UPLOAD_CHUNK, idempotent=True, PATH=tmp, OFF=sent,
                                DATA=base64.b64encode(data).decode())
                if reply.get("written") != len(data):
                    raise KglError(f"short upload write for {rpath}")
                digest.update(data)
                sent += len(data)
                if progress:
                    progress(sent, size)
                if not data:
                    break
        st = rt.call(COMMIT_UPLOAD, PATH=tmp, TARGET=rpath, SIZE=size, HASH=digest.hexdigest())
    except (Exception, KeyboardInterrupt):
        try:
            rt.call(DISCARD_UPLOAD, idempotent=True, timeout=2, PATH=tmp)
        except (KglError, OSError):
            pass
        raise
    return {"remote": rpath, **st}


def download_file(rt: Runtime, remote: str, local: Path, progress: Progress | None = None) -> dict:
    rpath = resolve_path(remote, rt.settings.workdir)
    st = rt.call(STAT, idempotent=True, PATH=rpath)
    if not st.get("exists"):
        raise KglError(f"remote file not found: {rpath}")
    local = Path(local)
    if local.is_dir():
        local = local / os.path.basename(rpath)
    local.parent.mkdir(parents=True, exist_ok=True)
    chunk, off, size = rt.settings.chunk_kb * 1024, 0, st["size"]
    fd, name = tempfile.mkstemp(prefix=".kgl-download-", dir=local.parent)
    tmp = Path(name)
    try:
        with os.fdopen(fd, "wb") as f:
            while off < size:
                r = rt.call(DOWNLOAD_CHUNK, idempotent=True, PATH=rpath, OFF=off, N=min(chunk, size - off))
                data = base64.b64decode(r["data"], validate=True)
                if not data or r.get("offset") != off or r.get("size") != size:
                    raise KglError(f"remote file changed during download: {rpath}")
                f.write(data)
                off += len(data)
                if progress:
                    progress(off, size)
        if off != size or _sha256(tmp) != st["sha256"]:
            raise KglError(f"download verification failed for {rpath}")
        os.replace(tmp, local)
    finally:
        tmp.unlink(missing_ok=True)
    return {"local": str(local), "size": size, "sha256": st["sha256"]}
