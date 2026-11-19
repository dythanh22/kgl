"""Upload/download through the kernel (chunked base64). Works on any Jupyter
kernel; does not depend on the Contents API (unverified on Kaggle)."""
from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path
from typing import Callable

from .errors import KglError
from .remote import DOWNLOAD_CHUNK, STAT, UPLOAD_CHUNK, resolve_path
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
    with open(local, "rb") as f:
        first = True
        while True:
            data = f.read(chunk)
            if not data and not first:
                break
            rt.call(UPLOAD_CHUNK, PATH=rpath, MODE="wb" if first else "ab",
                    DATA=base64.b64encode(data).decode())
            first = False
            sent += len(data)
            if progress:
                progress(sent, size)
            if not data:
                break
    st = rt.call(STAT, PATH=rpath)
    if not st.get("exists") or st["size"] != size or st["sha256"] != _sha256(local):
        raise KglError(f"upload verification failed for {rpath} (size/sha256 mismatch)")
    return {"remote": rpath, "size": size, "sha256": st["sha256"]}


def download_file(rt: Runtime, remote: str, local: Path, progress: Progress | None = None) -> dict:
    rpath = resolve_path(remote, rt.settings.workdir)
    st = rt.call(STAT, PATH=rpath)
    if not st.get("exists"):
        raise KglError(f"remote file not found: {rpath}")
    local = Path(local)
    if local.is_dir():
        local = local / os.path.basename(rpath)
    local.parent.mkdir(parents=True, exist_ok=True)
    tmp = local.with_name(local.name + ".part")
    chunk, off, size = rt.settings.chunk_kb * 1024, 0, st["size"]
    with open(tmp, "wb") as f:
        while off < size:
            r = rt.call(DOWNLOAD_CHUNK, PATH=rpath, OFF=off, N=chunk)
            data = base64.b64decode(r["data"])
            if not data:
                break
            f.write(data)
            off += len(data)
            if progress:
                progress(off, size)
    if off != size or _sha256(tmp) != st["sha256"]:
        tmp.unlink(missing_ok=True)
        raise KglError(f"download verification failed for {rpath}")
    os.replace(tmp, local)
    return {"local": str(local), "size": size, "sha256": st["sha256"]}
