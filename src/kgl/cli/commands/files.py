"""Upload and download notebook files."""

from pathlib import Path
import sys

from ...client.files import download_file, upload_file
from ...core.config import Settings
from ...core.state import SessionStore
from ..common import EXIT_OK, open_runtime


def _progress(label: str):
    if not sys.stderr.isatty():
        return None

    def p(done: int, total: int) -> None:
        pct = 100 if total == 0 else done * 100 // total
        sys.stderr.write(f"\r{label}: {pct:3d}% ({done}/{total} bytes)")
        if done >= total:
            sys.stderr.write("\n")
        sys.stderr.flush()
    return p


def cmd_upload(a, st: Settings, store: SessionStore) -> int:
    rt, _ = open_runtime(st, store)
    try:
        r = upload_file(rt, Path(a.local), a.remote, _progress("upload"))
    finally:
        rt.close()
    print(f"uploaded {r['size']} bytes -> {r['remote']} (sha256 verified)")
    return EXIT_OK


def cmd_download(a, st: Settings, store: SessionStore) -> int:
    rt, _ = open_runtime(st, store)
    try:
        r = download_file(rt, a.remote, Path(a.local), _progress("download"))
    finally:
        rt.close()
    print(f"downloaded {r['size']} bytes -> {r['local']} (sha256 verified)")
    return EXIT_OK
