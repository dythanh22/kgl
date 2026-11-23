"""Execute Python or shell commands on the notebook."""

from pathlib import Path
import re
import sys

from ...core.config import Settings
from ...core.errors import KglError
from ...core.state import SessionStore
from ...remote import SHELL, new_marker, render
from ..common import EXIT_OK, EXIT_REMOTE, open_runtime, write_stream, eprint

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _read_code(a) -> str:
    if a.file:
        return Path(a.file).read_text()
    if a.code:
        return " ".join(a.code)
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise KglError("no code given: pass code, -f FILE, or pipe via stdin")


def cmd_exec(a, st: Settings, store: SessionStore) -> int:
    rt, s = open_runtime(st, store)
    code = _read_code(a)
    execution = dict(idle_timeout=a.idle_timeout or None, total_timeout=a.timeout or None,
                     capture_output=False)
    try:
        if a.shell:
            marker = new_marker()
            code = render(SHELL, marker=marker, CWD=st.workdir, CMD=code)
            res, markers = rt.run_markers(code, write_stream, marker=marker, **execution)
        else:
            res = rt.run(code, write_stream, **execution)
    except KeyboardInterrupt:
        if a.interrupt_kernel:
            eprint("\ninterrupting remote kernel...")
            try:
                rt.client.interrupt_kernel(s.kernel_id)
            except KglError as e:
                eprint(f"interrupt failed: {e}")
        else:
            eprint("\nstopped waiting; remote execution may continue")
        return 130
    finally:
        rt.close()
    if res.error:
        eprint(ANSI.sub("", "\n".join(res.error["traceback"])) or
               f"{res.error['ename']}: {res.error['evalue']}")
        return EXIT_REMOTE
    if res.status != "ok":
        eprint(f"remote execution status: {res.status}")
        return EXIT_REMOTE
    if a.shell:
        if not markers or "rc" not in markers[-1]:
            raise KglError("remote shell completed without an exit status")
        rc = int(markers[-1]["rc"])
        return rc if rc >= 0 else 128 - rc
    return EXIT_OK
