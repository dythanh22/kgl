"""Parse CLI arguments and dispatch commands."""

from __future__ import annotations

import argparse
import logging
import sys

from ..backends import BACKENDS, DEFAULT_MODEL, DOCTOR_BACKENDS, INSTALLABLE_BACKENDS, OPTIONS
from .. import __version__
from ..client.model_fetch import INSTALL_DIR, MODEL_DIR
from ..core.config import Settings, seconds
from ..core.errors import AuthError, ConnectionLost, ExecutionError, KernelGone, KglError, NoSession
from ..core.state import SessionStore
from .commands.doctor import cmd_doctor
from .commands.execute import cmd_exec
from .commands.files import cmd_download, cmd_upload
from .commands.forward import forward
from .commands.infer import cmd_infer
from .commands.models import fetch as fetch_model, install_model_backend, plan_model
from .commands.serve import start, stop, status, logs, proxy
from .commands.session import cmd_auth, cmd_connect, cmd_status, cmd_stop
from .common import EXIT_CONNECTION, EXIT_ERROR, EXIT_REMOTE, eprint


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kgl", description="Use a Kaggle GPU notebook from your terminal.")
    p.add_argument("-V", "--version", action="version", version=f"kgl {__version__}")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging to stderr")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("auth", help="show how to connect a notebook").set_defaults(fn=cmd_auth)

    c = sub.add_parser("connect", help="attach using the VS Code Compatible URL from Run > Kaggle Jupyter Server")
    c.add_argument("--proxy-url", help="Jupyter base URL ('-' reads stdin; or $KGL_PROXY_URL)")
    c.add_argument("--header", action="append", help="extra HTTP header 'Name: value' (repeatable)")
    c.add_argument("--kernel-id", help="kernel id (prefix ok)")
    c.add_argument("--new-kernel", action="store_true", help="create a kernel instead of reusing")
    c.set_defaults(fn=cmd_connect)

    s = sub.add_parser("status", help="connection, GPU/RAM, session time, model server")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_status)

    e = sub.add_parser("exec", help="run Python (or --shell) on the VM; code from args, -f or stdin")
    e.add_argument("code", nargs="*")
    e.add_argument("-f", "--file")
    e.add_argument("--shell", action="store_true", help="treat input as a bash command")
    e.add_argument("--timeout", type=float, default=0, help="total seconds (0 = unlimited)")
    e.add_argument("--idle-timeout", type=float, default=0, help="seconds without output (0 = unlimited)")
    e.add_argument("--interrupt-kernel", action="store_true", help="interrupt the shared kernel on Ctrl-C")
    e.set_defaults(fn=cmd_exec)

    u = sub.add_parser("upload", help="upload LOCAL to REMOTE (relative -> /kaggle/working)")
    u.add_argument("local")
    u.add_argument("remote")
    u.set_defaults(fn=cmd_upload)
    d = sub.add_parser("download", help="download REMOTE to LOCAL")
    d.add_argument("remote")
    d.add_argument("local")
    d.set_defaults(fn=cmd_download)

    sv = sub.add_parser("serve", help="manage the model server on the VM")
    ss = sv.add_subparsers(dest="serve_cmd", required=True)
    st_ = ss.add_parser("start")
    st_.add_argument("--model", help="Hub model ID or checkpoint path on the VM")
    st_.add_argument("--backend", choices=BACKENDS)
    st_.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="float16")
    st_.add_argument("--port", type=int, default=8000, help="VM-local port")
    st_.add_argument("--no-wait", action="store_true")
    st_.add_argument("--require-gpu", action="store_true", help="fail model startup when CUDA is unavailable")
    st_.add_argument("--wait-timeout", type=float, default=900)
    st_.add_argument("--preset", metavar="JSON", help="local path to a model preset")
    st_.add_argument("--hf-repo")
    st_.add_argument("--quant")
    st_.add_argument("--model-dir", default=MODEL_DIR)
    st_.add_argument("--hf-token-file", help="absolute path to a 0600 token file on the VM")
    for name, option in OPTIONS.items():
        if option.public:
            st_.add_argument(option.flag, dest=name, type=option.type)
    st_.set_defaults(fn=start)
    ss.add_parser("stop").set_defaults(fn=stop)
    ss.add_parser("status").set_defaults(fn=status)
    lg = ss.add_parser("logs")
    lg.add_argument("-n", "--lines", type=int, default=50)
    px = ss.add_parser("proxy", help="serve the active model through a local HTTP endpoint")
    px.add_argument("--listen", default="127.0.0.1:8001")
    px.add_argument("--api-key", help="require this Bearer key (or $KGL_PROXY_KEY)")
    px.add_argument("--static", metavar="DIR", help="serve static files from DIR on loopback")
    lg.set_defaults(fn=logs)
    px.set_defaults(fn=proxy)

    model = sub.add_parser("model", help="prepare model files on the VM")
    model_commands = model.add_subparsers(dest="model_cmd", required=True)
    install = model_commands.add_parser("install", help="build a pinned backend on the VM")
    install.add_argument("backend", choices=INSTALLABLE_BACKENDS)
    install.add_argument("--revision", required=True, help="full llama.cpp Git commit SHA")
    install.add_argument("--directory", default=INSTALL_DIR)
    install.add_argument("--jobs", type=int, default=2)
    install.add_argument("--timeout", type=float, default=1800)
    plan = model_commands.add_parser("plan", help="compare a GGUF with available GPU memory")
    plan.add_argument("--model", required=True, help="GGUF path on the VM")
    plan.add_argument("--ctx", type=int, default=4096)
    plan.add_argument("--log", help="llama.cpp log path on the VM for measured buffer sizes")
    install.set_defaults(fn=install_model_backend)
    plan.set_defaults(fn=plan_model)
    fetch = model_commands.add_parser("fetch", help="resolve or download and verify a GGUF")
    fetch.add_argument("--preset", metavar="JSON", help="local path to a model preset")
    fetch.add_argument("--hf-repo")
    fetch.add_argument("--quant")
    fetch.add_argument("--model-dir", default=MODEL_DIR)
    fetch.add_argument("--hf-token-file", help="absolute path to a 0600 token file on the VM")
    fetch.add_argument("--timeout", type=float, default=1800)
    fetch.set_defaults(fn=fetch_model)

    fw = sub.add_parser("forward", help="forward HTTP to a notebook loopback port")
    fw.add_argument("--remote-port", type=int, required=True)
    fw.add_argument("--listen", default="127.0.0.1:8014")
    fw.set_defaults(fn=forward)

    i = sub.add_parser("infer", help="send one chat prompt to the model on the VM")
    i.add_argument("prompt", nargs="*", help="prompt (or stdin)")
    i.add_argument("--system")
    i.add_argument("--max-tokens", type=int)
    i.add_argument("--temperature", type=float)
    i.add_argument("--no-stream", action="store_true")
    i.add_argument("--json", action="store_true")
    i.add_argument("--timing", action="store_true")
    i.add_argument("--timeout", type=float)
    i.set_defaults(fn=cmd_infer)

    k = sub.add_parser("stop", help="stop model server, forget the local session")
    k.add_argument("--keep-server", action="store_true")
    k.add_argument("--kill-kernel", action="store_true", help="also DELETE the kernel (may break the notebook)")
    k.set_defaults(fn=cmd_stop)

    dr = sub.add_parser("doctor", help="run the PoC spikes and print a pass/fail report")
    dr.add_argument("--backend", choices=DOCTOR_BACKENDS, default="echo")
    dr.add_argument("--model", default=DEFAULT_MODEL)
    dr.add_argument("--port", type=int, default=8000)
    dr.add_argument("-n", type=int, default=20, help="number of inference requests")
    dr.add_argument("--keep", action="store_true", help="leave the server running")
    dr.add_argument("--wait-timeout", type=float, default=900)
    dr.add_argument("--report", help="write JSON report to this path")
    dr.set_defaults(fn=cmd_doctor)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    try:
        st = Settings.from_env()
        store = SessionStore(st.home)
        for name in ("timeout", "idle_timeout", "wait_timeout"):
            if getattr(args, name, None) is not None:
                seconds(getattr(args, name), "--" + name.replace("_", "-"), allow_zero=args.cmd == "exec")
        return args.fn(args, st, store)
    except KeyboardInterrupt:
        return 130
    except ExecutionError as e:
        eprint(f"remote error: {e}")
        return EXIT_REMOTE
    except TimeoutError as e:
        eprint(f"timeout: {e}")
        return EXIT_CONNECTION
    except (ConnectionLost, AuthError, KernelGone) as e:
        eprint(f"connection error: {e}")
        return EXIT_CONNECTION
    except NoSession as e:
        eprint(f"error: {e}")
        return EXIT_CONNECTION
    except KglError as e:
        eprint(f"error: {e}")
        return EXIT_ERROR
