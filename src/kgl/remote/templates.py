"""Load VM snippets as source without importing or running them locally."""

from importlib.resources import files
import json
from textwrap import indent

from ..protocol import INSTANCE_HEADER, REQUEST_ID_HEADER, TIMEOUT_HEADER


def _read(name: str) -> str:
    return "\n" + files(__package__).joinpath("snippets").joinpath(name + ".py").read_text(encoding="utf-8")


def _read_http(name: str) -> str:
    return (_read(name).replace("__INSTANCE_HEADER__", json.dumps(INSTANCE_HEADER))
            .replace("__REQUEST_ID_HEADER__", json.dumps(REQUEST_ID_HEADER))
            .replace("__TIMEOUT_HEADER__", json.dumps(TIMEOUT_HEADER)))


def _locked_process_operation(operation: str) -> str:
    return (_read("process") + 'with (d / "serve.lock").open("a") as lock:\n'
            + '    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)\n'
            + indent(operation, "    "))


_CONTAINER_UPTIME = _read("container_uptime")
PROBE = _CONTAINER_UPTIME + _read("probe")
UPLOAD_CHUNK = _read("upload_chunk")
COMMIT_UPLOAD = _read("commit_upload")
DISCARD_UPLOAD = _read("discard_upload")
FILE_SIZE = _read("file_size")
STAT = _read("stat")
DOWNLOAD_CHUNK = _read("download_chunk")
SHELL = _read("shell")
SERVE_START = _locked_process_operation(_read("serve_start"))
SERVE_HEALTH = _CONTAINER_UPTIME + _read("serve_health")
SERVE_STOP = _locked_process_operation(_read("serve_stop"))
SERVE_LOGS = _read("serve_logs")
INFER = _read_http("infer")
CANCEL_INFER = _read_http("cancel_infer")
FORWARD = _read("forward")
