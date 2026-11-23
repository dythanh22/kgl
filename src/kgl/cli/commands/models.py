"""Build llama.cpp, plan memory, and fetch model files."""

import json
import sys

from ...client.model_fetch import fetch_model, read_preset, vm_operation
from ...core.errors import ServeError
from ...core.state import SessionStore
from ..common import EXIT_OK, with_runtime


def provision_output(name: str, text: str) -> None:
    sys.stderr.write(text)
    sys.stderr.flush()


@with_runtime
def fetch(a, rt, session, store):
    preset = read_preset(a.preset) if a.preset else None
    result = fetch_model(rt, preset, a.hf_repo, a.quant, a.model_dir, a.hf_token_file,
                         a.timeout, provision_output)
    print(json.dumps(result))
    return EXIT_OK


@with_runtime
def install_model_backend(a, rt, s, store: SessionStore) -> int:
    result = vm_operation(rt, "install", {"revision": a.revision, "directory": a.directory,
                                          "jobs": a.jobs}, a.timeout, provision_output)
    s.llama_manifest = result["manifest"]
    store.save(s, expected=s)
    print(json.dumps(result))
    return EXIT_OK


@with_runtime
def plan_model(a, rt, session, store) -> int:
    if a.ctx <= 0:
        raise ServeError("--ctx must be positive")
    result = vm_operation(rt, "plan", {"model": a.model, "ctx": a.ctx, "log": a.log},
                          30, provision_output)
    print(json.dumps(result, indent=2))
    return EXIT_OK
