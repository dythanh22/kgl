"""Start, stop, inspect, and proxy a model server."""

import json
import os

from ...backends import DEFAULT_BACKEND, DEFAULT_GGUF_BACKEND, OPTIONS, backend_spec
from ...client.llama import prepare_llama
from ...client.serve_manager import ServeManager
from ...core.state import ServeRecord
from ...core.errors import KglError, ServeError
from ...proxy.handler import make_proxy_server
from ..common import EXIT_OK, with_runtime, eprint
from .models import provision_output


@with_runtime
def start(a, rt, s, store):
    mgr = ServeManager(rt, a.port)
    backend = a.backend or (DEFAULT_GGUF_BACKEND if a.preset or a.hf_repo else DEFAULT_BACKEND)
    spec = backend_spec(backend)
    options = {name: getattr(a, name) for name, opt in OPTIONS.items()
               if opt.public and getattr(a, name) is not None}
    model = a.model or spec.default_model
    try:
        spec.validate_options(options, a.require_gpu)
    except ValueError as exc:
        raise ServeError(str(exc)) from exc
    if spec.passthrough:
        model, resolved = prepare_llama(rt, a, s.llama_manifest, provision_output)
        options.update(resolved)
    elif any(value is not None for value in (a.preset, a.hf_repo, a.quant, a.hf_token_file)):
        raise ServeError("GGUF options require the llama backend")

    def record_start(result):
        s.serve = ServeRecord(port=a.port, model=spec.model_id(model), backend=backend,
                              instance=result["instance"], state="starting", **options.get("settings", {}))
        store.save(s, expected=s)

    h = mgr.start(model, backend, a.dtype, wait=not a.no_wait,
                  wait_timeout=a.wait_timeout, progress=eprint, require_gpu=a.require_gpu,
                  options=options, on_start=record_start)
    if not a.no_wait:
        health = h["health"]
        s.serve.state = "ready"
        for key in ("capabilities", "defaults", "alias", "quant"):
            if key in health:
                setattr(s.serve, key, health[key])
        store.save(s, expected=s)
    print("model server ready" if not a.no_wait else "model server launching")
    return EXIT_OK


@with_runtime
def logs(a, rt, s, store):
    print("\n".join(ServeManager(rt).logs(a.lines)))
    return EXIT_OK


@with_runtime
def status(a, rt, s, store):
    if not s.serve:
        raise KglError("no model server recorded; run `kgl serve start` first")
    mgr = ServeManager(rt, s.serve.port)
    kernel = rt.client.get_kernel(s.kernel_id)
    busy = kernel.get("execution_state") == "busy"
    if busy and (rt.settings.control_kernel_id or s.control_kernel_id) in (None, s.kernel_id):
        result = {"state": "busy", "instance": s.serve.instance}
    else:
        result = mgr.health()
        expected = s.serve.instance
        if expected and result.get("health") and result["health"].get("instance") != expected:
            result["state"] = "wrong_instance"
    print(json.dumps(result, indent=2))
    return EXIT_OK


@with_runtime
def stop(a, rt, s, store):
    if not s.serve:
        print(json.dumps(ServeManager(rt).stop()))
        return EXIT_OK
    mgr = ServeManager(rt, s.serve.port)
    print(json.dumps(mgr.stop(s.serve.instance)))
    s.serve = None
    store.save(s, expected=s)
    return EXIT_OK


@with_runtime
def proxy(a, rt, s, store):
    if not s.serve:
        raise KglError("no model server recorded; run `kgl serve start` first")
    if not s.serve.instance:
        raise ServeError("server instance is missing; run kgl serve stop and kgl serve start")
    host, _, port = a.listen.rpartition(":")
    key = a.api_key or os.environ.get("KGL_PROXY_KEY")
    srv = make_proxy_server(rt, s.serve.port, s.serve.model, host or "127.0.0.1",
                            int(port), key, static=a.static)
    url = f"http://{host or '127.0.0.1'}:{port}"
    endpoint = "/handle" if backend_spec(s.serve.backend).custom else "/v1"
    eprint(f"Model endpoint: {url}{endpoint} "
           f"(model={s.serve.model}, auth={'on' if key else 'off'}); Ctrl-C to stop.\n"
           "Keep this terminal open and send requests from a second terminal.")
    if a.static:
        eprint(f"Open the demo at {url}/")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return EXIT_OK
