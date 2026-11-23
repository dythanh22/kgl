"""Check transport and inference from the CLI."""

import json
from pathlib import Path

from ...client.doctor import run_doctor, steps_to_dict
from ...core.config import Settings
from ...core.state import SessionStore
from ...backends import backend_spec
from ..common import EXIT_ERROR, EXIT_OK, open_runtime


def cmd_doctor(a, st: Settings, store: SessionStore) -> int:
    rt, s = open_runtime(st, store)

    def say(step) -> None:
        print(f"[{'PASS' if step.ok else 'FAIL'}] {step.name:<16} {step.seconds:>7.2f}s  {step.detail}")

    try:
        steps = run_doctor(rt, a.backend, a.model, a.port, a.n, a.keep, a.wait_timeout, say)
    finally:
        rt.close()
    if a.report:
        Path(a.report).write_text(json.dumps(steps_to_dict(steps), indent=2))
        print(f"report written to {a.report}")
    ok = bool(steps) and all(x.ok for x in steps)
    if ok and not backend_spec(a.backend).gpu_supported:
        print("\nTransport verified with the echo backend. Re-run with --backend transformers "
              "to verify a real model on the GPU.")
    return EXIT_OK if ok else EXIT_ERROR
