"""Bring the documented Qwen worker to ready on the connected Kaggle VM."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from kgl.client.model_fetch import INSTALL_DIR, read_preset
from kgl.core.config import Settings
from kgl.core.errors import KglError

HERE = Path(__file__).resolve().parent
PRESET = HERE / "qwen3.8-35b-a3b.json"


class QwenError(KglError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def main():
    preset = read_preset(PRESET)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ctx", type=int, default=preset["context"], help="worker context window in tokens (default: %(default)s)")
    parser.add_argument("--quant", choices=preset["inputs"], default=preset["default_quant"],
                        help="attached GGUF quant (default: %(default)s)")
    args = parser.parse_args()
    if args.ctx < 2:
        parser.error("--ctx must be at least 2")
    health, port = prepare_qwen(preset, args.quant, args.ctx)
    print(f"Qwen ready: instance={health['instance']} pid={health['pid']} port={port} ctx={args.ctx}")


def prepare_qwen(preset, quant, ctx, *, phase=lambda name: None, start=None, quiet=False, settings=None):
    model = preset["inputs"][quant]
    build = preset["build"]
    package = f"{INSTALL_DIR}/{build['revision']}"
    kgl = [sys.executable, "-m", "kgl"]
    # Run preparation inside the notebook, including access to its private dataset.
    source = (HERE / "restore_worker.py").read_text()
    st = settings or Settings.from_env()
    workdir = st.workdir
    env = {**os.environ, "KGL_HOME": str(st.home), "KGL_WORKDIR": workdir}
    stderr = subprocess.PIPE if quiet else None
    phase("input")
    check = f"""
try:
    check_input(Path({model!r}), {quant!r})
except FileNotFoundError as exc:
    print("missing_input" if not Path({model!r}).is_file() else "gpu_unavailable")
except RuntimeError:
    print("wrong_gpu")
else:
    print("ok")
"""
    result = subprocess.check_output([*kgl, "exec", "--timeout", "30", source + check],
                                     text=True, stderr=stderr, env=env).strip()
    if result != "ok":
        messages = {"missing_input": f"Attach the {quant} GGUF from the preset.",
                    "wrong_gpu": "Select two Tesla T4 GPUs on the notebook.",
                    "gpu_unavailable": "Check nvidia-smi in the notebook."}
        raise QwenError(result if result in messages else "input_probe",
                        messages.get(result, "Could not verify the notebook input."))
    phase("build")
    restore_call = f"""
try:
    print("ok:" + str(restore(Path({package!r}), Path({workdir!r}), {build['dataset']!r})))
except (ValueError, RuntimeError):
    print("err:build_verify")
except Exception:
    print("err:build_access")
"""
    output = subprocess.check_output(
        [*kgl, "exec", "--timeout", "180", source + restore_call], text=True, stderr=stderr, env=env
    )
    details, _, last_line = output.strip().rpartition("\n")
    if last_line == "err:build_verify":
        raise QwenError("build_verify", "Saved llama.cpp build failed verification on this VM. See BUILD.md.")
    if not last_line.startswith("ok:"):
        raise QwenError("build_access", "Could not restore the llama.cpp build dataset. Check dataset access and Internet.")
    if details and not quiet:
        print(details, flush=True)
    port = str(int(last_line[3:]))
    phase("worker")
    arguments = [
        "serve", "start", "--backend", "llama", "--port", port,
        "--model", model, "--preset", str(PRESET),
        "--llama-manifest", package + "/manifest.json", "--ctx", str(ctx),
        "--tensor-split", "1,1", "--require-gpu", "--wait-timeout", "360",
    ]
    if start:
        start(arguments)
    else:
        subprocess.run([*kgl, *arguments], env=env, check=True)
    phase("health")
    status = json.loads(subprocess.check_output([*kgl, "serve", "status"], text=True, stderr=stderr, env=env))
    health = status.get("health", {})
    if (status["state"] != "ready" or health.get("model") != model
            or health.get("alias") != preset["alias"] or health.get("quant") != quant
            or health.get("backend") != "llama" or health.get("n_ctx") != ctx
            or set(health.get("devices", [])) != {"cuda:0", "cuda:1"}):
        raise KglError(f"Qwen readiness check failed: {status['state']}, "
                         f"context={health.get('n_ctx')}, devices={health.get('devices')}")
    return health, int(port)


if __name__ == "__main__":
    try:
        main()
    except QwenError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(4)
    except KglError as exc:
        sys.exit(str(exc))
    except subprocess.CalledProcessError as exc:
        if exc.returncode == 3:
            print("Check the notebook session and reconnect with kgl connect.", file=sys.stderr)
        raise SystemExit(exc.returncode)
