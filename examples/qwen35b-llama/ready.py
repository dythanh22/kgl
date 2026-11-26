"""Bring the documented Qwen worker to ready on the connected Kaggle VM."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from kgl.client.model_fetch import INSTALL_DIR, read_preset
from kgl.core.config import Settings
from kgl.core.errors import KglError

HERE = Path(__file__).resolve().parent
PRESET = HERE / "qwen3.8-35b-a3b.json"


def main():
    preset = read_preset(PRESET)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ctx", type=int, default=preset["context"], help="worker context window in tokens (default: %(default)s)")
    parser.add_argument("--quant", choices=preset["inputs"], default=preset["default_quant"],
                        help="attached GGUF quant (default: %(default)s)")
    args = parser.parse_args()
    if args.ctx < 2:
        parser.error("--ctx must be at least 2")
    model = preset["inputs"][args.quant]
    build = preset["build"]
    package = f"{INSTALL_DIR}/{build['revision']}"
    kgl = [sys.executable, "-m", "kgl"]
    # Run preparation inside the notebook, including access to its private dataset.
    source = (HERE / "restore_worker.py").read_text()
    workdir = Settings.from_env().workdir
    call = (f"print(prepare(Path({model!r}), Path({package!r}), Path({workdir!r}), "
            f"{build['dataset']!r}, {args.quant!r}))")
    output = subprocess.check_output(
        [*kgl, "exec", "--timeout", "180", source + "\n" + call], text=True
    )
    details, _, port_line = output.strip().rpartition("\n")
    if details:
        print(details, flush=True)
    port = str(int(port_line))
    subprocess.run([
        *kgl, "serve", "start", "--backend", "llama", "--port", port,
        "--model", model, "--preset", str(PRESET),
        "--llama-manifest", package + "/manifest.json", "--ctx", str(args.ctx),
        "--tensor-split", "1,1", "--require-gpu", "--wait-timeout", "360",
    ], check=True)
    status = json.loads(subprocess.check_output([*kgl, "serve", "status"], text=True))
    health = status.get("health", {})
    if (status["state"] != "ready" or health.get("model") != model
            or health.get("alias") != preset["alias"] or health.get("quant") != args.quant
            or health.get("backend") != "llama" or health.get("n_ctx") != args.ctx
            or set(health.get("devices", [])) != {"cuda:0", "cuda:1"}):
        raise SystemExit(f"Qwen readiness check failed: {status['state']}, "
                         f"context={health.get('n_ctx')}, devices={health.get('devices')}")
    print(f"Qwen ready: instance={health['instance']} pid={health['pid']} port={port} ctx={args.ctx}")


if __name__ == "__main__":
    try:
        main()
    except KglError as exc:
        sys.exit(str(exc))
    except subprocess.CalledProcessError as exc:
        if exc.returncode == 3:
            print("Check the notebook session and reconnect with kgl connect.", file=sys.stderr)
        raise SystemExit(exc.returncode)
