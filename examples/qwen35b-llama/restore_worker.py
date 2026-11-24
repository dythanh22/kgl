"""Notebook-side preparation for ready.py."""
import json
from pathlib import Path
import shutil
import socket
import subprocess
from tempfile import TemporaryDirectory


def verify(package, revision):
    manifest = json.loads((package / "manifest.json").read_text())
    if manifest["revision"] != revision or manifest["cuda_arch"] != 75:
        raise ValueError(f"Unexpected llama.cpp build in {package}")
    result = subprocess.run([str(package / "llama"), "serve", "--version"],
                            text=True, capture_output=True)
    version = result.stdout + result.stderr
    if result.returncode or revision[:7] not in version:
        raise RuntimeError(f"Cannot verify llama.cpp in {package}: {version}")


def prepare(model, package, workdir):
    if not model.is_file():
        raise FileNotFoundError(f"Attach the Q4_K_M model using Kaggle Add Input: {model}")
    cards = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    ).splitlines()
    if len(cards) != 2 or any("T4" not in card for card in cards):
        raise RuntimeError(f"Select GPU T4 x2 in Kaggle. Found: {cards}")
    if package.exists():
        verify(package, package.name)
    else:
        import kagglehub

        dataset = kagglehub.dataset_download(
            "thanhduy1022/llama-cpp-f4f6f2e-sm75-214105c7/versions/1"
        )
        package.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=package.parent) as tmp:
            restored = Path(tmp) / package.name
            shutil.copytree(Path(dataset) / package.name, restored)
            binary = restored / "llama"
            binary.chmod(binary.stat().st_mode | 0o100)
            verify(restored, package.name)
            restored.rename(package)

    # Reproduce the recorded launch arguments so serve start can check the live
    # process identity and reject configuration conflicts before starting another.
    metadata = workdir / ".kgl" / "serve.json"
    if metadata.is_file():
        args = json.loads(metadata.read_text())["args"]
        config = {flag: args[args.index(flag) + 1] for flag in
                  ("--backend", "--model", "--ctx", "--tensor-split", "--port") if flag in args}
        print("Recorded worker configuration:", json.dumps(config))
        return int(args[args.index("--port") + 1])
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]
