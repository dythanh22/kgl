"""Offline packaging check: python -m tests.wheel_smoke (setuptools and wheel required)."""
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile

from .fake_jupyter import FakeJupyter


def main():
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        source = work / "source"
        shutil.copytree(root / "src/kgl", source / "src/kgl", ignore=shutil.ignore_patterns("__pycache__"))
        for name in ("README.md", "pyproject.toml"):
            shutil.copy(root / name, source)
        def pip(*args):
            subprocess.run([sys.executable, "-m", "pip", *args], cwd=work, check=True)
        pip("wheel", "--no-deps", "--no-build-isolation", "--no-index", str(source))
        target = work / "install"
        pip("install", "--no-index", "--no-deps", "--target", str(target), str(next(work.glob("*.whl"))))
        env = dict(os.environ, PYTHONPATH=str(target), KGL_HOME=str(work / "home"),
                   KGL_WORKDIR=str(work / "remote"))
        def cli(*args, data=""):
            return subprocess.run([str(target / "bin/kgl"), *args], input=data, text=True,
                                  stdout=subprocess.PIPE, cwd=work, env=env, timeout=45, check=True).stdout
        cli("--version")
        cli("--help")
        fake = FakeJupyter()
        try:
            cli("connect", "--proxy-url", "-", data=fake.url + "\n")
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            try:
                cli("serve", "start", "--backend", "echo", "--port", str(port))
                assert json.loads(cli("infer", "--json", "ping"))["text"] == "echo: ping"
            finally:
                cli("stop")
        finally:
            fake.stop()
        print("wheel smoke OK")


if __name__ == "__main__":
    main()
