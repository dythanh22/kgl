"""Optional localhost integration using the current interpreter's Jupyter installation."""
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest

from kgl.core.config import Settings
from kgl.transport.jupyter import JupyterClient
from kgl.remote import PROBE
from kgl.transport.runtime import Runtime
from kgl.client.serve_manager import ServeManager
from kgl.core.state import Session


@unittest.skipUnless(importlib.util.find_spec("jupyter_server") and importlib.util.find_spec("ipykernel"),
                     "requires jupyter-server and ipykernel")
class RealKernelTest(unittest.TestCase):
    def test_namespace_survives_probe_and_reconnect(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, JUPYTER_RUNTIME_DIR=tmp, IPYTHONDIR=tmp,
                       JUPYTER_CONFIG_DIR=tmp, JUPYTER_TOKEN="local-test")
            with open(Path(tmp) / "server.log", "w+") as log:
                process = subprocess.Popen(
                    [sys.executable, "-m", "jupyter_server", "--no-browser",
                     "--ServerApp.ip=127.0.0.1", "--ServerApp.port=18888",
                     "--ServerApp.allow_root=True", f"--ServerApp.root_dir={tmp}"],
                    cwd=tmp, env=env, stdout=log, stderr=log)
                try:
                    info_file = Path(tmp) / f"jpserver-{process.pid}.json"
                    deadline = time.monotonic() + 20
                    while not info_file.exists():
                        if process.poll() is not None or time.monotonic() >= deadline:
                            self.fail("local Jupyter server failed to start")
                        time.sleep(.1)
                    info = json.loads(info_file.read_text())
                    url = info["url"] + "?token=local-test"
                    client = JupyterClient(url)
                    kid = client.create_kernel("python3")["id"]
                    rt = Runtime(Session(url, kid), Settings(Path(tmp), workdir=tmp))
                    try:
                        rt.run('info = "sentinel"', check=True, total_timeout=30)
                        rt.call(PROBE, WORKDIR=tmp)
                        with socket.socket() as sock:
                            sock.bind(("127.0.0.1", 0))
                            port = sock.getsockname()[1]
                        mgr = ServeManager(rt, port)
                        first = mgr.start("echo", "echo", wait_timeout=10)
                        try:
                            reused = mgr.start("echo", "echo", wait_timeout=10)
                            self.assertEqual(first["health"]["instance"], reused["health"]["instance"])
                        finally:
                            mgr.stop()
                        rt.drop_channel()
                        self.assertEqual(rt.run("print(info)", total_timeout=10).text.strip(), "sentinel")
                    finally:
                        rt.close()
                        client.delete_kernel(kid)
                finally:
                    process.terminate()
                    process.wait(timeout=15)
