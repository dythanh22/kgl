import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer


class AgentExampleTests(unittest.TestCase):
    def test_headless_success_and_failures(self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self.send_response(self.server.ready)
                self.end_headers()
                self.wfile.write(b'{"health":{"model":"echo"}}')

            def do_POST(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'data: {"choices":[{"delta":{"content":"ping"}}]}\n\n')
                if self.server.complete:
                    self.wfile.write(b'data: [DONE]\n\n')

        example = Path(__file__).resolve().parents[1] / "examples/agent_infer.py"
        with tempfile.TemporaryDirectory() as tmp, HTTPServer(("127.0.0.1", 0), Handler) as server:
            cli = Path(tmp) / "kgl"
            cli.write_text('#!/bin/sh\n[ "$STALE" = yes ] && exit 3\necho \'{"connection":"ok"}\'\n')
            cli.chmod(0o700)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            try:
                for stale, ready, complete, ok in [
                    ("no", 200, True, True), ("yes", 200, True, False),
                    ("no", 503, True, False), ("no", 200, False, False)
                ]:
                    server.ready, server.complete = ready, complete
                    p = subprocess.run([sys.executable, str(example), "--proxy",
                                        f"http://127.0.0.1:{server.server_port}", "ping"],
                                       input="", text=True, capture_output=True, timeout=10,
                                       env=dict(os.environ, PATH=tmp + ":" + os.environ["PATH"], STALE=stale))
                    if ok:
                        self.assertEqual(p.returncode, 0, p.stderr)
                        self.assertEqual(json.loads(p.stdout)["text"], "ping")
                    else:
                        self.assertEqual(p.returncode, 3)
                        self.assertFalse(p.stdout)
                        self.assertTrue(p.stderr)
            finally:
                server.shutdown()
