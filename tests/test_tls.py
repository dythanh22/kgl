"""wss:// / https:// path with a throw-away self-signed certificate."""
import os
import shutil
import ssl
import subprocess
import tempfile
import unittest
from unittest import mock

from kgl.config import Settings
from kgl.runtime import Runtime
from kgl.state import Session

from .fake_jupyter import FakeJupyter


@unittest.skipUnless(shutil.which("openssl"), "openssl not available")
class TlsTests(unittest.TestCase):
    def test_https_and_wss(self):
        tmp = tempfile.mkdtemp()
        crt, key = os.path.join(tmp, "c.pem"), os.path.join(tmp, "k.pem")
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", key,
                        "-out", crt, "-days", "1", "-subj", "/CN=localhost",
                        "-addext", "subjectAltName=DNS:localhost,IP:127.0.0.1"],
                       check=True, capture_output=True)
        srv = FakeJupyter()
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(crt, key)
        srv.httpd.socket = ctx.wrap_socket(srv.httpd.socket, server_side=True)
        try:
            with mock.patch.dict(os.environ, {"KGL_CA_FILE": crt}):
                kid = next(iter(srv.kernels))
                rt = Runtime(Session(base_url=f"https://localhost:{srv.port}", kernel_id=kid),
                    Settings(home=tmp, workdir=tmp, ping_interval=5))
                self.assertEqual(rt.run("print('tls ok')").text.strip(), "tls ok")
                rt.close()
        finally:
            srv.stop()


if __name__ == "__main__":
    unittest.main()
