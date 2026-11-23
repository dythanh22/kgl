"""HTTP forwarding through fake Jupyter, including chunked binary transfers."""
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

from kgl.proxy.forward import make_forward_server
from .support import FakeEnvTest


class ForwardTests(FakeEnvTest):
    def setUp(self):
        super().setUp()

        class Service(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                if self.path != "/echo?q=1":
                    return self.send_error(404)
                data = self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(201)
                self.send_header("X-Request-ID", self.headers["X-Request-ID"])
                self.end_headers()
                self.wfile.write(data)

        upstream = ThreadingHTTPServer(("127.0.0.1", 0), Service)
        self.proxy = make_forward_server(self.rt, upstream.server_port, "127.0.0.1", 0)
        for server in (upstream, self.proxy):
            threading.Thread(target=server.serve_forever, daemon=True).start()
            self.addCleanup(server.server_close)
            self.addCleanup(server.shutdown)

    def post(self, body=b"x", path="/echo?q=1", **headers):
        conn = HTTPConnection("127.0.0.1", self.proxy.server_port, timeout=15)
        try:
            conn.request("POST", path, body, {"X-Request-ID": "example", **headers})
            reply = conn.getresponse()
            return reply.status, reply.getheader("X-Request-ID"), reply.read()
        finally:
            conn.close()

    def test_binary_roundtrip_and_upstream_status(self):
        for body in (b"", b"hello", bytes(range(256)) * 9):
            self.assertEqual(self.post(body), (201, "example", body))
        self.assertEqual(self.post(path="/missing")[0], 404)

    def test_rejects_bad_headers_and_size_before_execution(self):
        count = len(self.server.executed)
        self.assertEqual(self.post(Host="bad.invalid")[0], 400)
        self.assertEqual(self.post(Origin="http://bad.invalid")[0], 403)
        self.assertEqual(self.post(**{"Content-Length": str(self.settings.request_bytes + 1)})[0], 413)
        self.assertEqual(len(self.server.executed), count)

    def test_connection_loss_returns_error_and_next_request_succeeds(self):
        self.server.fail_next_execute = True
        self.assertEqual(self.post()[0], 503)
        self.assertEqual(self.post()[0], 201)
