"""Expose a notebook HTTP service on local loopback."""
from ...core.errors import KglError
from ...proxy.forward import make_forward_server
from ..common import EXIT_OK, eprint, with_runtime


@with_runtime
def forward(a, rt, session, store):
    host, sep, port = a.listen.rpartition(":")
    if not sep or not host or not port.isdigit() or not 0 < int(port) <= 65535:
        raise KglError("--listen requires HOST:PORT")
    with make_forward_server(rt, a.remote_port, host, int(port)) as server:
        eprint(f"Forwarding http://{host}:{server.server_port} to notebook port {a.remote_port}")
        server.serve_forever()
    return EXIT_OK
