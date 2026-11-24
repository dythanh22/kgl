"""Port ownership and reuse decisions of the daily Qwen proxy."""
from pathlib import Path
import importlib.util
import os
import socket
import sys
import unittest
from unittest.mock import patch

folder = Path(__file__).resolve().parents[1] / "examples/qwen35b-llama"
spec = importlib.util.spec_from_file_location("qwen_proxy", folder / "proxy.py")
proxy = importlib.util.module_from_spec(spec)
with patch.object(sys, "path", [str(folder), *sys.path]):
    spec.loader.exec_module(proxy)


class QwenProxyTests(unittest.TestCase):
    def test_foreign_listener_remains_reachable(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            sock.listen()
            addr = sock.getsockname()
            with patch.object(proxy, "PORT", addr[1]), patch.object(proxy, "probe") as probe:
                with self.assertRaisesRegex(RuntimeError, "another application"):
                    proxy.manage(Path.cwd(), "target")
                probe.assert_not_called()
            with socket.create_connection(addr, 1):
                pass

    def test_busy_or_matching_instance_never_signalled(self):
        for result, busy in [((503, {"state": "busy"}), True),
                             ((200, {"state": "ready", "health": {
                                 "model": proxy.MODEL, "instance": "target"}}), False)]:
            with self.subTest(busy=busy), patch.object(proxy, "listener", return_value=os.getpid()), \
                    patch.object(proxy, "check_owner"), patch.object(proxy, "probe", return_value=result), \
                    patch.object(proxy.signal, "pidfd_send_signal") as send:
                if busy:
                    with self.assertRaisesRegex(RuntimeError, "busy"):
                        proxy.manage(Path.cwd(), "target")
                else:
                    self.assertEqual(proxy.manage(Path.cwd(), "target"), ("reused", os.getpid()))
                send.assert_not_called()

    def test_stale_proxy_on_same_remote_port_checks_new_target(self):
        with patch.object(proxy, "listener", return_value=os.getpid()), \
                patch.object(proxy, "check_owner"), patch.object(proxy, "probe", return_value=(
                    503, {"state": "wrong_instance", "health": {"model": proxy.MODEL, "instance": "target"}})), \
                patch.object(proxy, "target_ready", side_effect=RuntimeError("not ready")) as target_ready, \
                patch.object(proxy.signal, "pidfd_send_signal") as send:
            with self.assertRaisesRegex(RuntimeError, "not ready"):
                proxy.manage(Path.cwd(), "target")
            target_ready.assert_called_once_with("target")
            send.assert_not_called()

    def test_unavailable_target_preserves_old_proxy(self):
        with patch.object(proxy, "listener", return_value=os.getpid()), \
                patch.object(proxy, "check_owner"), patch.object(proxy, "probe", return_value=(
                    200, {"state": "ready", "health": {"model": proxy.MODEL, "instance": "old"}})), \
                patch.object(proxy, "target_ready", side_effect=RuntimeError("not ready")), \
                patch.object(proxy.signal, "pidfd_send_signal") as send:
            with self.assertRaisesRegex(RuntimeError, "not ready"):
                proxy.manage(Path.cwd(), "target")
            send.assert_not_called()
