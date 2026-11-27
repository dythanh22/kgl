"""Port ownership and reuse decisions of the daily Qwen proxy."""
from pathlib import Path
import importlib.util
import os
import socket
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from kgl.core.state import ServeRecord

folder = Path(__file__).resolve().parents[1] / "examples/qwen35b-llama"
spec = importlib.util.spec_from_file_location("qwen_proxy", folder / "proxy.py")
proxy = importlib.util.module_from_spec(spec)
with patch.object(sys, "path", [str(folder), *sys.path]):
    spec.loader.exec_module(proxy)


class QwenProxyTests(unittest.TestCase):
    worker = ServeRecord(port=8000, backend="llama", instance="target", model="/example-q4.gguf",
                         alias="qwen-local", quant="Q4_K_M")

    def ready(self, instance="target"):
        return 200, {"state": "ready", "health": {
            "model": self.worker.model, "instance": instance, "backend": "llama",
            "alias": self.worker.alias, "quant": self.worker.quant,
        }}

    def test_foreign_listener_remains_reachable(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            sock.listen()
            addr = sock.getsockname()
            with patch.object(proxy, "PORT", addr[1]), patch.object(proxy, "probe") as probe:
                with self.assertRaisesRegex(RuntimeError, "another application"):
                    proxy.manage(Path.cwd(), self.worker)
                probe.assert_not_called()
            with socket.create_connection(addr, 1):
                pass

    def test_busy_or_matching_instance_never_signalled(self):
        for result, busy in [((503, {"state": "busy"}), True),
                             (self.ready(), False)]:
            with self.subTest(busy=busy), patch.object(proxy, "listener", return_value=os.getpid()), \
                    patch.object(proxy, "check_owner"), patch.object(proxy, "probe", return_value=result), \
                    patch.object(proxy.signal, "pidfd_send_signal") as send:
                if busy:
                    with self.assertRaisesRegex(RuntimeError, "busy"):
                        proxy.manage(Path.cwd(), self.worker)
                else:
                    self.assertEqual(proxy.manage(Path.cwd(), self.worker), ("reused", os.getpid()))
                send.assert_not_called()

    def test_stale_proxy_on_same_remote_port_checks_new_target(self):
        with patch.object(proxy, "listener", return_value=os.getpid()), \
                patch.object(proxy, "check_owner"), patch.object(proxy, "probe", return_value=(
                    503, {"state": "wrong_instance", "health": {"model": self.worker.model, "instance": "target"}})), \
                patch.object(proxy, "target_ready", side_effect=RuntimeError("not ready")) as target_ready, \
                patch.object(proxy.signal, "pidfd_send_signal") as send:
            with self.assertRaisesRegex(RuntimeError, "not ready"):
                proxy.manage(Path.cwd(), self.worker)
            target_ready.assert_called_once_with(self.worker)
            send.assert_not_called()

    def test_unavailable_target_preserves_old_proxy(self):
        with patch.object(proxy, "listener", return_value=os.getpid()), \
                patch.object(proxy, "check_owner"), patch.object(proxy, "probe", return_value=self.ready("old")), \
                patch.object(proxy, "target_ready", side_effect=RuntimeError("not ready")), \
                patch.object(proxy.signal, "pidfd_send_signal") as send:
            with self.assertRaisesRegex(RuntimeError, "not ready"):
                proxy.manage(Path.cwd(), self.worker)
            send.assert_not_called()

    def test_cross_store_protects_ready_busy_and_unavailable_target(self):
        home = Path("/new")
        previous = Path("/old")
        for report, error in ((self.ready("old"), "still serving"),
                              ((503, {"state": "busy"}), "busy"),
                              ((503, {"state": "connection_lost"}), "not ready")):
            with self.subTest(state=report[1]["state"]), \
                    patch.object(proxy, "listener", return_value=os.getpid()), \
                    patch.object(proxy, "check_owner", return_value=previous), \
                    patch.object(proxy, "probe", return_value=report), \
                    patch.object(proxy, "target_ready", side_effect=RuntimeError("not ready")), \
                    patch.object(proxy.signal, "pidfd_send_signal") as send:
                with self.assertRaisesRegex(RuntimeError, error):
                    proxy.manage(home, self.worker, previous)
                send.assert_not_called()

    def test_cross_store_switch_and_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            previous = home / "previous"
            child = Mock(pid=789)
            child.poll.return_value = None
            with patch.object(proxy, "listener", side_effect=[os.getpid(), os.getpid(), child.pid, child.pid]), \
                    patch.object(proxy, "check_owner", side_effect=[previous, home]) as owner, \
                    patch.object(proxy, "probe", side_effect=[
                        (503, {"state": "connection_lost"}), (503, {"state": "connection_lost"}),
                        self.ready(), self.ready()]), \
                    patch.object(proxy, "target_ready") as ready, \
                    patch.object(proxy, "open_pidfd", return_value=42), \
                    patch.object(proxy.os, "close"), \
                    patch.object(proxy.signal, "pidfd_send_signal") as send, \
                    patch.object(proxy.select, "poll") as poll, \
                    patch.object(proxy.subprocess, "Popen", return_value=child):
                poll.return_value.poll.return_value = [(42, proxy.select.POLLIN)]
                self.assertEqual(proxy.manage(home, self.worker, previous), ("switched", child.pid))
                self.assertEqual(proxy.manage(home, self.worker, previous), ("reused", child.pid))
                owner.assert_called_with(child.pid, home, previous)
                ready.assert_called_once_with(self.worker)
                send.assert_called_once_with(42, proxy.signal.SIGTERM)


    def test_cross_store_rechecks_previous_proxy_after_target_ready(self):
        previous = Path("/old")
        with patch.object(proxy, "listener", return_value=os.getpid()), \
                patch.object(proxy, "check_owner", return_value=previous), \
                patch.object(proxy, "probe", side_effect=[
                    (503, {"state": "connection_lost"}), self.ready("old")]), \
                patch.object(proxy, "target_ready"), \
                patch.object(proxy.signal, "pidfd_send_signal") as send:
            with self.assertRaisesRegex(RuntimeError, "recovered"):
                proxy.manage(Path("/new"), self.worker, previous)
            send.assert_not_called()

    def test_stop_owned_verifies_birth_before_signalling(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(proxy, "listener", return_value=os.getpid()), \
                patch.object(proxy, "check_owner"), patch.object(proxy.signal, "pidfd_send_signal") as send:
            with self.assertRaisesRegex(RuntimeError, "identity"):
                proxy.stop_owned(Path(tmp), os.getpid(), "different")
            send.assert_not_called()
            with patch.object(proxy.select, "poll") as poll:
                poll.return_value.poll.return_value = [(1, proxy.select.POLLIN)]
                proxy.stop_owned(Path(tmp), os.getpid(), proxy.process_birth(os.getpid()))
            send.assert_called_once()

    def test_launcher_only_replaces_its_recorded_proxy(self):
        with patch.object(proxy, "listener", return_value=os.getpid()), patch.object(proxy, "check_owner"), \
                patch.object(proxy, "probe", return_value=self.ready("old")), \
                patch.object(proxy.signal, "pidfd_send_signal") as send:
            with self.assertRaisesRegex(RuntimeError, "owns port"):
                proxy.manage(Path.cwd(), self.worker, replace_only=(0, ""))
            send.assert_not_called()
