import contextlib
import io
import json
from pathlib import Path
import os
import sys
import unittest
from unittest import mock

from kgl.cli import main
from kgl.core.errors import ConnectionLost
from kgl.transport.jupyter import ExecResult
from kgl.core.state import ServeRecord, SessionStore

from .support import FakeEnvTest


class CliTests(FakeEnvTest):
    def setUp(self):
        super().setUp()
        env = {"KGL_HOME": self.home, "KGL_WORKDIR": self.workdir, "KGL_CHUNK_KB": "1"}
        p = mock.patch.dict(os.environ, env)
        p.start()
        self.addCleanup(p.stop)

    def kgl(self, *argv, stdin=None):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), \
                mock.patch.object(sys, "stdin", io.StringIO(stdin or "")):
            rc = main(list(argv))
        return rc, out.getvalue(), err.getvalue()

    def test_connect_prompt_and_headless(self):
        with mock.patch.dict(os.environ, {"KGL_PROXY_URL": ""}), mock.patch("getpass.getpass") as prompt:
            rc, _, error = self.kgl("connect")
            self.assertNotEqual(rc, 0)
            self.assertIn("--proxy-url", error)
            prompt.assert_not_called()
            prompt.return_value = self.server.url + "?token=hidden-test-token"
            with mock.patch("sys.stdin", new=mock.Mock(isatty=lambda: True)):
                out = io.StringIO()
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                    self.assertEqual(main(["connect"]), 0)
                self.assertNotIn("hidden-test-token", out.getvalue())
                prompt.assert_called_once()

    def test_full_flow(self):
        rc, out, err = self.kgl("status")
        self.assertEqual(rc, 3)  # no session
        self.assertIn("no active session", err)

        rc, out, _ = self.kgl("connect", "--proxy-url", "-", stdin=self.server.url + "\n")
        self.assertEqual(rc, 0, out)
        self.assertIn("connected", out)

        rc, out, _ = self.kgl("exec", "print(6*7)")
        self.assertEqual((rc, out.strip()), (0, "42"))
        bad = os.path.join(self.tmp, "bad.py")
        with open(bad, "w") as fh:
            fh.write("raise RuntimeError('x')")
        rc, out, err = self.kgl("exec", "-f", bad)
        self.assertEqual(rc, 4)
        self.assertIn("RuntimeError", err)
        rc, out, _ = self.kgl("exec", stdin="print('from stdin')")
        self.assertEqual((rc, out.strip()), (0, "from stdin"))
        rc, out, _ = self.kgl("exec", "--shell", "echo shell-ok; exit 7")
        self.assertEqual(rc, 7)
        self.assertEqual(out.strip(), "shell-ok")  # marker line is hidden
        self.assertEqual(self.kgl("exec", "--shell", "exit 127")[0], 127)
        self.assertEqual(self.kgl("exec", "--shell", "kill -TERM $$")[0], 143)
        self.assertEqual(self.kgl("exec", "print('@@KGL@@{}')")[:2], (0, "@@KGL@@{}\n"))
        with mock.patch("kgl.transport.runtime.Runtime.run", return_value=ExecResult(status="aborted")):
            rc, _, err = self.kgl("exec", "print('skipped')")
            self.assertEqual(rc, 4)
            self.assertIn("aborted", err)

        f = os.path.join(self.tmp, "a.txt")
        Path(f).write_text("hello" * 500)
        self.assertEqual(self.kgl("upload", f, "a.txt")[0], 0)
        back = os.path.join(self.tmp, "b.txt")
        self.assertEqual(self.kgl("download", "a.txt", back)[0], 0)
        self.assertEqual(Path(back).read_text(), "hello" * 500)

        port = str(19000 + os.getpid() % 500)
        rc, out, err = self.kgl("serve", "start", "--backend", "echo", "--port", port,
                                "--wait-timeout", "30")
        self.assertEqual(rc, 0, err)
        rc, out, _ = self.kgl("infer", "hi", "there", "--no-stream")
        self.assertEqual((rc, out.strip()), (0, "echo: hi there"))
        rc, out, _ = self.kgl("infer", "stream", "me")
        self.assertEqual((rc, out.strip()), (0, "echo: stream me"))
        rc, out, _ = self.kgl("infer", "j", "--json")
        self.assertEqual(json.loads(out)["text"], "echo: j")

        rc, out, _ = self.kgl("status", "--json")
        data = json.loads(out)
        self.assertEqual(data["connection"], "ok")
        self.assertEqual(data["serve"]["state"], "ready")
        self.assertNotIn("token", out)

        self.assertEqual(self.kgl("stop")[0], 0)
        self.assertEqual(self.kgl("status")[0], 3)

    def test_doctor_with_echo_backend(self):
        self.kgl("connect", "--proxy-url", self.server.url)
        port = str(19600 + os.getpid() % 300)
        rc, out, err = self.kgl("doctor", "-n", "5", "--port", port, "--wait-timeout", "30")
        # step 3 (GPU) legitimately FAILs on this machine; everything else must pass.
        lines = [l for l in out.splitlines() if l.startswith("[")]
        failed = [l for l in lines if l.startswith("[FAIL]")]
        self.assertTrue(all("gpu_visible" in line for line in failed), out)
        self.assertTrue(any("infer_x5" in l and l.startswith("[PASS]") for l in lines), out)
        self.assertTrue(any("reconnect" in l and l.startswith("[PASS]") for l in lines), out)

    def test_doctor_reuses_recorded_worker_with_custom_options(self):
        self.kgl("connect", "--proxy-url", self.server.url)
        port = str(19900 + os.getpid() % 300)
        rc, _, err = self.kgl("serve", "start", "--backend", "echo", "--dtype", "float32",
                              "--port", port, "--wait-timeout", "10")
        self.assertEqual(rc, 0, err)
        try:
            _, out, _ = self.kgl("doctor", "-n", "1", "--port", port, "--wait-timeout", "10")
            self.assertRegex(out, r"\[PASS\].*infer_x1")
            self.assertNotIn("serve_stop", out)
            self.assertEqual(self.kgl("infer", "after doctor")[:2], (0, "echo: after doctor\n"))

            rc, out, _ = self.kgl("doctor", "--backend", "transformers", "-n", "1",
                                  "--port", port, "--wait-timeout", "10")
            self.assertEqual(rc, 1)
            self.assertRegex(out, r"\[FAIL\].*serve_start")
            self.assertNotIn("infer_x1", out)
        finally:
            self.kgl("serve", "stop")

    def test_stop_preserves_session_when_kernel_delete_outcome_is_unknown(self):
        self.kgl("connect", "--proxy-url", self.server.url)
        store = SessionStore(Path(self.home))
        session = store.load()
        session.serve = ServeRecord(port=8000, instance="test-worker", backend="echo", model="echo")
        store.save(session)
        with mock.patch("kgl.cli.commands.session.ServeManager.stop", return_value={"stopped": True}) as stop, \
                mock.patch("kgl.transport.jupyter.JupyterClient.delete_kernel",
                           side_effect=ConnectionLost("delete response lost")):
            rc, _, err = self.kgl("stop", "--kill-kernel")
        self.assertEqual(rc, 3)
        self.assertIn("delete response lost", err)
        stop.assert_called_once_with("test-worker")
        self.assertEqual(store.load().kernel_id, session.kernel_id)
        self.assertIsNone(store.load().serve)

        # Simulate a DELETE that reached Jupyter before its response was lost.
        self.server.kernels.pop(session.kernel_id)
        self.assertEqual(self.kgl("stop", "--kill-kernel")[0], 0)
        self.assertIsNone(store.load())

    def test_auth_never_prints_secrets(self):
        with mock.patch.dict(os.environ, {"KAGGLE_API_TOKEN": "SUPERSECRET"}):
            rc, out, _ = self.kgl("auth")
        self.assertEqual(rc, 0)
        self.assertNotIn("SUPERSECRET", out)
        self.assertIn("VS Code Compatible URL", out)


if __name__ == "__main__":
    unittest.main()
