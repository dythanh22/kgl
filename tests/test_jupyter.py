import time
import unittest

from kgl.core.errors import AuthError, ConnectionLost, ExecutionError, KernelGone
from kgl.transport.jupyter import JupyterClient
from kgl.transport.runtime import Runtime
from kgl.core.state import Session

from .support import FakeEnvTest


class JupyterTests(FakeEnvTest):
    def test_rest_and_execute_stream(self):
        self.assertEqual(len(self.rt.client.list_kernels()), 1)
        chunks = []
        res = self.rt.run("import time\nprint('a', flush=True)\ntime.sleep(0.2)\nprint('b')",
                          lambda n, t: chunks.append(t))
        self.assertEqual(res.status, "ok")
        self.assertEqual(res.text.split(), ["a", "b"])
        self.assertGreaterEqual(len(chunks), 1)

    def test_remote_error(self):
        res = self.rt.run("raise ValueError('boom')")
        self.assertEqual(res.status, "error")
        self.assertEqual(res.error["ename"], "ValueError")
        with self.assertRaises(ExecutionError):
            self.rt.run("raise ValueError('boom')", check=True)

    def test_idle_timeout(self):
        with self.assertRaises(TimeoutError):
            self.rt.run("import time; time.sleep(3)", idle_timeout=0.5)

    def test_unsent_failure_retries_once_but_sent_failure_does_not(self):
        self.rt.run("print(1)")  # open channel
        self.server.fail_next_execute = True
        with self.assertRaises(ConnectionLost) as cm:
            self.rt.run("print(2)")
        self.assertTrue(cm.exception.sent)
        self.assertEqual(self.server.executed.count("print(2)"), 1)  # not re-run
        # idempotent ops may retry transparently
        self.server.fail_next_execute = True
        res = self.rt.run("print(3)", idempotent=True)
        self.assertEqual(res.text.strip(), "3")

    def test_kernel_gone(self):
        self.server.kernels.clear()
        with self.assertRaises(KernelGone):
            self.rt.run("print(1)")

    def test_interrupt_endpoint(self):
        self.rt.client.interrupt_kernel(self.session.kernel_id)
        self.assertEqual(self.server.interrupts, 1)


class TokenTests(FakeEnvTest):
    token = "s3cret"

    def test_token_in_url_used_for_rest_and_ws(self):
        self.assertEqual(self.rt.run("print('ok')").text.strip(), "ok")

    def test_bad_token_is_auth_error(self):
        bad = JupyterClient(self.server.url.replace("s3cret", "nope"))
        with self.assertRaises(AuthError):
            bad.list_kernels()


if __name__ == "__main__":
    unittest.main()
