import os
import shutil
import tempfile
import unittest

from kgl.config import Settings
from kgl.runtime import Runtime
from kgl.state import Session

from .fake_jupyter import FakeJupyter


class FakeEnvTest(unittest.TestCase):
    """Starts a fake Jupyter server + temp 'remote workdir' per test."""

    token: str | None = None

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="kgl-test-")
        self.workdir = os.path.join(self.tmp, "working")
        os.makedirs(self.workdir)
        self.home = os.path.join(self.tmp, "home")
        self.server = FakeJupyter(token=self.token)
        self.settings = Settings(home=__import__("pathlib").Path(self.home), workdir=self.workdir,
                                 chunk_kb=1, ping_interval=5)
        kid = next(iter(self.server.kernels))
        self.session = Session(base_url=self.server.url, kernel_id=kid)
        self.rt = Runtime(self.session, self.settings)

    def tearDown(self):
        self.rt.close()
        self.server.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)
