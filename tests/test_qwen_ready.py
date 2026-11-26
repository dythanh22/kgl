"""Exercise the Qwen example's notebook setup without downloading the build."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

DATASET_ID = "owner/test-build/versions/2"

SCRIPT = Path(__file__).resolve().parents[1] / "examples/qwen35b-llama/restore_worker.py"
spec = importlib.util.spec_from_file_location("restore_worker", SCRIPT)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class QwenReadyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model = self.root / "model.gguf"
        self.model.touch()
        self.package = self.root / "installed/f4f6f2e7db76e98c95cc9b7fc608ea669c806044"
        self.dataset = self.root / "dataset"
        self.source = self.dataset / self.package.name
        self.source.mkdir(parents=True)
        (self.source / "llama").write_text("#!/bin/sh\necho f4f6f2e\n")
        (self.source / "manifest.json").write_text(json.dumps({
            "revision": self.package.name, "cuda_arch": 75,
        }))
        self.hub = Mock()
        self.hub.dataset_download.return_value = str(self.dataset)
        hub_patch = patch.dict(sys.modules, kagglehub=self.hub)
        hub_patch.start()
        self.addCleanup(hub_patch.stop)
        gpu_patch = patch.object(worker.subprocess, "check_output", return_value="Tesla T4\nTesla T4\n")
        self.cards = gpu_patch.start()
        self.addCleanup(gpu_patch.stop)

    def prepare(self):
        return worker.prepare(self.model, self.package, self.root, DATASET_ID, "Q4_K_M")

    def test_restore_then_reuse_recorded_port(self):
        self.assertGreater(self.prepare(), 0)
        (self.root / ".kgl").mkdir()
        (self.root / ".kgl/serve.json").write_text('{"args":["--port","49409"]}')
        self.assertEqual(self.prepare(), 49409)
        self.hub.dataset_download.assert_called_once_with(DATASET_ID)

    def test_bad_manifest_leaves_destination_retryable(self):
        (self.source / "manifest.json").write_text('{"revision":"other","cuda_arch":75}')
        with self.assertRaisesRegex(ValueError, "Unexpected llama.cpp build"):
            self.prepare()
        self.assertFalse(self.package.exists())
        (self.source / "manifest.json").write_text(json.dumps({
            "revision": self.package.name, "cuda_arch": 75,
        }))
        self.assertGreater(self.prepare(), 0)

    def test_missing_input_or_gpu(self):
        self.cards.return_value = "Tesla T4\n"
        with self.assertRaisesRegex(RuntimeError, "T4 x2"):
            self.prepare()
        self.model.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Add Input"):
            self.prepare()
        self.hub.dataset_download.assert_not_called()
