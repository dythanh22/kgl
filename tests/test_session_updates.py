import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from kgl.errors import KglError
from kgl.state import Session, SessionStore


class SessionUpdateTests(unittest.TestCase):
    def test_process_reconnect_beats_stale_save_and_clear(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))
            store.path.write_text(json.dumps({"base_url": "http://old", "kernel_id": "a",
                                              "connected_at": 1}))
            old = store.load()
            subprocess.run([sys.executable, "-c",
                            "from kgl.state import SessionStore, Session; from pathlib import Path; "
                            "import sys; SessionStore(Path(sys.argv[1])).save(Session('http://new','b'))",
                            directory], check=True)
            with self.assertRaises(KglError):
                store.save(old, expected=old)
            with self.assertRaises(KglError):
                store.clear(expected=old)
            self.assertEqual(store.load().kernel_id, "b")

    def test_interrupted_writer_preserves_previous_session(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))
            store.save(Session("http://old", "a"))
            child = subprocess.run([sys.executable, "-c",
                                    "from kgl.state import SessionStore, Session; from pathlib import Path; "
                                    "import sys, os; os.replace = lambda *a: os._exit(19); "
                                    "SessionStore(Path(sys.argv[1])).save(Session('http://new','b'))",
                                    directory])
            self.assertEqual(child.returncode, 19)
            self.assertEqual(store.load().kernel_id, "a")
            store.save(Session("http://next", "c"), expected=store.load())
            self.assertEqual(store.load().kernel_id, "c")
