"""Offline launcher contracts. All sessions and ownership records are synthetic."""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

from kgl.core.state import Session, ServeRecord
from kgl.desktop.kaggle import KaggleAdapterError, KaggleSessions
from kgl.desktop.launcher import Launcher, LaunchError, parse_notebook

ROOT = Path(__file__).resolve().parents[1]
SRC = str(ROOT / "src")
EXAMPLE = ROOT / "examples/qwen35b-llama"
SECRET = "https://example.invalid/?token=fixture-secret"


class LauncherTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name)
        self.app = Mock()
        self.launcher = Launcher(self.home, self.app)
        self.app.connect.side_effect = lambda url: self.launcher.sessions.save(Session(url, "kernel"))
        self.started = True
        self.app.prepare.side_effect = self.prepare
        self.app.proxy.return_value = ("started", 123, "567")
        self.events = []

    def prepare(self, session, phase, recorded):
        for name in ("input", "build", "worker"):
            phase(name)
        session.serve = ServeRecord(8000, "model", "llama", instance="instance")
        self.launcher.sessions.save(session, expected=session)
        recorded(session, dict(instance="instance", started=self.started))
        phase("health")

    def attach(self):
        return self.launcher.execute("attach", "account/book", SECRET, self.events.append)

    def test_restart_reuse_stop_and_detach(self):
        rec = self.attach()
        self.assertEqual(rec.phase, "ready")
        self.assertEqual(rec.owned_worker, "instance")
        self.assertEqual(self.launcher.path.stat().st_mode & 0o777, 0o600)
        self.assertNotIn("fixture-secret", self.launcher.path.read_text() + str(self.events))
        self.launcher = Launcher(self.home, self.app)
        self.started = False
        self.app.proxy.return_value = ("reused", 123, "567")
        again = self.launcher.execute("start", rec.notebook)
        with self.assertRaisesRegex(LaunchError, "session changed"):
            self.launcher.execute("stop-worker", expected_run="previous-attachment")
        self.app.stop.assert_not_called()
        self.assertEqual(again.owned_worker, "instance")
        self.assertTrue(again.proxy_owned)
        self.assertEqual(self.launcher.execute("stop-worker").phase, "worker-stopped")
        self.app.stop.assert_called_once()
        self.assertEqual(self.launcher.execute("detach").phase, "idle")
        self.app.stop_proxy.assert_called_once()
        self.assertIsNone(self.launcher.sessions.load())

    def test_foreign_worker_and_changed_revision(self):
        self.started = False
        self.attach()
        with self.assertRaises(LaunchError):
            self.launcher.execute("stop-worker")
        self.app.stop.assert_not_called()
        self.launcher.sessions.save(Session(SECRET, "other"))
        with self.assertRaises(LaunchError):
            self.launcher.execute("detach")
        self.app.stop_proxy.assert_not_called()
        self.assertEqual(self.launcher.sessions.load().kernel_id, "other")

    def test_cancel_after_start_before_proxy(self):
        cancel = threading.Event()

        def emit(event):
            if event["phase"] == "worker":
                cancel.set()

        with self.assertRaisesRegex(LaunchError, "cancelled"):
            self.launcher.execute("attach", "a/b", SECRET, emit, cancel)
        self.assertEqual(self.launcher.read().error_code, "cancelled")
        self.assertEqual(self.launcher.read().owned_worker, "instance")
        self.app.proxy.assert_not_called()

    def test_crash_lock_and_schema(self):
        record = self.attach()
        self.launcher.save(replace(record, phase="worker"))
        with self.launcher.lock():
            with self.assertRaisesRegex(LaunchError, "running"):
                Launcher(self.home, self.app).execute("attach", "other/book", SECRET)
        self.assertEqual(self.launcher.status().phase, "unknown")
        self.launcher.path.write_text('{"schema":2}')
        with self.assertRaisesRegex(LaunchError, "Inspect launcher.json"):
            self.launcher.execute("start", "a/b")
        self.assertEqual(self.launcher.path.read_text(), '{"schema":2}')

    def test_killed_launcher_releases_lock_with_child_alive(self):
        code = """
import subprocess, sys, time
from kgl.desktop.launcher import Launcher, Record
with Launcher(sys.argv[1], None).lock():
    Launcher(sys.argv[1], None).save(Record(notebook="a/b", source="attached", phase="connect"))
    subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()"],
                     stdout=subprocess.DEVNULL, close_fds=True)
    print("locked", flush=True)
    time.sleep(30)
"""
        with subprocess.Popen([sys.executable, "-c", code, str(self.home)],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True) as owner:
            try:
                self.assertEqual(owner.stdout.readline().strip(), "locked")
                with self.assertRaises(LaunchError), self.launcher.lock():
                    pass
            finally:
                owner.kill()
                owner.wait()
            # The child still has its stdin pipe open until Popen exits the context.
            with self.launcher.lock():
                pass
            self.assertEqual(self.launcher.status().error_code, "interrupted_connect")

    def test_unverified_gpu_operations_are_blocked(self):
        for action in ("start", "end-session"):
            with self.assertRaisesRegex(LaunchError, "M0"):
                self.launcher.execute(action, "owner/book")
        self.assertFalse(self.app.mock_calls)

    def test_notebook_url_saved_config_and_auto_session_lifecycle(self):
        self.assertEqual(parse_notebook("https://www.kaggle.com/code/Owner_1/Test-KGL/edit?x=1"), "owner_1/test-kgl")
        self.assertEqual(parse_notebook("kaggle.com/kernels/owner/book/"), "owner/book")
        self.assertEqual(parse_notebook("https://example.com/code/owner/book"), "")
        self.assertEqual(parse_notebook("test-kgl"), "")

        # Contract fixture mirroring sanitized M0 live responses (kagglesdk==0.1.38, 2026-11-27).
        class FakeOp(dict):
            @classmethod
            def to_dict(cls, instance):
                return dict(instance)

        kclient = Mock()
        oclient = Mock()
        kclient.get_kernel.return_value = Mock(metadata=Mock(
            ref="owner/test-kgl", kernel_type="notebook", language="python",
            enable_gpu=True, enable_internet=True, machine_shape="NvidiaTeslaT4", docker_image="img",
        ))
        kclient.create_kernel_session.return_value = FakeOp(
            name="operations/create-kernel-session/100001",
            metadata={"kernelSessionId": 100001},
        )
        oclient.get_operation.side_effect = [
            FakeOp(name="operations/create-kernel-session/100001", done=False),
            FakeOp(name="operations/create-kernel-session/100001", done=True, response={
                "jupyterUrl": "https://kkb.invalid/base",
                "tokenizedJupyterUrl": SECRET,
            }),
            FakeOp(name="operations/create-kernel-session/100001", done=True, error={"code": 1}),
        ]
        kclient.cancel_kernel_session.return_value = Mock(error_message="")
        ctx = Mock()
        ctx.__enter__ = Mock(return_value=Mock(
            kernels=Mock(kernels_api_client=kclient), common=Mock(operations_client=oclient),
        ))
        ctx.__exit__ = Mock(return_value=False)
        sdk = KaggleSessions(client_factory=lambda: ctx)
        auto = Launcher(self.home, self.app, kaggle=sdk)

        with patch("kgl.desktop.kaggle.time.sleep"):
            rec = auto.execute("start", "https://www.kaggle.com/code/Owner/Test-KGL/")
        self.assertEqual((rec.phase, rec.source, rec.notebook, rec.session_id),
                         ("ready", "created", "owner/test-kgl", 100001))
        self.assertEqual(auto.saved_notebook(), "owner/test-kgl")
        auto.remember_notebook("owner/second-book")
        auto.remember_notebook("owner/test-kgl")
        self.assertEqual(auto.recent_notebooks(), ["owner/test-kgl", "owner/second-book"])
        self.assertEqual(auto.saved_language(), "vi")
        auto.remember_language("en")
        self.assertEqual(auto.saved_language(), "en")
        self.assertEqual(auto.saved_notebook(), "owner/test-kgl")
        auto.remember_language("vi")
        self.assertIsNone(auto.quota())
        auto.kaggle = Mock(quota=Mock(return_value={"remaining_s": 3600.0, "total_s": 108000.0}))
        self.assertEqual(auto.quota()["remaining_s"], 3600.0)
        auto.kaggle = sdk
        self.assertNotIn("fixture-secret", auto.path.read_text())
        # Created GPU session must use end-session, never local-only detach.
        with self.assertRaisesRegex(LaunchError, "end-session"):
            auto.execute("detach")
        with patch("kgl.desktop.kaggle.JupyterClient.list_kernels", side_effect=LaunchError("gone")):
            ended = auto.execute("end-session")
        self.assertEqual((ended.phase, ended.notebook, ended.session_id), ("idle", "owner/test-kgl", 0))
        self.assertEqual(auto.status().notebook, "owner/test-kgl")
        kclient.cancel_kernel_session.assert_called_once()

        # Startup failure after session creation automatically cancels the new GPU session; unconfirmed cancel keeps ID.
        kclient.create_kernel_session.return_value = FakeOp(
            name="operations/create-kernel-session/100002", metadata={"kernelSessionId": 100002},
            done=True, response={"tokenizedJupyterUrl": SECRET},
        )
        self.app.prepare.side_effect = RuntimeError("boom")
        oclient.get_operation.side_effect = [FakeOp(name="operations/create-kernel-session/100002", done=True, error={"code": 1})]
        with patch("kgl.desktop.kaggle.JupyterClient.list_kernels", side_effect=LaunchError("gone")), \
                self.assertRaises(LaunchError):
            auto.execute("start")  # Uses saved_notebook() without re-specifying --notebook
        self.assertEqual((auto.read().phase, auto.read().session_id), ("idle", 0))

        # Lost HTTP response during CreateKernelSession leaves unknown state so user does not double-allocate.
        kclient.create_kernel_session.side_effect = OSError("socket reset")
        with self.assertRaisesRegex(LaunchError, "Lost response"):
            auto.execute("start")
        self.assertEqual((auto.read().phase, auto.read().error_code), ("unknown", "create_uncertain"))

    def test_failed_connect_missing_session_detach_and_worker_reconcile(self):
        self.app.connect.side_effect = RuntimeError("bad url")
        with self.assertRaisesRegex(LaunchError, "Jupyter URL"):
            self.launcher.execute("attach", "account/book", SECRET)
        rec = self.launcher.read()
        self.assertEqual((rec.phase, rec.notebook, rec.error_code), ("idle", "account/book", "failed_connect"))
        self.app.connect.side_effect = lambda url: self.launcher.sessions.save(Session(url, "kernel"))
        rec = self.attach()
        self.assertTrue(rec.proxy_owned)
        # Simulate crash between SessionStore.save(ready) and Launcher.save(rec) during worker phase.
        session = self.launcher.sessions.load()
        self.launcher.sessions.save(session, expected=session)
        self.launcher.save(replace(rec, phase="worker", proxy_owned=False, proxy_pid=0, proxy_birth=""))
        self.started = False
        self.app.proxy.return_value = ("started", 456, "789")
        recovered = self.launcher.execute("start", rec.notebook)
        self.assertEqual((recovered.phase, recovered.revision), ("ready", self.launcher.sessions.load().revision))
        # Simulate external kgl stop clearing session.json; detach must still clean up owned proxy and reset.
        self.launcher.sessions.clear()
        with self.assertRaisesRegex(LaunchError, "Session record changed"):
            self.launcher.execute("refresh")
        self.assertEqual(self.launcher.execute("detach").phase, "idle")
        self.app.stop_proxy.assert_called_once()

    def test_qwen_refresh_and_idempotent_make_launcher(self):
        from kgl.core.config import Settings
        from kgl.core.errors import ConnectionLost
        with patch.object(sys, "path", [str(EXAMPLE), *sys.path]):
            import launch
        before = os.environ.get("KGL_HOME")
        with patch.dict(os.environ, {"KGL_HOME": str(self.home)}):
            first = launch.make_launcher()
            second = launch.make_launcher()
            self.assertEqual(os.environ["KGL_HOME"], str(self.home))
        self.assertEqual(os.environ.get("KGL_HOME"), before)
        self.assertEqual(first.home, (self.home / "desktop").resolve())
        self.assertEqual(first.home, second.home)
        apps_dir = self.home / "applications"
        entry = launch.install_desktop_entry(apps_dir, python_exe=sys.executable)
        text = entry.read_text()
        self.assertIn("StartupWMClass=KglDesktop", text)
        self.assertIn("kgl-desktop.svg", text)
        self.assertEqual(entry.stat().st_mode & 0o777, 0o755)
        launch.uninstall_desktop_entry(apps_dir)
        self.assertFalse(entry.exists())
        qwen = launch.Qwen(Settings(home=self.home))
        launcher = Launcher(self.home, qwen)
        session = Session(SECRET, "kernel")
        launcher.sessions.save(session)
        rec = replace(launcher.read(), notebook="a/b", run="r1", source="attached",
                      phase="worker-stopped", kernel="kernel", revision=session.revision)
        launcher.save(rec)
        with patch("kgl.transport.jupyter.JupyterClient.get_kernel", return_value={"execution_state": "idle"}):
            self.assertEqual(launcher.execute("refresh").phase, "worker-stopped")
        with patch("kgl.transport.jupyter.JupyterClient.get_kernel", side_effect=ConnectionLost("gone")):
            out = launcher.execute("refresh")
            self.assertEqual((out.phase, out.error_code), ("unknown", "connection_lost"))
        session.serve = ServeRecord(8000, "model", "llama", instance="inst")
        launcher.sessions.save(session, expected=session)
        launcher.save(replace(rec, phase="ready", worker="inst", revision=session.revision))
        with patch.object(launch.proxy, "listener", return_value=None):
            out = launcher.execute("refresh")
            self.assertEqual((out.phase, out.error_code), ("unknown", "proxy_down"))
        with patch.object(launch.proxy, "listener", side_effect=RuntimeError("foreign")):
            out = launcher.execute("refresh")
            self.assertEqual((out.phase, out.error_code), ("unknown", "port_conflict"))
        with patch.object(launch.proxy, "listener", return_value=123), \
                patch.object(launch.proxy, "check_owner"), \
                patch.object(launch.proxy, "probe", return_value=(503, {"state": "connection_lost"})):
            out = launcher.execute("refresh")
            self.assertEqual((out.phase, out.error_code), ("unknown", "connection_lost"))

    def test_cli_attach_fake_jupyter_missing_input(self):
        from .fake_jupyter import FakeJupyter
        server = FakeJupyter(token="fixture-secret")
        self.addCleanup(server.stop)
        cmd = [sys.executable, str(EXAMPLE / "launch.py")]
        pythonpath = os.pathsep.join(filter(None, [SRC, os.environ.get("PYTHONPATH")]))
        env = dict(os.environ, PYTHONPATH=pythonpath, KGL_HOME=str(self.home), KGL_WORKDIR=str(self.home))
        result = subprocess.run([*cmd, "attach", "--notebook", "a/b"], capture_output=True, text=True,
                                input=server.url + "?token=fixture-secret\n", env=env, cwd=self.home)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("fixture-secret", result.stderr + result.stdout)
        status = json.loads(subprocess.check_output([*cmd, "status"], env=env))
        self.assertEqual(status["error_code"], "missing_input", result.stderr)
        self.assertTrue(status["kernel"])
        subprocess.run([*cmd, "detach"], env=env, capture_output=True, check=True)
        self.assertTrue(server.kernels)


class TtkTests(unittest.TestCase):
    def test_window_with_fixture(self):
        try:
            import tkinter as tk
        except ImportError:
            self.skipTest("tkinter unavailable")
        try:
            root = tk.Tk()
        except tk.TclError:
            self.skipTest("desktop display unavailable")
        self.addCleanup(root.destroy)
        with patch.object(sys, "path", [str(EXAMPLE), *sys.path]):
            from desktop import Window
        with tempfile.TemporaryDirectory() as tmp:
            window = Window(root, Launcher(Path(tmp), Mock()))
            root.update_idletasks()
            self.assertEqual(str(window.buttons["attach"]["state"]), "normal")
            self.assertEqual(str(window.buttons["start"]["state"]), "disabled")
            self.assertEqual(str(window.buttons["chat"]["state"]), "disabled")
            def wait_for_result():
                deadline = time.monotonic() + 4
                while window.running and time.monotonic() < deadline:
                    root.update()
                    time.sleep(0.01)
                self.assertFalse(window.running)

            app = window.launcher.app
            store = window.launcher.sessions
            app.connect.side_effect = lambda url: store.save(Session(url, "fixture"))
            def prepare(session, phase, recorded):
                phase("worker")
                session.serve = ServeRecord(8000, "model", "llama", instance="instance")
                store.save(session, expected=session)
                recorded(session, {"started": True, "instance": "instance"})

            app.prepare.side_effect = prepare
            app.proxy.return_value = ("reused", 123, "567")
            window.slug.set("owner/book")
            with patch("desktop.simpledialog.askstring", return_value=SECRET):
                window.buttons["attach"].invoke()
            self.assertEqual(str(window.buttons["start"]["state"]), "disabled")
            wait_for_result()
            self.assertEqual(str(window.buttons["chat"]["state"]), "normal")
            with patch("desktop.webbrowser.open") as opened:
                window.buttons["chat"].invoke()
                opened.assert_called_once_with("http://127.0.0.1:8001/")
            with patch("desktop.messagebox.askyesno", return_value=False) as confirm:
                window.buttons["detach"].invoke()
                self.assertIn("Dừng worker", confirm.call_args.args[1])
            window.running = True
            window.cancellable = True
            with patch("desktop.messagebox.askyesno", return_value=True) as ask, \
                    patch("desktop.messagebox.showinfo") as info:
                window.close()
                self.assertTrue(window.cancel.is_set())
                window.close()
                ask.assert_called_once()
                info.assert_called_once()
            window.running = False
            window.cancel.clear()
            app.refresh.return_value = "busy"
            window.buttons["refresh"].invoke()
            wait_for_result()
            self.assertEqual(window.record.phase, "busy")
            self.assertEqual(str(window.buttons["chat"]["state"]), "disabled")
            with patch("desktop.messagebox.askyesno", return_value=True):
                window.buttons["detach"].invoke()
            wait_for_result()
            self.assertEqual(window.record.phase, "idle")
            self.assertEqual(window.slug.get(), "owner/book")

            # Verify 1-click auto-create, quota badge, contextual buttons, and end-session UI controls.
            kaggle = Mock()
            kaggle.create.return_value = ("operations/create-kernel-session/99", 99, SECRET)
            kaggle.quota.return_value = {"remaining_s": 46800.0, "total_s": 108000.0}
            window.launcher.kaggle = kaggle
            window.launcher.can_create = True
            window.controls()
            self.assertEqual(str(window.buttons["start"]["state"]), "normal")
            self.assertEqual(window.buttons["start"].winfo_manager(), "grid")
            self.assertEqual(window.buttons["end-session"].winfo_manager(), "")
            window.slug.set("invalid-without-owner")
            window.buttons["start"].invoke()
            self.assertFalse(window.running)
            window.slug.set("https://www.kaggle.com/code/Owner/Book/")
            window.buttons["start"].invoke()
            wait_for_result()
            self.assertEqual(window.slug.get(), "owner/book")
            self.assertEqual(window.record.phase, "ready")
            self.assertEqual(window.quota_var.get(), "GPU tuần: còn 13h 00p / 30h")
            self.assertEqual(str(window.buttons["end-session"]["state"]), "normal")
            self.assertEqual(window.buttons["end-session"].winfo_manager(), "grid")
            self.assertEqual(window.buttons["detach"].winfo_manager(), "")
            self.assertEqual(str(window.buttons["detach"]["state"]), "disabled")
            window.copy_api()
            self.assertEqual(root.clipboard_get(), "http://127.0.0.1:8001/v1")
            with patch("desktop.messagebox.askyesno", return_value=True):
                window.buttons["end-session"].invoke()
            wait_for_result()
            kaggle.cancel.assert_called_once()
            self.assertEqual((window.record.phase, window.slug.get()), ("idle", "owner/book"))

            # Verify resolved font never falls back to X11 'fixed' and VI <-> EN switch updates live.
            import tkinter.font as tkf
            self.assertNotEqual(tkf.Font(root=root, family=window.font_family, size=10).actual("family"), "fixed")
            window.lang_btn.invoke()
            self.assertEqual(window.lang, "en")
            self.assertEqual(window.launcher.saved_language(), "en")
            self.assertIn("Start Qwen", str(window.buttons["start"]["text"]))
            self.assertEqual(window.quota_var.get(), "Weekly GPU: 13h 00m / 30h left")
            self.assertIn("No active session", window.message.get())
            self.assertNotEqual(tkf.Font(root=root, family=window.font_family, size=10).actual("family"), "fixed")
            window.lang_btn.invoke()
            self.assertEqual(window.lang, "vi")
            self.assertIn("Chạy Qwen", str(window.buttons["start"]["text"]))


class QwenFlowTests(unittest.TestCase):
    """Real KGL transport, checksumming and gateway with synthetic GPU/model."""
    def test_qwen_worker_reuse_stream_and_stop(self):
        import hashlib
        from urllib.request import Request, urlopen
        from kgl.core.config import Settings
        from kgl.proxy.handler import make_proxy_server
        from kgl.transport.runtime import Runtime
        from .fake_jupyter import FakeJupyter
        from .test_llama import UPSTREAM
        with patch.object(sys, "path", [str(EXAMPLE), *sys.path]):
            import launch
            import ready
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            home = root / "home"
            preset = json.loads(ready.PRESET.read_text())
            quant = preset["default_quant"]
            model = root / preset["files"][quant]
            model.write_bytes(b"GGUF")
            preset["checksums"][quant] = hashlib.sha256(model.read_bytes()).hexdigest()
            preset["inputs"][quant] = str(model)
            preset["context"] = 2048
            source = root / "preset.json"
            source.write_text(json.dumps(preset))
            revision = preset["build"]["revision"]
            build = root / revision
            build.mkdir()
            binary = build / "llama"
            binary.write_text(f"#!{sys.executable}\nimport sys\nif '--version' in sys.argv:\n"
                              f"    print('{revision}')\n    sys.exit(0)\n" + UPSTREAM)
            binary.chmod(0o700)
            options = {k: "--" + k for k in ("model", "host", "port", "ctx", "gpu_layers",
                                               "split_mode", "parallel", "tensor_split")}
            (build / "manifest.json").write_text(json.dumps(dict(revision=revision, cuda_arch=75,
                                                                 argv=["llama"], options=options)))
            smi = root / "nvidia-smi"
            smi.write_text("#!/bin/sh\ncase \"$1\" in\n"
                           "*gpu=name) printf 'Tesla T4\\nTesla T4\\n';;\n"
                           "*uuid,index) printf 'A, 0\\nB, 1\\n';;\n"
                           "*compute-apps*) printf '%s, A, 42\\n%s, B, 42\\n' \"$PPID\" \"$PPID\";;\nesac\n")
            # Report the gateway PID (parent of the simulated nvidia-smi).
            smi.chmod(0o700)
            pythonpath = os.pathsep.join(filter(None, [SRC, os.environ.get("PYTHONPATH")]))
            env = dict(PYTHONPATH=pythonpath, KGL_HOME=str(root / "ambient-home"),
                       KGL_WORKDIR=str(root / "ambient-work"), PATH=tmp + os.pathsep + os.environ["PATH"])
            with patch.dict(os.environ, env), patch.object(launch, "PRESET", source), \
                    patch.object(ready, "PRESET", source), patch.object(ready, "INSTALL_DIR", tmp):
                server = FakeJupyter()
                settings = Settings(home=home, workdir=tmp)
                app = launch.Qwen(settings)
                launcher = Launcher(home, app)
                try:
                    # Port 8001 belongs to the user. Exercise HTTP on a free test port.
                    with patch.object(app, "proxy", return_value=("reused", 123, "567")):
                        (build / "manifest.json").write_text(json.dumps(dict(revision=revision, cuda_arch=70)))
                        with self.assertRaisesRegex(LaunchError, "verification"):
                            launcher.execute("attach", "demo/test", server.url)
                        self.assertEqual(launcher.read().error_code, "build_verify")
                        self.assertFalse((root / "ambient-home" / "session.json").exists())
                        (build / "manifest.json").unlink()
                        binary.unlink()
                        build.rmdir()
                        with self.assertRaisesRegex(LaunchError, "dataset"):
                            launcher.execute("start", "demo/test")
                        self.assertEqual(launcher.read().error_code, "build_access")
                        build.mkdir()
                        binary.write_text(f"#!{sys.executable}\nimport sys\nif '--version' in sys.argv:\n"
                                          f"    print('{revision}')\n    sys.exit(0)\n" + UPSTREAM)
                        binary.chmod(0o700)
                        (build / "manifest.json").write_text(json.dumps(dict(revision=revision, cuda_arch=75,
                                                                             argv=["llama"], options=options)))
                        digest = preset["checksums"][quant]
                        preset["checksums"][quant] = "0" * 64
                        source.write_text(json.dumps(preset))
                        with self.assertRaisesRegex(LaunchError, "checksum"):
                            launcher.execute("start", "demo/test")
                        self.assertEqual(launcher.read().error_code, "checksum")
                        self.assertFalse(launcher.read().owned_worker)
                        preset["checksums"][quant] = digest
                        source.write_text(json.dumps(preset))
                        record = launcher.execute("start", "demo/test")
                        again = launcher.execute("start", "demo/test")
                    self.assertTrue(record.owned_worker)
                    self.assertEqual(again.owned_worker, record.owned_worker)
                    session = launcher.sessions.load()
                    rt = Runtime(session, settings)
                    try:
                        with make_proxy_server(rt, session.serve.port, session.serve.model, listen_port=0) as srv:
                            thread = threading.Thread(target=srv.serve_forever)
                            thread.start()
                            try:
                                body = json.dumps({"model": "qwen-local", "stream": True,
                                                   "messages": [{"role": "user", "content": "test"}]})
                                req = Request(f"http://127.0.0.1:{srv.server_port}/v1/chat/completions",
                                              data=body.encode(), headers={"Content-Type": "application/json"})
                                with urlopen(req, timeout=20) as response:
                                    stream = response.read().decode()
                                self.assertIn("answer", stream)
                                self.assertIn("data: [DONE]", stream)
                            finally:
                                srv.shutdown()
                                thread.join()
                    finally:
                        rt.close()
                    self.assertEqual(launcher.execute("stop-worker").phase, "worker-stopped")
                    self.assertEqual(launcher.execute("detach").phase, "idle")
                finally:
                    try:
                        session = launcher.sessions.load()
                        if session and session.serve:
                            app.stop(session)
                    finally:
                        server.stop()
