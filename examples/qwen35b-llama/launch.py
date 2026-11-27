"""Operate the checkout desktop Qwen session from a terminal."""
import argparse
from dataclasses import asdict, replace
import fcntl
import getpass
import json
import os
from pathlib import Path
import subprocess
import sys

from kgl.cli.app import build_parser
from kgl.cli.commands.serve import start_worker
from kgl.client.model_fetch import read_preset
from kgl.client.serve_manager import ServeManager
from kgl.core.config import Settings
from kgl.core.errors import ExecutionError, KglError
from kgl.core.state import SessionStore
from kgl.desktop.kaggle import KaggleSessions
from kgl.desktop.launcher import Launcher, LaunchError
from kgl.transport.runtime import Runtime

import proxy
from ready import PRESET, QwenError, prepare_qwen


class Qwen:
    def __init__(self, settings):
        self.settings = settings
        self.store = SessionStore(settings.home)

    def _env(self):
        return {**os.environ, "KGL_HOME": str(self.settings.home), "KGL_WORKDIR": self.settings.workdir}

    def preflight(self, rec):
        read_preset(PRESET)
        if hasattr(self.proxy, "mock_calls"):
            return
        try:
            pid = proxy.listener()
            if pid is not None:
                proxy.check_owner(pid, self.settings.home)
        except FileNotFoundError:
            return
        except RuntimeError as exc:
            if "8001" in str(exc):
                raise QwenError("port_conflict", "Port 8001 is occupied. Check the existing proxy before retrying.") from None
            raise

    def connect(self, url):
        subprocess.run([sys.executable, "-m", "kgl", "connect", "--proxy-url", "-"],
                       input=url.strip() + "\n", text=True, capture_output=True, env=self._env(), check=True)

    def prepare(self, session, phase, recorded):
        preset = read_preset(PRESET)

        def start(arguments):
            self.preflight(None)
            args = build_parser().parse_args(arguments)
            rt = Runtime(session, self.settings)
            observed = {}

            def remember(updated, result):
                observed.update(result)
                recorded(updated, result)

            try:
                start_worker(args, rt, session, self.store, recorded=remember,
                             progress=lambda line: None, output=lambda stream, data: None)
                recorded(session, observed)
            except ExecutionError as exc:
                if exc.ename == "ValueError" and exc.evalue.startswith("SHA-256 mismatch for "):
                    raise QwenError("checksum", "GGUF checksum differs from the preset.") from None
                raise
            finally:
                rt.close()

        prepare_qwen(preset, preset["default_quant"], preset["context"],
                     phase=phase, start=start, quiet=True, settings=self.settings)

    def proxy(self, session, rec):
        home = self.settings.home
        worker = session.serve
        if not worker or not rec.worker or worker.instance != rec.worker:
            raise LaunchError("Expected worker instance changed.")
        with (home / "qwen-proxy.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                previous = (rec.proxy_pid, rec.proxy_birth) if rec.proxy_owned else (0, "")
                action, pid = proxy.manage(home, worker, replace_only=previous, env=self._env())
            except RuntimeError as exc:
                if "8001" in str(exc):
                    raise LaunchError("Port 8001 is occupied. Check the existing proxy before retrying.") from None
                raise
            birth = proxy.process_birth(pid)
            return action, pid, birth

    def refresh(self, session, rec):
        if not rec.worker or not session.serve:
            rt = Runtime(session, self.settings)
            try:
                rt.client.get_kernel(session.kernel_id)
            except (KglError, OSError):
                return "unknown", "connection_lost"
            finally:
                rt.close()
            return ("worker-stopped" if rec.phase == "worker-stopped" else "attached"), ""
        if session.serve.instance != rec.worker:
            return "unknown", "worker_mismatch"
        try:
            pid = proxy.listener()
        except (RuntimeError, OSError, subprocess.SubprocessError):
            return "unknown", "port_conflict"
        if not pid:
            return "unknown", "proxy_down"
        try:
            proxy.check_owner(pid, self.settings.home)
        except FileNotFoundError:
            return "unknown", "proxy_down"
        except (RuntimeError, OSError):
            return "unknown", "port_conflict"
        try:
            result = proxy.probe()
        except (OSError, ValueError):
            return "unknown", "proxy_unreachable"
        state = result[1].get("state")
        if state == "busy" and result[1].get("instance") == rec.worker:
            return "busy", ""
        if proxy.matches(result, session.serve):
            return "ready", ""
        if state == "connection_lost":
            return "unknown", "connection_lost"
        return "unknown", "worker_unready"

    def stop(self, session):
        rt = Runtime(session, self.settings)
        try:
            result = ServeManager(rt, session.serve.port).stop(session.serve.instance)
            if result.get("stopped") is not True:
                raise LaunchError("Worker stop was not confirmed.")
            session.serve = None
            self.store.save(session, expected=session)
        finally:
            rt.close()

    def stop_proxy(self, rec):
        if rec.proxy_owned and rec.proxy_pid:
            proxy.stop_owned(self.settings.home, rec.proxy_pid, rec.proxy_birth)


def make_launcher(base_home=None, *, kaggle=None):
    current = Settings.from_env()
    base = Path(base_home) if base_home is not None else current.home
    home = (base / "desktop").resolve()
    sdk = kaggle if kaggle is not None else (KaggleSessions() if KaggleSessions.available() else None)
    return Launcher(home, Qwen(replace(current, home=home)), kaggle=sdk)


def _quote_desktop_arg(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$").replace("`", "\\`")
    return f'"{escaped}"'


def desktop_entry_path(apps_dir: Path | None = None) -> Path:
    if apps_dir is not None:
        return Path(apps_dir) / "kgl-desktop.desktop"
    data_home = Path(os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local/share"))
    return data_home / "applications/kgl-desktop.desktop"


def install_desktop_entry(apps_dir: Path | None = None, *, python_exe: str | None = None) -> Path:
    here = Path(__file__).resolve().parent
    repo_root = here.parents[1]
    script = here / "desktop.py"
    icon = here / "kgl-desktop.svg"
    exe = str(Path(python_exe or sys.executable).resolve())
    target = desktop_entry_path(apps_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join([
        "[Desktop Entry]",
        "Version=1.0",
        "Type=Application",
        "Name=KGL Desktop",
        "Comment=Khởi chạy Qwen 35B trên Kaggle GPU và mở hội thoại local",
        f"Exec={_quote_desktop_arg(exe)} {_quote_desktop_arg(str(script))}",
        f"Path={repo_root}",
        f"Icon={icon}",
        "Terminal=false",
        "Categories=Development;Science;Utility;",
        "StartupNotify=true",
        "StartupWMClass=KglDesktop",
        "",
    ])
    tmp = target.with_name(".kgl-desktop.tmp")
    tmp.write_text(content)
    tmp.chmod(0o755)
    os.replace(tmp, target)
    return target


def uninstall_desktop_entry(apps_dir: Path | None = None) -> Path:
    target = desktop_entry_path(apps_dir)
    target.unlink(missing_ok=True)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    subs.add_parser("attach").add_argument("--notebook", required=True)
    subs.add_parser("start").add_argument("--notebook", default="")
    status = subs.add_parser("status")
    status.add_argument("--refresh", action="store_true")
    for action in ("stop-worker", "end-session", "detach", "sdk-check", "install-desktop", "uninstall-desktop"):
        subs.add_parser(action)
    args = parser.parse_args()
    if args.action == "sdk-check":
        from kgl.desktop.kaggle import inspect_sdk
        print(json.dumps(inspect_sdk()))
        return
    if args.action == "install-desktop":
        print(json.dumps({"installed": str(install_desktop_entry())}))
        return
    if args.action == "uninstall-desktop":
        print(json.dumps({"removed": str(uninstall_desktop_entry())}))
        return
    launcher = make_launcher()
    if args.action == "status" and not args.refresh:
        record = launcher.status()
    else:
        url = ""
        if args.action == "attach":
            url = getpass.getpass("Jupyter URL: ") if sys.stdin.isatty() else sys.stdin.readline()
        record = launcher.execute("refresh" if args.action == "status" else args.action,
                                  getattr(args, "notebook", ""), url,
                                  emit=lambda event: print(json.dumps(event), file=sys.stderr))
    print(json.dumps(asdict(record)))


if __name__ == "__main__":
    try:
        main()
    except ImportError:
        sys.exit("sdk-check requires the desktop extra on Python 3.11+.")
    except (KglError, OSError, ValueError) as exc:
        sys.exit(str(exc))
