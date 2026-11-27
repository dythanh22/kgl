"""Linux checkout launcher. Session credentials are managed by SessionStore."""
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import fcntl
import json
import os
from pathlib import Path
import re
import tempfile
import time
from urllib.parse import urlsplit
import uuid

from ..core.errors import KglError
from ..core.state import SessionStore


class LaunchError(KglError):
    """A user-facing message containing no remote output."""


SLUG = re.compile(r"[a-z0-9][a-z0-9_-]*/[a-z0-9][a-z0-9_-]*")
PHASES = {"idle", "create", "connect", "attached", "input", "build", "worker", "health",
          "proxy", "ready", "busy", "worker-stopped", "unknown", "detach", "stop-worker", "end-session"}
ACTIVE = {"create", "connect", "input", "build", "worker", "health", "proxy", "detach", "stop-worker", "end-session"}
ERROR_CODES = {"missing_input", "wrong_gpu", "gpu_unavailable", "input_probe",
               "checksum", "build_verify", "build_access", "port_conflict",
               "auth_failed", "notebook_not_found", "quota_exceeded", "sdk_missing", "sdk_version",
               "metadata_failed", "create_failed", "create_rejected", "create_uncertain",
               "create_timeout", "poll_failed", "missing_url", "cancel_unconfirmed"}


def parse_notebook(value: str) -> str:
    """Normalize 'owner/slug' or a https://www.kaggle.com/code/owner/slug URL."""
    raw = (value or "").strip()
    if raw.startswith(("kaggle.com/", "www.kaggle.com/")):
        raw = "https://" + raw
    if "://" in raw:
        sp = urlsplit(raw)
        if sp.scheme not in ("http", "https") or sp.hostname not in ("kaggle.com", "www.kaggle.com"):
            return ""
        parts = [p for p in sp.path.split("/") if p]
        if len(parts) >= 3 and parts[0] in ("code", "kernels"):
            candidate = f"{parts[1]}/{parts[2]}".lower()
            return candidate if SLUG.fullmatch(candidate) else ""
        return ""
    candidate = raw.strip("/").lower()
    return candidate if SLUG.fullmatch(candidate) else ""


def _atomic_write_json(directory: Path, target: Path, payload: dict) -> None:
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory.chmod(0o700)
    with tempfile.NamedTemporaryFile(mode="w", dir=directory, delete=False) as out:
        tmp = Path(out.name)
        try:
            json.dump(payload, out)
            out.flush()
            os.fsync(out.fileno())
            os.replace(tmp, target)
            fd = os.open(directory, os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        finally:
            tmp.unlink(missing_ok=True)


@dataclass
class Record:
    schema: int = 1
    notebook: str = ""
    run: str = ""
    source: str = ""
    phase: str = "idle"
    operation: str = ""
    session_id: int = 0
    kernel: str = ""
    revision: str = ""
    worker: str = ""
    owned_worker: str = ""
    proxy_pid: int = 0
    proxy_birth: str = ""
    proxy_action: str = ""
    proxy_owned: bool = False
    checked: float = 0.0
    error_code: str = ""


class Launcher:
    def __init__(self, home, app, kaggle=None):
        self.home = Path(home)
        self.sessions = SessionStore(home)
        self.path = self.home / "launcher.json"
        self.config_path = self.home / "config.json"
        self.app = app
        self.kaggle = kaggle
        self.can_create = kaggle is not None

    @contextmanager
    def lock(self):
        self.home.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.home.chmod(0o700)
        with (self.home / "launcher.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise LaunchError("Another launcher operation is running.") from None
            yield

    def _read_config(self) -> dict:
        try:
            data = json.loads(self.config_path.read_text())
            return data if isinstance(data, dict) else {}
        except (FileNotFoundError, ValueError, OSError):
            return {}

    def recent_notebooks(self) -> list[str]:
        data = self._read_config()
        items = []
        raw_list = data.get("recent") if isinstance(data.get("recent"), list) else []
        for candidate in [data.get("notebook"), *raw_list]:
            parsed = parse_notebook(str(candidate or ""))
            if parsed and parsed not in items:
                items.append(parsed)
        return items[:8]

    def saved_notebook(self) -> str:
        recent = self.recent_notebooks()
        return recent[0] if recent else ""

    def saved_language(self) -> str:
        lang = str(self._read_config().get("lang") or os.environ.get("KGL_LANG") or "vi").strip().lower()
        return "en" if lang == "en" else "vi"

    def remember_language(self, lang: str) -> None:
        chosen = "en" if str(lang).strip().lower() == "en" else "vi"
        cfg = self._read_config()
        payload = {"notebook": self.saved_notebook(), "recent": self.recent_notebooks(), "lang": chosen}
        if not payload["notebook"] and "notebook" in cfg:
            payload["notebook"] = ""
        _atomic_write_json(self.home, self.config_path, payload)

    def remember_notebook(self, notebook: str) -> None:
        if SLUG.fullmatch(notebook):
            recent = [notebook, *[item for item in self.recent_notebooks() if item != notebook]][:8]
            _atomic_write_json(self.home, self.config_path, {
                "notebook": notebook, "recent": recent, "lang": self.saved_language(),
            })

    def quota(self) -> dict | None:
        reader = getattr(self.kaggle, "quota", None)
        info = reader() if callable(reader) else None
        if (isinstance(info, dict)
                and all(isinstance(info.get(k), (int, float)) and not isinstance(info.get(k), bool)
                        for k in ("remaining_s", "total_s"))):
            return info
        return None

    def read(self):
        try:
            data = json.loads(self.path.read_text())
            defaults = asdict(Record())
            if not isinstance(data, dict) or not set(data).issubset(defaults) or "schema" not in data:
                raise ValueError
            merged = {**defaults, **data}
            rec = Record(**merged)
            if (any(type(merged[k]) is not type(v) for k, v in defaults.items())
                    or rec.schema != 1 or rec.phase not in PHASES or rec.session_id < 0
                    or (rec.notebook and not SLUG.fullmatch(rec.notebook))
                    or (rec.phase != "idle" and (rec.source not in ("attached", "created") or not rec.notebook))):
                raise ValueError
            return rec
        except FileNotFoundError:
            return Record()
        except (TypeError, ValueError):
            raise LaunchError("Inspect launcher.json before continuing (unsupported or damaged record).") from None

    def save(self, rec):
        _atomic_write_json(self.home, self.path, asdict(rec))

    def session(self, rec):
        session = self.sessions.load()
        if (session and rec.phase == "worker" and session.kernel_id == rec.kernel
                and session.serve and session.serve.instance == rec.worker
                and session.revision != rec.revision):
            rec.revision = session.revision
            self.save(rec)
        if not session or (session.kernel_id, session.revision) != (rec.kernel, rec.revision):
            raise LaunchError("Session record changed. Inspect the desktop session directory.")
        return session

    def status(self):
        rec = self.read()
        if rec.phase in ACTIVE:
            try:
                with self.lock():
                    rec = self.read()
                    if rec.phase in ACTIVE:
                        return replace(rec, phase="unknown", error_code="interrupted_" + rec.phase)
            except LaunchError:
                pass
        if rec.phase == "idle" and not rec.notebook:
            saved = self.saved_notebook()
            if saved:
                rec = replace(rec, notebook=saved)
        return rec

    def _preflight(self, rec):
        check = getattr(self.app, "preflight", None)
        if callable(check):
            check(rec)

    def execute(self, action, notebook="", url="", emit=lambda event: None, cancel=None, *, expected_run=None):
        with self.lock():
            rec = self.read()
            if expected_run is not None and expected_run != rec.run:
                raise LaunchError("The desktop session changed. Review the current notebook.")
            raw_notebook = notebook.strip()
            normalized = parse_notebook(raw_notebook) if raw_notebook else ""
            if raw_notebook and not normalized:
                raise LaunchError("Enter notebook as owner/slug or https://www.kaggle.com/code/owner/slug.")
            notebook = normalized or rec.notebook or self.saved_notebook()
            if action in ("start", "attach") and not SLUG.fullmatch(notebook):
                raise LaunchError("Enter notebook as owner/slug or https://www.kaggle.com/code/owner/slug.")

            was_idle = rec.phase == "idle"
            if (action == "end-session" or (action == "start" and was_idle)) and self.kaggle is None:
                raise LaunchError("Automatic GPU management awaits live M0 verification. Use Kaggle and attach.")

            if action == "attach":
                if not url.strip() or not was_idle or self.sessions.load():
                    raise LaunchError("Attach needs a URL and an empty desktop session. Inspect status first.")
                self.remember_notebook(notebook)
                rec = Record(notebook=notebook, run=uuid.uuid4().hex, source="attached")
            elif action == "start" and was_idle:
                if self.sessions.load():
                    raise LaunchError("Desktop session store is not empty. Inspect status or detach first.")
                self._preflight(rec)
                self.remember_notebook(notebook)
                rec = Record(notebook=notebook, run=uuid.uuid4().hex, source="created")
            elif action == "end-session":
                if rec.source != "created" or rec.session_id <= 0:
                    raise LaunchError("Only GPU sessions created by this launcher can be ended automatically.")
            elif action == "detach":
                if was_idle:
                    raise LaunchError("Select the attached notebook or attach its Jupyter URL first.")
                if rec.source == "created" and rec.session_id > 0:
                    raise LaunchError("Use end-session to stop and release the launcher-created GPU session.")
            elif was_idle or (normalized and normalized != rec.notebook):
                raise LaunchError("Select the attached notebook or attach its Jupyter URL first.")
            elif action == "start":
                self._preflight(rec)

            provisioning = action == "start" and was_idle
            last_phase = None

            def phase(name, *, check_cancel=True):
                nonlocal last_phase
                if last_phase:
                    emit(dict(phase=last_phase, outcome="done", error_code=""))
                last_phase = name
                if check_cancel and cancel and cancel.is_set():
                    raise LaunchError("Launch cancelled. Inspect status before detaching.")
                rec.phase = name
                rec.error_code = ""
                self.save(rec)
                emit(dict(phase=name, outcome="started", error_code=""))

            def recorded(session, result):
                rec.kernel, rec.revision = session.kernel_id, session.revision
                rec.worker = result["instance"]
                if result.get("started"):
                    rec.owned_worker = rec.worker
                elif rec.worker != rec.owned_worker:
                    rec.owned_worker = ""
                self.save(rec)

            try:
                error_code = ""
                if action == "detach":
                    current = self.sessions.load()
                    if current is not None:
                        session = self.session(rec)
                        phase("detach", check_cancel=False)
                        self.app.stop_proxy(rec)
                        self.sessions.clear(expected=session)
                    elif rec.proxy_owned and rec.proxy_pid:
                        phase("detach", check_cancel=False)
                        self.app.stop_proxy(rec)
                    rec = Record(notebook=self.saved_notebook())
                elif action == "end-session":
                    current = self.sessions.load()
                    session_url = ""
                    if current and (current.kernel_id, current.revision) == (rec.kernel, rec.revision):
                        session_url = current.base_url
                    phase("end-session", check_cancel=False)
                    self.kaggle.cancel(rec.session_id, rec.operation, session_url)
                    if current is not None:
                        if (current.kernel_id, current.revision) == (rec.kernel, rec.revision):
                            self.sessions.clear(expected=current)
                    proxy_error = False
                    if rec.proxy_owned and rec.proxy_pid:
                        try:
                            self.app.stop_proxy(rec)
                        except Exception:
                            proxy_error = True
                    rec = Record(notebook=self.saved_notebook() or rec.notebook)
                    if proxy_error:
                        error_code = "proxy_cleanup_failed"
                else:
                    if provisioning:
                        # Write-ahead intent before network call to Kaggle CreateKernelSession.
                        phase("create")
                        op_name, session_id, created_url = self.kaggle.create(notebook)
                        rec.operation, rec.session_id = op_name, session_id
                        self.save(rec)
                        if not created_url:
                            polled_id, created_url = self.kaggle.poll(op_name, cancel=cancel)
                            if polled_id and not rec.session_id:
                                rec.session_id = polled_id
                                self.save(rec)
                        url = created_url
                    if action == "attach" or provisioning:
                        phase("connect")
                        self.app.connect(url)
                        session = self.sessions.load()
                        if not session:
                            raise LaunchError("Connection was not saved.")
                        rec.kernel, rec.revision = session.kernel_id, session.revision
                        self.save(rec)
                    session = self.session(rec)
                    if action in ("start", "attach"):
                        self.app.prepare(session, phase, recorded)
                        session = self.session(rec)
                        phase("proxy")
                        action_taken, pid, birth = self.app.proxy(session, rec)
                        same_proxy = (pid, birth) == (rec.proxy_pid, rec.proxy_birth)
                        rec.proxy_owned = action_taken in ("started", "switched") or (same_proxy and rec.proxy_owned)
                        rec.proxy_action = action_taken
                        rec.proxy_pid, rec.proxy_birth = pid, birth
                        self.save(rec)
                        if cancel and cancel.is_set():
                            raise LaunchError("Launch cancelled. Cleaning up session.")
                        emit(dict(phase="proxy", outcome="done", error_code=""))
                        rec.phase = "ready"
                    elif action == "refresh":
                        refreshed = self.app.refresh(session, rec)
                        rec.phase, error_code = refreshed if isinstance(refreshed, tuple) else (refreshed, "")
                        if (rec.phase == "unknown" and error_code == "connection_lost"
                                and rec.source == "created" and self.kaggle
                                and self.kaggle.operation_stopped(rec.operation)):
                            error_code = "session_ended"
                    elif action == "stop-worker":
                        if not rec.owned_worker or not session.serve or session.serve.instance != rec.owned_worker:
                            raise LaunchError("Worker ownership could not be confirmed.")
                        phase("stop-worker", check_cancel=False)
                        self.app.stop(session)
                        rec.revision = session.revision
                        rec.worker = rec.owned_worker = ""
                        rec.phase = "worker-stopped"
                    else:
                        raise LaunchError("Unknown launcher action.")
                rec.checked = time.time()
                rec.error_code = error_code
                self.save(rec)
                emit(dict(phase=rec.phase, outcome="done", error_code=error_code))
                return rec
            except Exception as exc:
                failed = "refresh" if action == "refresh" else rec.phase
                cancelled = bool(cancel and cancel.is_set())
                error_code = "cancelled" if cancelled else "failed_" + failed
                code = getattr(exc, "code", None)
                if code in ERROR_CODES:
                    error_code = code
                if action == "attach" and failed == "connect" and self.sessions.load() is None:
                    rec = Record(notebook=rec.notebook, checked=time.time(), error_code=error_code)
                elif provisioning and rec.session_id > 0 and self.kaggle is not None:
                    # Revoke newly created GPU VM immediately on provisioning failure or cancellation.
                    current = self.sessions.load()
                    session_url = current.base_url if current and current.kernel_id == rec.kernel else ""
                    try:
                        emit(dict(phase="end-session", outcome="started", error_code=error_code))
                        self.kaggle.cancel(rec.session_id, rec.operation, session_url)
                        if current is not None and current.kernel_id == rec.kernel:
                            self.sessions.clear(expected=current)
                        if rec.proxy_owned and rec.proxy_pid:
                            try:
                                self.app.stop_proxy(rec)
                            except Exception:
                                pass
                        rec = Record(notebook=rec.notebook, checked=time.time(), error_code=error_code)
                    except Exception:
                        rec.phase = "unknown"
                        rec.error_code = "cancel_unconfirmed"
                elif provisioning and rec.session_id == 0 and failed == "create" and code in (
                    "auth_failed", "notebook_not_found", "quota_exceeded", "wrong_gpu",
                    "sdk_missing", "sdk_version", "metadata_failed", "create_rejected", "cancelled",
                ):
                    # Failed or cancelled cleanly before CreateKernelSession was sent or rejected by server.
                    rec = Record(notebook=rec.notebook, checked=time.time(), error_code=error_code)
                else:
                    rec.phase = "unknown"
                    rec.error_code = error_code
                self.save(rec)
                emit(dict(phase=failed, outcome="failed", error_code=rec.error_code))
                if isinstance(exc, LaunchError):
                    raise
                if error_code == code:
                    raise LaunchError(str(exc)) from None
                if action == "attach" and failed == "connect":
                    raise LaunchError("Could not connect with that Jupyter URL. Check the URL and notebook.") from None
                raise LaunchError(f"Failed during {failed}. Check the notebook and refresh status.") from None
