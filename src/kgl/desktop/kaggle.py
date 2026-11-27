"""Kaggle SDK adapter for interactive notebook session creation and cancellation."""
from importlib.metadata import PackageNotFoundError, version
import time
from types import SimpleNamespace

from ..core.errors import KglError
from ..transport.jupyter import JupyterClient

PINNED_SDK = "0.1.38"
_KERNELS_TYPES = "kagglesdk.kernels.types.kernels_api_service"
_OPS_TYPES = "kagglesdk.common.types.operations_service"


def _req(module_name: str, class_name: str, **fields):
    try:
        obj = getattr(__import__(module_name, fromlist=[class_name]), class_name)()
    except ImportError:
        obj = SimpleNamespace()
    for key, value in fields.items():
        setattr(obj, key, value)
    return obj


class KaggleAdapterError(KglError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def inspect_sdk():
    from kagglesdk.kernels.types.kernels_api_service import (
        ApiCancelKernelSessionRequest, ApiCreateKernelSessionRequest,
        ApiGetKernelSessionStatusRequest,
    )
    if version("kagglesdk") != PINNED_SDK:
        raise ValueError(f"M0 requires kagglesdk {PINNED_SDK}")
    create = ApiCreateKernelSessionRequest()
    create.slug = "example/example"
    create.kernel_type = "notebook"
    create.language = "python"
    create.machine_shape = "NvidiaTeslaT4"
    cancel = ApiCancelKernelSessionRequest()
    cancel.kernel_session_id = 42
    status = ApiGetKernelSessionStatusRequest()
    return {"sdk": version("kagglesdk"), "machine_shape": create.machine_shape,
            "status_by_session_id": hasattr(status, "kernel_session_id"),
            "cancel_by_session_id": cancel.kernel_session_id == 42}


def _classify_sdk_error(exc: Exception, default_code: str, default_msg: str) -> KaggleAdapterError:
    resp = getattr(exc, "response", None)
    status = getattr(resp, "status_code", None)
    text = str(exc).lower()
    if status in (401, 403) or "unauthorized" in text or "forbidden" in text or "unauthenticated" in text:
        return KaggleAdapterError("auth_failed", "Kaggle authentication failed. Check ~/.kaggle/kaggle.json or KAGGLE_API_TOKEN.")
    if status == 404 or "not found" in text:
        return KaggleAdapterError("notebook_not_found", "Notebook not found or inaccessible with current Kaggle credentials.")
    if status == 429 or "quota" in text or "resource_exhausted" in text:
        return KaggleAdapterError("quota_exceeded", "Kaggle GPU quota or active session limit reached.")
    return KaggleAdapterError(default_code, default_msg)


class KaggleSessions:
    """Manage interactive notebook sessions through pinned kagglesdk."""

    def __init__(self, client_factory=None) -> None:
        self._client_factory = client_factory

    @staticmethod
    def available() -> bool:
        try:
            return version("kagglesdk") == PINNED_SDK
        except PackageNotFoundError:
            return False

    def _open_client(self):
        if self._client_factory is not None:
            return self._client_factory()
        try:
            if version("kagglesdk") != PINNED_SDK:
                raise KaggleAdapterError("sdk_version", f"Expected kagglesdk=={PINNED_SDK} in the Python environment.")
            from kagglesdk import KaggleClient
        except (ImportError, PackageNotFoundError):
            raise KaggleAdapterError("sdk_missing", f"Install kagglesdk=={PINNED_SDK} to start GPU sessions automatically.") from None
        client = KaggleClient()
        http = client.http_client()
        http._init_session()
        if not getattr(http, "_signed_in", False):
            raise KaggleAdapterError("auth_failed", "Missing Kaggle credentials (~/.kaggle/kaggle.json or KAGGLE_API_TOKEN).")
        return client

    def create(self, notebook: str) -> tuple[str, int, str]:
        owner, slug = notebook.split("/", 1)
        with self._open_client() as client:
            kclient = client.kernels.kernels_api_client
            greq = _req(_KERNELS_TYPES, "ApiGetKernelRequest", user_name=owner, kernel_slug=slug)
            try:
                meta = kclient.get_kernel(greq).metadata
            except KaggleAdapterError:
                raise
            except Exception as exc:
                raise _classify_sdk_error(exc, "metadata_failed", "Could not read notebook metadata from Kaggle.") from None
            if not meta or not getattr(meta, "enable_gpu", False):
                raise KaggleAdapterError("wrong_gpu", "Enable GPU T4 x 2 on the notebook in Kaggle before launching.")
            creq = _req(
                _KERNELS_TYPES, "ApiCreateKernelSessionRequest",
                slug=meta.ref or notebook,
                kernel_type=str(meta.kernel_type or "notebook"),
                language=str(meta.language or "python"),
                enable_internet=bool(meta.enable_internet),
            )
            if meta.machine_shape:
                creq.machine_shape = str(meta.machine_shape)
            if meta.docker_image:
                creq.docker_image = meta.docker_image
            try:
                op = kclient.create_kernel_session(creq)
            except Exception as exc:
                fallback = ("create_rejected", "Kaggle rejected the session creation request.") if getattr(
                    exc, "response", None) is not None else (
                    "create_uncertain", "Lost response while requesting GPU session. Check Kaggle before retrying.")
                raise _classify_sdk_error(exc, *fallback) from None
            return self._extract_operation(op)

    @staticmethod
    def _extract_operation(op) -> tuple[str, int, str]:
        raw = op.__class__.to_dict(op) if hasattr(op, "to_dict") else dict(op)
        op_name = str(raw.get("name") or getattr(op, "name", "") or "")
        meta = raw.get("metadata") or {}
        sid = meta.get("kernelSessionId")
        if sid is None and "/" in op_name:
            tail = op_name.rsplit("/", 1)[-1]
            if tail.isdigit():
                sid = int(tail)
        session_id = int(sid) if sid is not None else 0
        resp = raw.get("response") or {}
        url = str(resp.get("tokenizedJupyterUrl") or resp.get("jupyterUrl") or "") if raw.get("done") else ""
        return op_name, session_id, url

    def poll(self, op_name: str, *, cancel=None, timeout: float = 300.0, interval: float = 3.0) -> tuple[int, str]:
        deadline = time.monotonic() + timeout
        with self._open_client() as client:
            oclient = client.common.operations_client
            while time.monotonic() < deadline:
                if cancel and cancel.is_set():
                    raise KaggleAdapterError("cancelled", "Launch cancelled while waiting for Kaggle VM.")
                oreq = _req(_OPS_TYPES, "GetOperationRequest", name=op_name)
                try:
                    op = oclient.get_operation(oreq)
                except Exception as exc:
                    raise _classify_sdk_error(exc, "poll_failed", "Lost contact with Kaggle while waiting for VM.") from None
                raw = op.__class__.to_dict(op) if hasattr(op, "to_dict") else dict(op)
                if raw.get("error"):
                    raise KaggleAdapterError("create_failed", "Kaggle failed to start the notebook session.")
                _, session_id, url = self._extract_operation(op)
                if raw.get("done"):
                    if not url:
                        raise KaggleAdapterError("missing_url", "Kaggle started the session without returning a Jupyter URL.")
                    return session_id, url
                time.sleep(min(interval, max(0.05, deadline - time.monotonic())))
        raise KaggleAdapterError("create_timeout", "Timed out waiting for Kaggle to allocate the GPU session.")

    def operation_stopped(self, op_name: str) -> bool:
        if not op_name:
            return False
        try:
            with self._open_client() as client:
                oreq = _req(_OPS_TYPES, "GetOperationRequest", name=op_name)
                op = client.common.operations_client.get_operation(oreq)
                raw = op.__class__.to_dict(op) if hasattr(op, "to_dict") else dict(op)
                return bool(raw.get("done") and raw.get("error"))
        except Exception:
            return False

    def cancel(self, session_id: int, op_name: str = "", session_url: str = "", *, timeout: float = 25.0) -> None:
        deadline = time.monotonic() + timeout
        with self._open_client() as client:
            kclient = client.kernels.kernels_api_client
            oclient = client.common.operations_client
            req = _req(_KERNELS_TYPES, "ApiCancelKernelSessionRequest", kernel_session_id=int(session_id))
            try:
                resp = kclient.cancel_kernel_session(req)
            except Exception as exc:
                raise _classify_sdk_error(exc, "cancel_unconfirmed", "Could not confirm GPU session cancellation on Kaggle.") from None
            if getattr(resp, "error_message", ""):
                raise KaggleAdapterError("cancel_unconfirmed", "Kaggle reported an error while cancelling the GPU session.")
            while time.monotonic() < deadline:
                op_stopped = not op_name
                if op_name:
                    try:
                        oreq = _req(_OPS_TYPES, "GetOperationRequest", name=op_name)
                        op = oclient.get_operation(oreq)
                        raw = op.__class__.to_dict(op) if hasattr(op, "to_dict") else dict(op)
                        op_stopped = bool(raw.get("done") and raw.get("error"))
                    except Exception:
                        op_stopped = False
                jupyter_stopped = True
                if session_url:
                    try:
                        JupyterClient(session_url, timeout=4.0).list_kernels()
                        jupyter_stopped = False
                    except KglError:
                        jupyter_stopped = True
                if op_stopped and jupyter_stopped:
                    return
                time.sleep(min(2.0, max(0.05, deadline - time.monotonic())))
        raise KaggleAdapterError("cancel_unconfirmed", "Cancellation requested, but VM shutdown was not yet confirmed.")

    def quota(self) -> dict | None:
        try:
            with self._open_client() as client:
                resp = client.kernels.kernels_api_client.get_accelerator_quota_statistics(
                    _req(_KERNELS_TYPES, "ApiGetAcceleratorQuotaStatisticsRequest")
                )
                gpu = resp.gpu_quota
                if not gpu:
                    return None
                sec = lambda v: float(v.total_seconds()) if hasattr(v, "total_seconds") else float(v or 0)
                used, reserved, total = sec(gpu.time_used), sec(gpu.time_reserved), sec(gpu.total_time_allowed)
                return {"used_s": used, "reserved_s": reserved, "total_s": total,
                        "remaining_s": max(0.0, total - used - reserved)}
        except Exception:
            return None
