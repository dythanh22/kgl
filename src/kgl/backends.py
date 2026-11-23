"""Backend registry and launch options shared by the CLI and VM."""
from dataclasses import dataclass
import json
import posixpath
import re
from typing import Callable

from .protocol import gpu_ready


def parse_device_map(value):
    if value in (None, "balanced"):
        return value
    try:
        placement = json.loads(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("device map must be 'balanced' or a JSON object of module names and GPU indices") from exc
    if (not isinstance(placement, dict) or not placement
            or any(type(index) is not int or index < 0 for index in placement.values())
            or len(set(placement.values())) < 2):
        raise ValueError("device map must assign modules to at least two CUDA GPU indices")
    return placement


def absolute(value):
    if not isinstance(value, str) or not posixpath.isabs(value):
        raise ValueError("expected an absolute VM path")


def positive(value):
    if type(value) is not int or value <= 0:
        raise ValueError("expected a positive integer")


def object_value(value):
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")


def weights(value):
    if (not re.fullmatch(r"(?:[0-9]+(?:\.[0-9]+)?)(?:,[0-9]+(?:\.[0-9]+)?)+", value)
            or not any(float(weight) > 0 for weight in value.split(","))):
        raise ValueError("--tensor-split requires comma-separated GPU weights")


@dataclass(frozen=True)
class Option:
    flag: str
    type: Callable = str
    validate: Callable | None = None
    public: bool = True


OPTIONS = {
    "handler": Option("--handler", validate=absolute),
    "device_map": Option("--device-map", validate=parse_device_map),
    "prefill_chunk_size": Option("--prefill-chunk-size", int, positive),
    "llama_manifest": Option("--llama-manifest", validate=absolute),
    "ctx": Option("--ctx", int, positive),
    "tensor_split": Option("--tensor-split", validate=weights),
    "settings": Option("--settings", json.loads, object_value, False),
}

DEFAULT_BACKEND = "transformers"
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@dataclass(frozen=True)
class BackendSpec:
    name: str
    default_model: str = DEFAULT_MODEL
    fixed_model: str | None = None
    options: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    gpu_supported: bool = True
    gpu_required_for_proxy: bool = True
    custom: bool = False
    passthrough: bool = False
    installable: bool = False

    def model_id(self, model: str) -> str:
        return self.fixed_model or model

    def validate_options(self, options, require_gpu=False):
        if require_gpu and not self.gpu_supported:
            raise ValueError("--require-gpu requires a model backend")
        for name, value in options.items():
            if name not in self.options:
                raise ValueError(f"{name} is unavailable for backend {self.name}")
            validate = OPTIONS[name].validate
            if validate:
                validate(value)

    def proxy_ready(self, health: dict) -> bool:
        return (not self.gpu_required_for_proxy
                or (self.passthrough and health.get("require_gpu") is False)
                or gpu_ready(health))


BACKEND_REGISTRY = {
    spec.name: spec for spec in (
        BackendSpec(DEFAULT_BACKEND, options=("device_map", "prefill_chunk_size"), capabilities=("usage",)),
        BackendSpec("echo", fixed_model="echo", gpu_supported=False, gpu_required_for_proxy=False),
        BackendSpec("python", default_model="",
                    options=("handler",), custom=True, gpu_required_for_proxy=False),
        BackendSpec("llama", options=("llama_manifest", "ctx", "tensor_split", "settings"),
                    passthrough=True, installable=True),
    )
}
BACKENDS = tuple(BACKEND_REGISTRY)
DOCTOR_BACKENDS = tuple(spec.name for spec in sorted(BACKEND_REGISTRY.values(), key=lambda spec: spec.gpu_supported)
                        if not spec.custom)
INSTALLABLE_BACKENDS = tuple(spec.name for spec in BACKEND_REGISTRY.values() if spec.installable)
DEFAULT_GGUF_BACKEND = "llama"
_UNKNOWN_BACKEND = BackendSpec("")


def backend_spec(name: str | None) -> BackendSpec:
    return BACKEND_REGISTRY.get(name, _UNKNOWN_BACKEND)
