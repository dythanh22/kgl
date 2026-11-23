"""Model adapters for the VM gateway."""

from .echo import EchoBackend
from .python import PythonBackend
from .llama import LlamaBackend
from .transformers import TransformersBackend

ADAPTERS = {adapter.name: adapter for adapter in
            (TransformersBackend, EchoBackend, PythonBackend, LlamaBackend)}
