"""Load a notebook-owned Python module implementing load and handle."""
import importlib.util
import json
from pathlib import Path
import sys

from .base import BaseBackend


class PythonBackend(BaseBackend):
    name = "python"
    routes = ("/internal/handle",)
    dtype = None
    device = None
    devices = ()

    def __init__(self, model, handler, require_gpu):
        self.model_path = model
        self.handler = handler
        self.require_gpu = require_gpu

    @classmethod
    def from_args(cls, args):
        if not args.handler:
            raise ValueError("python requires --handler")
        return cls(args.model, args.handler, args.require_gpu)

    def load(self):
        try:
            import torch
        except ImportError:
            torch = None
        cuda = torch is not None and torch.cuda.is_available()
        if self.require_gpu and not cuda:
            raise RuntimeError("CUDA is required but unavailable")
        self.device = "cuda:0" if cuda else "cpu"
        self.devices = [self.device]
        path = Path(self.handler)
        spec = importlib.util.spec_from_file_location("kgl_user_handler", path)
        if spec is None or spec.loader is None:
            raise ValueError("--handler must be a Python module")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        sys.path.insert(0, str(path.parent))
        spec.loader.exec_module(module)
        self.invoke = module.handle
        self.model = module.load(self.model_path, self.device)

    def handle(self, handler, request, generation, model_id):
        try:
            result = self.invoke(self.model, request)
        except ValueError as exc:
            return handler._json(400, {"error": {"message": str(exc)}})
        except Exception as exc:
            return handler._json(500, {"error": {"message": f"handler failed: {exc}"}})
        try:
            size = len(json.dumps(result, allow_nan=False).encode())
            if size > handler.request_limit:
                raise ValueError("handler output exceeds the byte limit")
        except (ValueError, TypeError, RecursionError) as exc:
            return handler._json(500, {"error": {"message": str(exc)}})
        handler._json(200, result)
