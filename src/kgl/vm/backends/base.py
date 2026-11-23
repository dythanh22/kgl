"""Hooks shared by model adapters in the VM gateway."""

from contextlib import closing

from ...backends import BACKEND_REGISTRY
from ...protocol import chat_args


class BaseBackend:
    failure = None
    routes = ("/v1/chat/completions", "/internal/chat/completions")

    @property
    def spec(self):
        return BACKEND_REGISTRY[self.name]

    def health(self):
        return {}

    def busy(self):
        return False

    def close(self):
        pass

    def parse_chat(self, request, model_id):
        return request, chat_args(request, model_id)

    def generation_options(self, messages, max_tokens):
        return {}

    def handle(self, handler, req, generation, model_id):
        req, args = self.parse_chat(req, model_id)
        try:
            options = self.generation_options(args[0], args[1])
        except (ValueError, TypeError) as e:
            return handler._json(400, {"error": {"message": str(e)}})
        except Exception as e:
            return handler._json(500, {"error": {"message": f"cannot prepare prompt: {e}"}})
        with closing(self.generate(*args, generation=generation, **options)) as pieces:
            handler._complete(req, pieces, generation)
