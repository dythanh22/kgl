import time

from .base import BaseBackend


class EchoBackend(BaseBackend):
    """No model/GPU: validates the transport end to end."""

    name = "echo"
    device = "cpu"
    devices = ["cpu"]
    dtype = None

    @classmethod
    def from_args(cls, args):
        return cls()

    def load(self):
        pass

    def generate(self, messages, max_tokens, temperature, top_p, *, generation=None):
        last = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
        for i, w in enumerate(f"echo: {last}".split(" ")):
            if generation and generation.should_stop():
                return
            if i:
                time.sleep(0.005)
            yield w if i == 0 else " " + w
