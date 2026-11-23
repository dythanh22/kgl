from collections import deque
import threading
import time


class Generation:
    def __init__(self, request_id, timeout):
        self.id = request_id
        self.deadline = time.monotonic() + timeout
        self.cancelled = threading.Event()
        self.finished = threading.Event()
        self.worker_stopped = True
        self.finish_reason = "stop"
        self.usage = None

    def should_stop(self):
        return self.cancelled.is_set() or time.monotonic() >= self.deadline


class State:
    def __init__(self):
        self.status = "loading"
        self.error = None
        self.lock = threading.Lock()
        self.active = None
        self.recent = deque(maxlen=128)
