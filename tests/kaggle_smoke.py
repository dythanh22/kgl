"""Explicit live GPU acceptance test. Run with python -m tests.kaggle_smoke."""
from dataclasses import replace
import json
import socket
import statistics
import threading
import time
from urllib.request import Request, urlopen
import uuid

from kgl.config import Settings
from kgl.errors import ConnectionLost
from kgl.runtime import Runtime
from kgl.serving import ServeManager, infer_chat, make_proxy_server
from kgl.state import SessionStore


def exercise_proxy(base_url: str):
    elapsed = []
    for i in range(20):
        stream = bool(i % 2)
        body = {"messages": [{"role": "user", "content": "Reply with one word: OK"}],
                "max_tokens": 8, "temperature": 0, "stream": stream}
        request = Request(base_url + "/v1/chat/completions", json.dumps(body).encode(),
                          {"Content-Type": "application/json"})
        start = time.monotonic()
        with urlopen(request, timeout=45) as response:
            if stream:
                events = [line[6:].strip() for line in response if line.startswith(b"data: ")]
                assert events and events[-1] == b"[DONE]" and events.count(b"[DONE]") == 1
                assert any(json.loads(e)["choices"][0]["delta"].get("content") for e in events[:-1])
            else:
                assert json.load(response)["choices"][0]["message"]["content"].strip()
        elapsed.append(time.monotonic() - start)
    return {"requests": len(elapsed), "p50_s": statistics.median(elapsed),
            "p95_s": sorted(elapsed)[18]}


def main():
    settings = Settings.from_env()
    session = SessionStore(settings.home).load()
    if session is None:
        raise SystemExit("Set KGL_HOME to the connected session or run kgl connect first.")
    settings = replace(settings, workdir=settings.workdir + "/kgl-validation-" + uuid.uuid4().hex)
    rt = Runtime(replace(session, serve=None), settings)
    model = "Qwen/Qwen2.5-0.5B-Instruct"
    try:
        port = rt.call("import socket\nwith socket.socket() as sock:\n"
                       "    sock.bind(('127.0.0.1', 0))\n    _emit({'port': sock.getsockname()[1]})")["port"]
        manager = ServeManager(rt, port)
        instance = None
        try:
            health = manager.start(model, wait_timeout=180)["health"]
            instance = health["instance"]
            assert health["device"].startswith("cuda"), health
            assert manager.start(model, wait_timeout=15)["health"]["instance"] == instance
            rt.session.serve = dict(instance=instance, backend="transformers", model=model, port=port)
            with make_proxy_server(rt, port, model, listen_port=0) as proxy:
                thread = threading.Thread(target=proxy.serve_forever, daemon=True)
                thread.start()
                try:
                    base = f"http://127.0.0.1:{proxy.server_port}"
                    with urlopen(base + "/ready", timeout=10) as response:
                        assert json.load(response)["health"]["instance"] == instance
                    metrics = exercise_proxy(base)
                finally:
                    proxy.shutdown()
                    thread.join()

            interrupted = False
            def disconnect(chunk):
                nonlocal interrupted
                if not interrupted and chunk["choices"][0]["delta"].get("content"):
                    interrupted = True
                    rt._channel.ws._sock.shutdown(socket.SHUT_RDWR)

            payload = {"messages": [{"role": "user", "content": "Count from one to fifty."}],
                       "max_tokens": 64, "stream": True}
            try:
                infer_chat(rt, port, payload, disconnect, timeout=30)
            except ConnectionLost as exc:
                assert interrupted and exc.sent
            else:
                raise AssertionError("expected the in-flight stream to fail")
            reply = infer_chat(rt, port, payload, timeout=45)
            assert reply.status == 200 and reply.text
        finally:
            stopped = manager.stop(instance)
        print(json.dumps(dict(health=health, metrics=metrics, recovered=True, cleanup=stopped), indent=2))
    finally:
        rt.close()


if __name__ == "__main__":
    main()
