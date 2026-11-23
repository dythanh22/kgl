"""Run and display a chat completion."""

import json
import sys

from ...client.infer import infer_chat
from ...core.config import Settings
from ...core.errors import KglError, ServeError
from ...core.state import SessionStore
from ...backends import backend_spec
from ...protocol import chat_defaults
from ..common import EXIT_OK, open_runtime, eprint


def cmd_infer(a, st: Settings, store: SessionStore) -> int:
    rt, s = open_runtime(st, store)
    if not s.serve:
        raise KglError("no model server recorded; run `kgl serve start` first")
    spec = backend_spec(s.serve.backend)
    if spec.custom:
        raise ServeError("use POST /handle for this Python handler")
    if not s.serve.instance:
        raise ServeError("server instance is missing; run kgl serve stop and kgl serve start")
    prompt = " ".join(a.prompt) if a.prompt else sys.stdin.read()
    msgs = ([{"role": "system", "content": a.system}] if a.system else []) + \
           [{"role": "user", "content": prompt}]
    defaults = s.serve.defaults
    payload = chat_defaults({"messages": msgs, "stream": not a.no_stream}, defaults)
    payload["max_tokens"] = a.max_tokens if a.max_tokens is not None else payload.get("max_tokens", 256)
    payload["temperature"] = a.temperature if a.temperature is not None else payload.get("temperature", 0.7)
    timeout = a.timeout if a.timeout is not None else defaults.get("timeout", 600)

    def on_chunk(obj: dict) -> None:
        d = (obj.get("choices") or [{}])[0].get("delta", {}).get("content")
        if d and not a.json:
            sys.stdout.write(d)
            sys.stdout.flush()

    try:
        r = infer_chat(rt, s.serve.port, payload, on_chunk if payload["stream"] else None,
                       timeout=timeout,
                       on_cancel=lambda confirmed: eprint(
                           "remote cancellation confirmed" if confirmed
                           else "remote cancellation unconfirmed; the worker may still be running"))
    finally:
        rt.close()
    if r.status != 200:
        raise ServeError(f"inference failed (status {r.status}): {r.error}")
    if spec.passthrough and r.finish_reason == "length" and not r.text and not r.tool_calls:
        raise ServeError("response reached the token limit before an answer; increase --max-tokens")
    if a.json:
        result = {"text": r.text, "latency_s": round(r.latency_s, 3),
                  "ttft_s": r.ttft_s and round(r.ttft_s, 3)}
        if spec.passthrough:
            result.update(reasoning=r.reasoning, finish_reason=r.finish_reason, usage=r.usage, timings=r.timings,
                          tool_calls=r.tool_calls, ttft_first_token_s=r.ttft_first_token_s,
                          ttft_first_content_s=r.ttft_first_content_s)
        print(json.dumps(result))
    else:
        if not payload["stream"]:
            print(r.text)
        else:
            print()
        if a.timing:
            measures = [f"latency {r.latency_s:.2f}s"]
            if r.ttft_first_token_s is not None and spec.passthrough:
                measures.append(f"first token {r.ttft_first_token_s:.2f}s")
            if r.ttft_s is not None:
                measures.append(f"ttft {r.ttft_s:.2f}s")
            for key, label in (("prompt_per_second", "prompt"), ("predicted_per_second", "generation")):
                rate = (r.timings or {}).get(key)
                if isinstance(rate, (float, int)):
                    measures.append(f"{label} {rate:.2f} tok/s")
            eprint("[" + ", ".join(measures) + "]")
    return EXIT_OK
