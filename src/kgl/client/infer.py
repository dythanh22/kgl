"""Collect JSON and SSE inference results over the kernel transport."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import logging
import time
from typing import Callable
import uuid

from ..core.errors import ConnectionLost, KglError, ProtocolError, ServeError
from ..remote import CANCEL_INFER, INFER, MarkerParser, new_marker, render
from ..transport.runtime import Runtime

log = logging.getLogger(__name__)


@dataclass
class InferResult:
    status: int = 0
    body: dict | list | str | int | float | bool | None = None
    text: str = ""
    error: str | None = None
    latency_s: float = 0.0
    ttft_s: float | None = None  # existing field, time to first streamed content
    ttft_first_token_s: float | None = None
    ttft_first_content_s: float | None = None
    reasoning: str = ""
    finish_reason: str | None = None
    usage: dict | None = None
    timings: dict | None = None
    tool_calls: list | None = None


def infer_chat(rt: Runtime, port: int, payload: dict,
               on_chunk: Callable[[dict], None] | None = None,
               timeout: float = 600.0, *, expected_instance: str | None = None,
               check_client=None, on_cancel: Callable[[bool], None] | None = None,
               custom: bool = False) -> InferResult:
    """Send an inference request to the VM server via the kernel."""
    res = InferResult()
    t0 = time.monotonic()
    pieces: list[str] = []
    reasoning: list[str] = []
    instance = expected_instance or (rt.session.serve.instance if rt.session.serve else None)
    if custom and not instance:
        raise ServeError("server instance is missing; restart the model server")
    request_id = uuid.uuid4().hex
    got_status = False
    submitted = False

    def mark_submitted() -> None:
        nonlocal submitted
        submitted = True

    def handle(m: dict) -> None:
        nonlocal got_status
        if "chunk" in m:
            if m["chunk"] == "[DONE]":
                return
            try:
                obj = json.loads(m["chunk"])
            except ValueError:
                return
            choice = (obj.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            content = delta.get("content")
            thought = delta.get("reasoning_content")
            tools = delta.get("tool_calls")
            if (content or thought or tools) and res.ttft_first_token_s is None:
                res.ttft_first_token_s = time.monotonic() - t0
            if content:
                if res.ttft_first_content_s is None:
                    res.ttft_s = res.ttft_first_content_s = time.monotonic() - t0
                pieces.append(content)
            if thought:
                reasoning.append(thought)
            if tools:
                if res.tool_calls is None:
                    res.tool_calls = []
                res.tool_calls.extend(tools)
            res.finish_reason = choice.get("finish_reason") or res.finish_reason
            res.usage = obj.get("usage") or res.usage
            res.timings = obj.get("timings") or res.timings
            if on_chunk:
                on_chunk(obj)
        if "status" in m:
            got_status = True
            res.status = m["status"]
            res.body = m.get("body")
            res.error = m.get("error")

    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    limit = rt.settings.request_bytes
    if len(data) > limit:
        raise ServeError(f"inference request exceeds {limit // 1024} KiB" if limit % 1024 == 0
                         else f"inference request exceeds {limit} bytes")
    marker = new_marker()
    parser = MarkerParser(handle, marker=marker, retain=False)
    code = render(INFER, marker=marker, PAYLOAD=base64.b64encode(data).decode(),
                  PORT=port, TIMEOUT=timeout,
                  OPTIONS={"instance": instance, "request_id": request_id, "custom": custom})
    try:
        rt.run(code, lambda name, text: parser.feed(text) if name == "stdout" else None,
               total_timeout=timeout, check=True, check_client=check_client, on_submit=mark_submitted,
               capture_output=False)
    except (KeyboardInterrupt, TimeoutError, ConnectionLost, OSError):
        if not submitted or got_status:
            raise
        control = rt.control_runtime() if instance else None
        confirmed = False
        try:
            if control:
                confirmed = control.call(CANCEL_INFER, timeout=4, PORT=port, INSTANCE=instance,
                                         REQUEST_ID=request_id).get("cancelled") is True
            elif instance and (rt.session.dedicated_kernel or rt.settings.interrupt_infer_kernel):
                deadline = time.monotonic() + 4
                rt.client.interrupt_kernel(rt.session.kernel_id, timeout=2)
                confirmed = rt.call(CANCEL_INFER, timeout=max(0.01, deadline - time.monotonic()),
                                    PORT=port, INSTANCE=instance,
                                    REQUEST_ID=request_id).get("cancelled") is True
        except (KglError, TimeoutError):
            pass
        finally:
            if control:
                control.close()
        if on_cancel:
            on_cancel(confirmed)
        elif confirmed:
            log.info("remote cancellation confirmed")
        else:
            log.warning("remote cancellation unconfirmed")
        raise
    if not got_status:
        raise ProtocolError("inference ended without a result marker")
    res.latency_s = time.monotonic() - t0
    if custom:
        return res
    if res.body:
        choice = (res.body.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        res.text = message.get("content") or ""
        res.reasoning = message.get("reasoning_content") or ""
        res.tool_calls = message.get("tool_calls")
        res.finish_reason = choice.get("finish_reason")
        res.usage = res.body.get("usage")
        res.timings = res.body.get("timings")
    else:
        res.text = "".join(pieces)
        res.reasoning = "".join(reasoning)
    return res
