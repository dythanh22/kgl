"""Run the Qwen evaluation through the local chat completions API."""
import argparse
from datetime import datetime, timezone
from http.client import HTTPException
import json
import os
from pathlib import Path
import re
import sys
import time
from urllib.error import HTTPError
from urllib.request import Request, build_opener, ProxyHandler


HERE = Path(__file__).resolve().parent
PROFILES = {
    "gui": {"stream": True, "stream_options": {"include_usage": True}, "max_tokens": 768},
    "api": {"stream": False},
    "script": {"stream": False, "temperature": 0, "max_tokens": 512},
    "perf": {"stream": True, "stream_options": {"include_usage": True}},
}


def check(case, result):
    if result.get("error"):
        return "transport"
    if result.get("finish_reason") == "length":
        return "length"
    answer = result["content"].strip()
    if result.get("finish_reason") != "stop" or not answer or not result["done"]:
        return "incomplete"
    if "exact" in case and answer != case["exact"]:
        return "content"
    for key, expected in (("required", True), ("forbidden", False)):
        for term in case.get(key, []):
            found = re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", answer, re.IGNORECASE)
            if bool(found) != expected:
                return "content"
    return "exact_pass" if "exact" in case else "needs_review"


def sse_events(response):
    data = []
    for raw in response:
        line = raw.decode("utf-8").rstrip("\r\n")
        if line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
        elif not line and data:
            yield "\n".join(data)
            data.clear()


def complete(opener, base, headers, payload):
    body = json.dumps(payload, ensure_ascii=False).encode()
    request = Request(base + "/v1/chat/completions", body, headers)
    streaming = payload["stream"]
    result = {"request_bytes": len(body), "content": "", "done": False}
    start = time.monotonic()
    try:
        with opener.open(request, timeout=120) as response:
            events = sse_events(response) if streaming else [response.read()]
            for event in events:
                elapsed = round(time.monotonic() - start, 4)
                if streaming:
                    result.setdefault("first_event_s", elapsed)
                if event == "[DONE]":
                    result["done"] = True
                    break
                data = json.loads(event)
                if data.get("error"):
                    result["error"] = data["error"].get("type", "upstream_error")
                    break
                for key in ("usage", "timings"):
                    if data.get(key):
                        result[key] = data[key]
                for choice in data.get("choices", []):
                    message = choice.get("delta" if streaming else "message", {})
                    if streaming and message.get("reasoning_content"):
                        result.setdefault("first_reasoning_s", elapsed)
                    if content := message.get("content"):
                        if streaming and content.strip():
                            result.setdefault("first_content_s", elapsed)
                        result["content"] += content
                    if reason := choice.get("finish_reason"):
                        result["finish_reason"] = reason
            if not streaming and "error" not in result:
                result["done"] = True
    except HTTPError as exc:
        with exc:
            result["error"] = f"http_{exc.code}"
            result["retry_after"] = exc.headers.get("Retry-After")
    except (OSError, HTTPException, ValueError) as exc:
        result["error"] = type(exc).__name__
    result["total_s"] = round(time.monotonic() - start, 4)
    return result


def performance_cases(short_output=False):
    # Use distinct prefixes across both sets to avoid prompt-cache reuse.
    editions = ("Oak", "Elm", "Pine") if short_output else ("Cedar", "Birch", "Maple")
    for edition in editions:
        entries = "\n".join(
            f"Record {i:02d}: Team {i % 7} reported delivery of {20 + i} cartons. "
            f"Inspection noted {i % 4} crushed cartons and assigned supervisor {i % 5} "
            "to call the supplier after the afternoon inventory. No replacement was approved."
            for i in range(1, 45)
        )
        if short_output:
            instruction = ("Write a 100–130 word English handover memo with three short paragraphs "
                           "covering delivery, inspection, and approval status. "
                           "The heading must be the register edition. Do not list individual records.")
        else:
            instruction = (f"Draft a 340–380 word handover memo in English headed {edition}. "
                           "Summarize the delivery, inspection, and approval status. Use three sections. "
                           "Then include six short practical follow-up suggestions marked as recommendations. "
                           "Avoid claiming any recommendation was approved.")
        yield {"id": f"P-{edition}", "required": [edition], "prompt":
               f"{edition} warehouse daily register.\n{entries}\n{instruction}"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=PROFILES)
    parser.add_argument("--origin", default="http://127.0.0.1:8001")
    parser.add_argument("--case", action="append", help="Run only the selected case IDs.")
    parser.add_argument("--short-output", action="store_true",
                        help="Use shorter memo prompts in the perf profile.")
    parser.add_argument("--thinking", choices=("on", "off"), help="Override the server's thinking mode.")
    parser.add_argument("--max-tokens", type=int, help="Override the profile's generation limit.")
    args = parser.parse_args()
    if args.max_tokens is not None and args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    if args.short_output and args.profile != "perf":
        parser.error("--short-output requires the perf profile")
    cases = list(performance_cases(args.short_output)) if args.profile == "perf" else json.loads(
        (HERE / "cases.json").read_text(encoding="utf-8"))
    if args.case:
        if set(args.case) - {case["id"] for case in cases}:
            parser.error("unknown case ID")
        cases = [case for case in cases if case["id"] in args.case]
    headers = {"Content-Type": "application/json"}
    if key := os.environ.get("KGL_PROXY_KEY"):
        headers["Authorization"] = "Bearer " + key
    opener = build_opener(ProxyHandler({}))
    base = args.origin.rstrip("/")
    with opener.open(Request(base + "/ready", headers=headers), timeout=20) as response:
        health = json.load(response)["health"]
    settings = dict(PROFILES[args.profile])
    if args.short_output:
        settings["max_tokens"] = 1024
    if args.max_tokens is not None:
        settings["max_tokens"] = args.max_tokens
    if args.thinking:
        settings["chat_template_kwargs"] = {"enable_thinking": args.thinking == "on"}
    print(json.dumps({"type": "run", "timestamp": datetime.now(timezone.utc).isoformat(),
                      "profile": args.profile, "settings": settings, "short_output": args.short_output,
                      "health": {key: health.get(key) for key in (
                          "instance", "model", "n_ctx", "devices", "defaults", "gpu_memory")}}), flush=True)
    failures = 0
    for case in cases:
        request = {"messages": [*case.get("history", []),
                                {"role": "user", "content": case["prompt"]}], **settings}
        if args.profile != "gui":
            request["model"] = "qwen-local"
        result = complete(opener, base, headers, request)
        result["check"] = check(case, result)
        failures += result["check"] in {"transport", "incomplete", "length", "content"}
        print(json.dumps({"id": case["id"], **result}, ensure_ascii=False), flush=True)
    return int(bool(failures))


if __name__ == "__main__":
    sys.exit(main())
