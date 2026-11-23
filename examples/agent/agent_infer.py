"""Run after kgl serve start and kgl serve proxy, using the same KGL_HOME."""
import argparse
import json
import os
import subprocess
import sys
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description="Return one JSON result from a ready kgl proxy")
    parser.add_argument("--proxy", default="http://127.0.0.1:8001")
    parser.add_argument("prompt")
    args = parser.parse_args()
    status = subprocess.run(["kgl", "status", "--json"], capture_output=True, text=True)
    if status.returncode:
        print(status.stderr or "Session unavailable; run kgl status", file=sys.stderr)
        return status.returncode
    if json.loads(status.stdout)["connection"] != "ok":
        raise RuntimeError("Session unavailable")
    headers = {"Content-Type": "application/json"}
    if os.environ.get("KGL_PROXY_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["KGL_PROXY_KEY"]
    with urlopen(Request(args.proxy + "/ready", headers=headers), timeout=10) as ready:
        model = json.load(ready)["health"]["model"]
    body = {"model": model, "messages": [{"role": "user", "content": args.prompt}], "stream": True}
    request = Request(args.proxy + "/v1/chat/completions", json.dumps(body).encode(), headers)
    pieces = []
    with urlopen(request, timeout=60) as stream:
        for line in stream:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                print(json.dumps({"model": model, "text": "".join(pieces)}, ensure_ascii=False))
                return 0
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(event["error"]["message"])
            pieces.append(event["choices"][0]["delta"].get("content") or "")
    raise RuntimeError("Incomplete stream; execution outcome unknown")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(3)
