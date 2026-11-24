"""Extract action items from text with the local Qwen proxy."""
import argparse
from http.client import HTTPException
import json
import os
import sys
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def stream_answer(response):
    data = []
    printed = False
    finished = False
    try:
        for raw in response:
            line = raw.decode("utf-8").rstrip("\r\n")
            if line.startswith("data:"):
                data.append(line[5:].lstrip(" "))
            if line or not data:
                continue
            event = "\n".join(data)
            data.clear()
            if event == "[DONE]":
                if not printed or not finished:
                    raise ValueError("Luồng đã kết thúc nhưng câu trả lời chưa hoàn tất.")
                return
            payload = json.loads(event)
            if "error" in payload:
                error = payload["error"]
                raise ValueError(f"{error.get('type', 'stream_error')}: {error.get('message', '')}")
            for choice in payload.get("choices", []):
                content = choice.get("delta", {}).get("content")
                if content:
                    print(content, end="", flush=True)
                    printed = True
                finished = finished or choice.get("finish_reason") == "stop"
                if choice.get("finish_reason") == "length":
                    raise ValueError("Đã chạm giới hạn token.")
        raise ValueError("Kết nối kết thúc trước khi nhận [DONE].")
    finally:
        if printed:
            print()


def explain_http(error):
    try:
        details = json.load(error).get("error", {})
    except (ValueError, UnicodeError):
        details = {}
    kind = details.get("type")
    message = details.get("message") or error.reason
    if error.code == 401:
        hint = "Kiểm tra KGL_PROXY_KEY."
    elif error.code == 400:
        hint = "Kiểm tra model, nội dung và giới hạn context."
    elif error.code == 503 and kind == "worker_busy":
        delay = error.headers.get("Retry-After", "1")
        hint = f"Worker đang bận. Thử lại sau ít nhất {delay} giây."
    elif error.code == 503:
        hint = "Kiểm tra /ready và kết nối lại Kaggle nếu phiên đã mất."
    else:
        hint = "Kiểm tra proxy và log của worker."
    print(f"HTTP {error.code} ({kind or 'proxy_error'}) · {message}\n{hint}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--model", default="qwen-local")
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()

    notes = sys.stdin.read().strip()
    if not notes:
        parser.error("Cần ghi chú dạng văn bản qua stdin.")
    headers = {"Content-Type": "application/json"}
    if key := os.environ.get("KGL_PROXY_KEY"):
        headers["Authorization"] = "Bearer " + key
    prompt = (
        "Trích xuất các đầu việc được giao trong ghi chú. Mỗi dòng gồm người phụ trách, "
        "việc và hạn. Giữ nguyên dữ kiện và ngôn ngữ của ghi chú. Nếu thiếu hạn, "
        "ghi rõ chưa có hạn.\n\n" + notes
    )
    payload = {"model": args.model, "stream": args.stream, "temperature": 0,
               "max_tokens": 512, "messages": [{"role": "user", "content": prompt}]}
    request = Request(args.base_url.rstrip("/") + "/chat/completions",
                      json.dumps(payload).encode("utf-8"), headers)
    try:
        with urlopen(request, timeout=120) as response:
            if args.stream:
                stream_answer(response)
            else:
                choice = json.load(response)["choices"][0]
                answer = choice["message"]["content"]
                if answer:
                    print(answer)
                if not answer or choice.get("finish_reason") != "stop":
                    raise ValueError("Câu trả lời trống hoặc chưa hoàn tất.")
    except HTTPError as error:
        with error:
            explain_http(error)
        return 1
    except (OSError, HTTPException, ValueError, KeyError, IndexError) as error:
        hint = "Luồng SSE chưa hoàn tất. Phần in ra có thể là câu trả lời dở." if args.stream else "Yêu cầu chưa hoàn tất."
        print(f"{hint} {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
