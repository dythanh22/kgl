"""Extract action items from text with the local Qwen proxy."""
import argparse
from http.client import HTTPException
import json
import os
import sys
from urllib.error import HTTPError
from urllib.request import Request, urlopen


ITEM_FIELDS = {"owner": {"type": "string"}, "task": {"type": "string"}, "due": {"type": ["string", "null"]}}
ITEMS_FORMAT = {"type": "json_schema", "json_schema": {"name": "action_items", "strict": True, "schema": {
    "type": "object", "properties": {"items": {"type": "array", "items": {
        "type": "object", "properties": ITEM_FIELDS, "required": list(ITEM_FIELDS), "additionalProperties": False,
    }}}, "required": ["items"], "additionalProperties": False,
}}}


def parse_items(answer):
    result = json.loads(answer)
    if not isinstance(result, dict) or set(result) != {"items"} or not isinstance(result["items"], list):
        raise ValueError("Cần JSON có đúng một trường items dạng danh sách.")
    for item in result["items"]:
        if (not isinstance(item, dict) or item.keys() != ITEM_FIELDS.keys()
                or any(not isinstance(item[key], str) or not item[key].strip() for key in ("owner", "task"))
                or (item["due"] is not None and (not isinstance(item["due"], str) or not item["due"].strip()))):
            raise ValueError("Mỗi đầu việc cần owner, task và due hợp lệ.")
    return result


def stream_answer(response, emit):
    data = []
    has_content = False
    finish = None
    for raw in response:
        line = raw.decode("utf-8").rstrip("\r\n")
        if line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
        if line or not data:
            continue
        event = "\n".join(data)
        data.clear()
        if event == "[DONE]":
            if not has_content or finish != "stop":
                raise ValueError("Luồng đã kết thúc nhưng câu trả lời chưa hoàn tất.")
            return
        payload = json.loads(event)
        if "error" in payload:
            error = payload["error"]
            raise ValueError(f"{error.get('type', 'stream_error')}: {error.get('message', '')}")
        for choice in payload.get("choices", []):
            content = choice.get("delta", {}).get("content")
            if content:
                emit(content)
                has_content = has_content or bool(content.strip())
            finish = choice.get("finish_reason") or finish
            if finish == "length":
                raise ValueError("Đã chạm giới hạn token.")
    raise ValueError("Kết nối kết thúc trước khi nhận [DONE].")


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
    parser.add_argument("--json", action="store_true", help="chỉ xuất JSON sau khi xác minh phản hồi hoàn tất")
    args = parser.parse_args()

    notes = sys.stdin.read().strip()
    if not notes:
        parser.error("Cần ghi chú dạng văn bản qua stdin.")
    headers = {"Content-Type": "application/json"}
    if key := os.environ.get("KGL_PROXY_KEY"):
        headers["Authorization"] = "Bearer " + key
    instructions = (
        "Chỉ liệt kê cam kết hoặc đầu việc được giao rõ ràng. Bỏ qua người chỉ hỏi, góp ý, "
        "nêu đề xuất hoặc nói rõ họ không nhận việc. Chép nguyên cụm hạn trong ghi chú, kể cả các từ "
        "như 'trước' hoặc 'chậm nhất'. Giữ ngôn ngữ của ghi chú, không tự bổ sung người hoặc hạn. "
        "Ví dụ: 'An chỉ đề xuất đổi mẫu và không nhận việc' phải bị loại khỏi danh sách."
    )
    output = (
        "Trả JSON với items là danh sách gồm owner, task, due. Dùng null khi không có hạn và items rỗng "
        "khi không có đầu việc."
        if args.json else
        "Mỗi dòng gồm người phụ trách, việc và hạn. Nếu thiếu hạn, ghi rõ chưa có hạn."
    )
    prompt = instructions + "\n" + output + "\n\nGhi chú:\n" + notes
    payload = {"model": args.model, "stream": args.stream,
               "max_tokens": 768, "messages": [{"role": "user", "content": prompt}]}
    if args.json:
        payload.update(response_format=ITEMS_FORMAT, chat_template_kwargs={"enable_thinking": False})
    request = Request(args.base_url.rstrip("/") + "/chat/completions",
                      json.dumps(payload).encode("utf-8"), headers)
    chunks = []
    try:
        with urlopen(request, timeout=120) as response:
            if args.stream:
                emit = chunks.append if args.json else lambda content: print(content, end="", flush=True)
                try:
                    stream_answer(response, emit)
                finally:
                    if not args.json:
                        print()
                answer = "".join(chunks)
            else:
                choice = json.load(response)["choices"][0]
                answer = choice["message"].get("content")
                if not isinstance(answer, str) or not answer.strip() or choice.get("finish_reason") != "stop":
                    if not args.json and answer:
                        print(answer)
                    raise ValueError("Câu trả lời trống hoặc chưa hoàn tất.")
                if not args.json:
                    print(answer)
            if args.json:
                print(json.dumps(parse_items(answer), ensure_ascii=False))
    except HTTPError as error:
        with error:
            explain_http(error)
        return 1
    except (OSError, HTTPException, ValueError, KeyError, IndexError, TypeError, AttributeError) as error:
        hint = ("Luồng SSE chưa hoàn tất. Phần in ra có thể là câu trả lời dở."
                if args.stream and not args.json else "Yêu cầu chưa hoàn tất hoặc kết quả không hợp lệ.")
        print(f"{hint} {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
