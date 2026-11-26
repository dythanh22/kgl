# Gọi Qwen từ curl và Python

Sau khi chạy `ready.py` và `proxy.py` trong `myenv`, dùng Base URL `http://127.0.0.1:8001/v1` và model `qwen-local`. Alias trong preset được liệt kê ở `/v1/models` khi worker đúng model, quant và instance đã sẵn sàng. API cũng chấp nhận ID chính tắc. Với session cũ thiếu alias, khởi động lại worker theo hướng dẫn Qwen khi có thể gián đoạn phục vụ. Với proxy chạy từ trước khi có alias, khởi động lại tiến trình proxy để nạp mã mới. [Hướng dẫn khởi động](QWEN35B-KAGGLE.md)

## Xác thực

Proxy mặc định chỉ nghe trên loopback. Để dùng bearer key, đặt `KGL_PROXY_KEY` trong môi trường của proxy **trước khi khởi động**, rồi dùng cùng giá trị ở client. Trong Bash, có thể nhập key mà không hiện lên màn hình:

```bash
read -rsp 'Proxy key: ' KGL_PROXY_KEY
printf '\n'
export KGL_PROXY_KEY
```

Giao diện chat hỏi key khi nhận 401. Python đọc key từ biến môi trường. Các lệnh curl bên dưới dùng mảng header để hỗ trợ cả proxy có key và proxy local không yêu cầu key.

## curl · Kiểm tra kết nối và gửi một tác vụ

```bash
base=http://127.0.0.1:8001
auth=()
if [[ -n ${KGL_PROXY_KEY:-} ]]; then
  auth=(-H "Authorization: Bearer $KGL_PROXY_KEY")
fi
curl -sS --fail-with-body --max-time 10 "${auth[@]}" "$base/ready"
curl -sS --fail-with-body --max-time 10 "${auth[@]}" "$base/v1/models"
curl -sS --fail-with-body --max-time 120 "${auth[@]}" \
  -H 'Content-Type: application/json' \
  --data-binary @- "$base/v1/chat/completions" <<'JSON'
{
  "model": "qwen-local",
  "messages": [{"role": "user", "content": "Trích xuất người, việc và hạn. Lan gửi báo cáo trước thứ Sáu. Minh đặt phòng trước thứ Ba."}],
  "chat_template_kwargs": {"enable_thinking": false},
  "max_tokens": 768,
  "stream": false
}
JSON
```

Đọc `choices[0].message.content` và kiểm tra `finish_reason=stop`. Để xem SSE, đổi `stream` thành `true` và thêm `-N` vào lệnh curl gửi chat. Một stream thành công có sự kiện `data: [DONE]`.

## Tùy chọn theo khả năng của worker

Xem `health.capabilities` trong `/ready` trước khi gửi các trường dưới đây. Proxy và gateway trả 400 nếu worker chưa hỗ trợ hoặc giá trị sai. Bỏ các trường mới để dùng hành vi mặc định của worker. Preset Qwen dùng `temperature=0.6`, `top_p=0.95` và `top_k=20` khi caller bỏ qua các giá trị đó.

| Trường | Giá trị được hỗ trợ |
|---|---|
| `chat_template_kwargs` | Object có đúng khóa `enable_thinking` kiểu boolean. `false` tắt sinh reasoning cho request. |
| `presence_penalty` | Số trong khoảng −2 đến 2. 0 là giá trị trung tính. |
| `repeat_penalty` | Hệ số dương hữu hạn trong miền float32 mà API chấp nhận. 1 là giá trị trung tính. |
| `response_format` | `{"type":"json_object"}` hoặc `{"type":"json_schema","json_schema":{"name":"result","schema":{...}}}` khi có thêm capability `json_schema`. `strict` là boolean tùy chọn trong wrapper. Các tham chiếu schema dùng JSON pointer cục bộ. |

Giới hạn sinh áp dụng cho cả reasoning và câu trả lời. Chọn một trong `max_tokens` hoặc `max_completion_tokens`. JSON Schema kiểm hình dạng câu trả lời. Đối chiếu người, việc và hạn với ghi chú trước khi dùng kết quả để tự động hóa quyết định. [Hợp đồng llama.cpp tại bản ghim](https://github.com/ggml-org/llama.cpp/blob/f4f6f2e7db76e98c95cc9b7fc608ea669c806044/tools/server/README.md#post-v1chatcompletions-openai-compatible-chat-completions-api)

## Python · Trích xuất đầu việc từ ghi chú

Từ gốc repo, trong `myenv`:

```bash
python examples/qwen35b-llama/action_items.py <<'NOTES'
Lan gửi báo cáo doanh thu trước thứ Sáu.
Minh xác nhận phòng họp.
Quân chỉ góp ý dùng lại mẫu cũ và không nhận việc.
NOTES
```

Thêm `--stream` để xem văn bản dần. Ví dụ JSON dùng cùng dữ kiện:

```bash
python examples/qwen35b-llama/action_items.py --json --stream <<'NOTES'
Lan gửi báo cáo doanh thu trước thứ Sáu.
Minh xác nhận phòng họp.
Quân chỉ góp ý dùng lại mẫu cũ và không nhận việc.
NOTES
```

Kết quả mong đợi có hai người được giao, gồm hạn “trước thứ Sáu” của Lan và `null` cho Minh. Đối chiếu từng trường với ghi chú trước khi dùng kết quả. Chẳng hạn, “thứ Sáu” và “trước thứ Sáu” là hai hạn khác nhau. [Kết quả chạy lại ví dụ](../../docs/reports/2026-11-25-qwen-l9-l10.md)

Dùng `--json` cho phản hồi JSON hoặc `--json --stream` để nhận qua SSE. Cả hai chế độ chỉ ghi stdout sau khi xác nhận phản hồi hoàn tất và kiểm cấu trúc `{"items":[{"owner":"…","task":"…","due":null}]}`. Giới hạn sinh của script là 768 token, timeout đọc HTTP là 120 giây. Proxy áp dụng giới hạn request mặc định 512 KiB. Khi chạy lại sau khi đổi VM, giữ nguyên Base URL, alias và ghi chú để đối chiếu. Có thể truyền ghi chú từ tệp bằng `< ghi-chu.txt`. Với một proxy thử, dùng `--base-url http://127.0.0.1:PORT/v1`.

| Tình huống | Cách xử lý |
|---|---|
| `401` | Kiểm tra bearer key của client trùng với proxy. |
| `400` | Kiểm tra model qua `/v1/models`, cấu trúc messages, capability và ngân sách context từ `/ready`. |
| `413` | Request vượt giới hạn byte của proxy. Gửi ít dữ liệu hoặc ít lượt lịch sử hơn. |
| `503 worker_busy` | Đọc `Retry-After`, chờ lượt hiện tại xong rồi gửi lại. |
| Phiên mất kết nối | Xem `/ready`. Nếu cần, chạy lại `kgl connect`, `ready.py` và `proxy.py`. |
| SSE báo lỗi, bị cắt trước `[DONE]` hoặc chạm giới hạn token | Python trả mã lỗi khác 0. Chế độ văn bản giữ phần đã in, còn chế độ JSON để stdout trống. Kiểm tra kết nối hoặc giảm đầu vào trước khi quyết định gửi lại. |
