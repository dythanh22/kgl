# Gọi Qwen từ curl và Python

Sau khi chạy `ready.py` và `proxy.py` trong `myenv`, dùng Base URL `http://127.0.0.1:8001/v1` và model `qwen-local`. Alias được liệt kê ở `/v1/models` khi worker sẵn sàng. Với proxy chạy từ trước khi có alias, khởi động lại tiến trình proxy để nạp mã mới. [Hướng dẫn khởi động](QWEN35B-KAGGLE.md)

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
curl -sS --fail-with-body "${auth[@]}" "$base/ready"
curl -sS --fail-with-body "${auth[@]}" "$base/v1/models"
curl -sS --fail-with-body "${auth[@]}" \
  -H 'Content-Type: application/json' \
  --data-binary @- "$base/v1/chat/completions" <<'JSON'
{
  "model": "qwen-local",
  "messages": [{"role": "user", "content": "Trích xuất người, việc và hạn. Lan gửi báo cáo trước thứ Sáu. Minh đặt phòng trước thứ Ba."}],
  "temperature": 0,
  "max_tokens": 512,
  "stream": false
}
JSON
```

Đọc `choices[0].message.content` và kiểm tra `finish_reason=stop`. Để xem SSE, đổi `stream` thành `true` và thêm `-N` vào lệnh curl gửi chat. Một stream thành công có sự kiện `data: [DONE]`.

## Python · Trích xuất đầu việc từ ghi chú

Từ gốc repo, trong `myenv`:

```bash
python examples/qwen35b-llama/action_items.py <<'NOTES'
Lan gửi báo cáo doanh thu trước thứ Sáu.
Minh xác nhận phòng họp trước thứ Ba.
Cả nhóm đã thống nhất mẫu báo cáo.
NOTES
```

Thêm `--stream` để xem đầu ra dần. Khi chạy lại sau khi đổi VM, giữ nguyên Base URL, alias và ghi chú để đối chiếu. Có thể truyền ghi chú từ tệp bằng `< ghi-chu.txt`. Với một proxy thử, dùng `--base-url http://127.0.0.1:PORT/v1`.

| Tình huống | Cách xử lý |
|---|---|
| `401` | Kiểm tra bearer key của client trùng với proxy. |
| `400` | Kiểm tra model qua `/v1/models`, cấu trúc messages và ngân sách context từ `/ready`. |
| `503 worker_busy` | Đọc `Retry-After`, chờ lượt hiện tại xong rồi gửi lại. |
| Phiên mất kết nối | Xem `/ready`. Nếu cần, chạy lại `kgl connect`, `ready.py` và `proxy.py`. |
| SSE báo lỗi, bị cắt trước `[DONE]` hoặc chạm giới hạn token | Python giữ phần text đã in và trả mã lỗi khác 0. Kiểm tra kết nối hoặc giảm đầu vào trước khi quyết định gửi lại. |
