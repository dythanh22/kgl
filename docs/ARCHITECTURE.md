# Kiến trúc

```
Agent / app / browser / terminal
        │ HTTP (chat / predict)                kgl exec / infer / upload ...
        ▼                                              │
  kgl serve proxy (127.0.0.1)  ───────────────►  cli.py
        │                                              │
        └──────────────► serving.py / files.py ◄───────┘
                               │
                          runtime.py   (session + lazy reconnect + retry policy)
                               │
                          jupyter.py   (REST + kernel channel, execute_request)
                               │
                            ws.py      (minimal RFC 6455 client, ws/wss)
                               │   Jupyter messaging over WebSocket
 ══════════════════════════════╪═══════════ Kaggle notebook proxy ═══
                               ▼
                    Kaggle VM: ipykernel
                         │  snippets from remote.py
                         ├─ upload/download/stat/probe (stdlib)
                         └─ subprocess: remote_server.py  (127.0.0.1:8000, not exposed)
                                         └─ transformers | image-classification | echo
```

## Module

| Module | Vai trò |
|---|---|
| `ws.py` | WebSocket client stdlib; buffer giữ byte của frame đang đọc, giữ các fragment qua timeout; ping/pong; wss |
| `jupyter.py` | `Endpoint` (giữ `?token=` của URL), REST kernels, `KernelChannel.execute` (idle/total timeout, keep-alive ping) |
| `runtime.py` | Giữ channel, kết nối lười, chính sách retry (xem dưới) |
| `remote.py` | Mã chạy trên VM và marker có nonce riêng cho từng lượt gọi nội bộ |
| `files.py` | Truyền chunk base64 vào file tạm riêng, kiểm tra SHA-256 rồi thay file đích nguyên tử |
| `remote_server.py` | Worker chat hoặc phân loại ảnh (stdlib `http.server`), mặc định bind 127.0.0.1 |
| `serving.py` | `ServeManager` (start/health/stop/logs), proxy chat và ảnh qua Jupyter |
| `demo.html` | Giao diện local qua `kgl serve proxy --demo`, gửi request cùng origin với proxy |
| `doctor.py` | Chạy lần lượt các spike và báo PASS/FAIL + thời gian |
| `state.py` / `config.py` | Session 0600, redact; cấu hình qua biến môi trường |

## Luồng inference

1. `kgl serve start`: upload `remote_server.py` (xác minh sha256) → `subprocess.Popen(..., start_new_session=True)` trên VM → poll `/health` qua kernel (`loading` → `ready`).
2. `kgl infer` hoặc proxy nhận request → đóng gói JSON base64 → `execute_request` chạy snippet `INFER` trên VM → snippet gọi route nội bộ trên loopback với ID request và ID instance đã lưu → JSON hoặc SSE đi về máy local qua marker stdout. `POST /predict` gửi ảnh dưới dạng base64 và worker ảnh trả JSON.
3. Không có cổng nào mở ra internet.

## Chính sách retry / mất kết nối

- Channel rảnh từ 20 giây được mở lại trước khi gửi. Khi đang chờ request, ping dùng nonce để theo dõi pong và có kiểm tra REST khi mất pong. Lỗi kết nối chắc chắn xảy ra trước khi gửi được thử lại một lần nếu còn thời gian. Lỗi trong lúc ghi có thể đã tới kernel; với exec hoặc inference, caller nhận lỗi kết quả thực thi chưa xác định. Thời gian chờ lock được tính trong deadline.
- SSE chỉ phát `[DONE]` sau khi server trên VM báo hoàn tất. Với lỗi upstream sau khi stream đã mở, proxy phát event lỗi khi client còn nhận dữ liệu được. Request mới sẽ mở lại channel sau khi WS đứt.
- `/health` của proxy local kiểm tra liveness. `/ready` kiểm tra kernel qua REST, execute qua WS, remote health và cấu hình phiên. Với Transformers, tiêu chí GPU là device CUDA. Mỗi instance mới có định danh riêng gắn với PID, thời điểm tạo tiến trình và tham số khởi động.
- Timeout: đóng channel (trạng thái stream không xác định), lần gọi sau mở channel mới.
- Worker ảnh chạy hàm `predict` đồng bộ. Nếu client ngắt hoặc hết thời gian chờ, worker chỉ xác nhận đã nhả lượt sau khi lời gọi đó kết thúc.
- Kernel biến mất → `KernelGone` → `kgl connect` lại.
- Ctrl-C trong `kgl exec` dừng lượt chờ. Cờ `--interrupt-kernel` cho phép gửi interrupt tới kernel dùng chung. Với inference, kernel phụ được cấu hình có thể chuyển tín hiệu hủy tới worker và kiểm tra worker đã nhả lượt.

## Mã thoát

0 OK · 1 lỗi chung · 2 sai cú pháp · 3 kết nối/phiên/timeout/chưa `connect` · 4 thực thi remote lỗi hoặc bị bỏ qua · `exec --shell` trả mã của lệnh hoặc 128 cộng số hiệu signal.

## Biến môi trường

| Biến | Công dụng |
|---|---|
| `KGL_HOME` | Thư mục trạng thái phiên trên máy local, mặc định `~/.config/kgl` |
| `KGL_WORKDIR` | Thư mục gốc trên VM cho đường dẫn tương đối, mặc định `/kaggle/working` |
| `KGL_PROXY_URL` | URL Jupyter cho `kgl connect`. Script cũng có thể truyền qua stdin bằng `--proxy-url -` |
| `KGL_PROXY_KEY` | Khóa bearer của proxy HTTP local. Bind ngoài loopback cần khóa |
| `KGL_CA_FILE` | File CA tùy chỉnh cho HTTPS và WSS |
| `KGL_CHUNK_KB` | Kích thước chunk trước base64, mặc định 256 KiB, phạm vi 1–512 KiB |
| `KGL_PING_INTERVAL` | Khoảng cách giữa các ping khi chờ request, mặc định 20 giây |
| `KGL_SESSION_LIMIT_HOURS` | Số giờ giả định để tính thời gian phiên còn lại, mặc định 12 |
| `KGL_CONTROL_KERNEL_ID` | ID đầy đủ của kernel phụ trên cùng Jupyter server và VM, dùng để đọc health, hủy inference hoặc dừng đúng instance khi kernel chính bận |

## Hạn chế đã biết của thiết kế

- Một kernel và một worker xử lý mỗi lần một lượt. Proxy từ chối lượt inference đồng thời với HTTP 503.
- Backend Transformers trả số token thực trong JSON và trong SSE khi bật `stream_options.include_usage`.
- WebSocket client tự viết: không hỗ trợ HTTP proxy, nén permessage-deflate. Nếu cần, thay `ws.py` bằng thư viện `websockets` (giao diện `send/recv/ping/close` đủ nhỏ để thay).
- `container_uptime_s` là ước lượng từ `/proc` của container; không phải số liệu chính thức của Kaggle.
- Chưa có checkpoint/resume, `.ipynb`, MCP.
