# Kiến trúc

```
Agent / app / browser / terminal
        │ HTTP (chat / handle)                kgl exec / infer / upload ...
        ▼                                              │
  kgl serve proxy (127.0.0.1)                   cli/app.py → cli/commands/
        │                                              │
        └────────────────────► client/ ◄──────────────┘
                               │
                       transport/runtime.py (session + reconnect + retry)
                               │
                       transport/jupyter.py (REST + kernel channel)
                               │
                       transport/ws.py      (RFC 6455, ws/wss)
                               │   Jupyter messaging over WebSocket
 ══════════════════════════════╪═══════════ Kaggle notebook proxy ═══
                               ▼
                    Kaggle VM: ipykernel
                         │  remote/snippets/*.py
                         ├─ upload/download/stat/probe (stdlib)
                         └─ subprocess: serve.pyz  (127.0.0.1:8000, not exposed)
                                         └─ transformers | python | echo | llama
                                                                                      └─ llama.cpp (loopback)
```

## Module

| Module | Vai trò |
|---|---|
| [`cli/`](../src/kgl/cli) | Parser, mã thoát và các nhóm lệnh session, exec, file, serve, model, infer, doctor, forward |
| [`core/`](../src/kgl/core) | Cấu hình môi trường, lỗi và lưu session 0600 |
| [`transport/`](../src/kgl/transport) | WebSocket, REST Jupyter, kernel channel, deadline và retry |
| [`remote/`](../src/kgl/remote) | Nạp snippet Python bằng `importlib.resources`, render tham số và đọc marker theo nonce |
| [`client/`](../src/kgl/client) | Truyền file, vòng đời gateway, inference, chuẩn bị model và doctor |
| [`proxy/`](../src/kgl/proxy) | HTTP local, kiểm tra Host/Origin/Bearer và demo cùng origin |
| [`protocol.py`](../src/kgl/protocol.py) | Header nội bộ, health và kiểm tra request dùng chung |
| [`vm/`](../src/kgl/vm) | Gateway, trạng thái generation, adapter backend và công cụ chuẩn bị llama.cpp trên VM |
| [`backends.py`](../src/kgl/backends.py) | Backend và hợp đồng tùy chọn khởi động dùng chung |

Example dùng API công khai trong `kgl`: Settings, SessionStore, Runtime, upload_file, download_file, infer_chat và KglError. Preset JSON nằm cạnh example của model.

## Luồng inference

1. `kgl serve start`: upload `serve.pyz` (xác minh sha256) → `subprocess.Popen(..., start_new_session=True)` trên VM → ghi instance `starting` vào phiên local → poll `/health` qua kernel (`loading` → `ready`). Tiến trình llama.cpp nằm cùng process group với gateway.
2. `kgl infer` hoặc proxy nhận request → đóng gói JSON base64 → `execute_request` chạy snippet `INFER` trên VM → snippet gọi route nội bộ trên loopback với ID request và ID instance đã lưu → JSON hoặc SSE đi về máy local qua marker stdout. `POST /handle` chuyển JSON tới hàm handle của module Python trên VM.
3. Không có cổng nào mở ra internet.

## Chính sách retry / mất kết nối

- Channel rảnh từ 20 giây được mở lại trước khi gửi. Khi đang chờ request, ping dùng nonce để theo dõi pong và có kiểm tra REST khi mất pong. Lỗi kết nối chắc chắn xảy ra trước khi gửi được thử lại một lần nếu còn thời gian. Lỗi trong lúc ghi có thể đã tới kernel; với exec hoặc inference, caller nhận lỗi kết quả thực thi chưa xác định. Thời gian chờ lock được tính trong deadline.
- SSE chỉ phát `[DONE]` sau khi server trên VM báo hoàn tất. Với lỗi upstream sau khi stream đã mở, proxy phát event lỗi khi client còn nhận dữ liệu được. Request mới sẽ mở lại channel sau khi WS đứt.
- `/health` của proxy local kiểm tra liveness. `/ready` kiểm tra kernel qua REST, execute qua WS, remote health và cấu hình phiên. Với Transformers, tiêu chí GPU là device CUDA. Mỗi instance mới có định danh riêng gắn với PID, thời điểm tạo tiến trình và tham số khởi động.
- Timeout: đóng channel (trạng thái stream không xác định), lần gọi sau mở channel mới.
- Worker Python chạy hàm `handle` đồng bộ. Nếu client ngắt hoặc hết thời gian chờ, worker chỉ xác nhận đã nhả lượt sau khi lời gọi đó kết thúc.
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
| `KGL_REQUEST_BYTES` | Giới hạn JSON trước base64, mặc định 524288 byte. Dùng cùng giá trị lúc start và lúc mở proxy |
| `KGL_INTERRUPT_INFER_KERNEL` | Giá trị `1` cho phép interrupt kernel chính khi cần hủy infer và thiếu kernel phụ. Kernel do kgl tạo có sẵn quyền fallback này |

## Hạn chế đã biết của thiết kế

- Một kernel và một worker xử lý mỗi lần một lượt. Proxy từ chối lượt inference đồng thời với HTTP 503.
- Backend Transformers trả số token thực trong JSON và trong SSE khi bật `stream_options.include_usage`.
- WebSocket client tự viết: không hỗ trợ HTTP proxy, nén permessage-deflate. Nếu cần, thay `ws.py` bằng thư viện `websockets` (giao diện `send/recv/ping/close` đủ nhỏ để thay).
- `container_uptime_s` là ước lượng từ `/proc` của container; không phải số liệu chính thức của Kaggle.
- Chưa có checkpoint/resume, `.ipynb`, MCP.

## Chuyển tiếp HTTP

`kgl forward` nhận HTTP trên loopback local và gửi đến cổng loopback đã chọn của notebook. CLI kiểm tra Host, Origin và giới hạn kích thước trước khi gửi qua Jupyter. Request nhỏ và response nhỏ đi trực tiếp trong message. Dữ liệu lớn hơn kích thước chunk dùng cơ chế truyền file có checksum. Forward giữ mã trạng thái, header ứng dụng và dữ liệu nhị phân. Cookie, thông tin origin của trình duyệt và header dành riêng cho từng kết nối được loại bỏ. Kết quả trả về sau khi đã đọc đủ response.
