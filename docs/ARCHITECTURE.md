# Kiến trúc

```
Agent / app / terminal
        │ HTTP (OpenAI-compatible)            kgl exec / infer / upload ...
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
                                         └─ transformers (GPU) | echo
```

## Module

| Module | Vai trò |
|---|---|
| `ws.py` | WebSocket client stdlib; buffer giữ byte của frame đang đọc, giữ các fragment qua timeout; ping/pong; wss |
| `jupyter.py` | `Endpoint` (giữ `?token=` của URL), REST kernels, `KernelChannel.execute` (idle/total timeout, keep-alive ping) |
| `runtime.py` | Giữ channel, kết nối lười, chính sách retry (xem dưới) |
| `remote.py` | Mã chạy trên VM + giao thức marker `@@KGL@@{json}` để trả dữ liệu có cấu trúc trong stdout |
| `files.py` | Upload/download chunk base64, xác minh sha256 |
| `remote_server.py` | Server chat-completions (stdlib `http.server`), backend `transformers`/`echo`, chỉ bind 127.0.0.1 |
| `serving.py` | `ServeManager` (start/health/stop/logs), `infer_chat`, shim OpenAI-compat local |
| `doctor.py` | Chạy lần lượt các spike và báo PASS/FAIL + thời gian |
| `state.py` / `config.py` | Session 0600, redact; cấu hình qua biến môi trường |

## Luồng inference

1. `kgl serve start`: upload `remote_server.py` (xác minh sha256) → `subprocess.Popen(..., start_new_session=True)` trên VM → poll `/health` qua kernel (`loading` → `ready`).
2. `kgl infer` / shim nhận request → đóng gói JSON base64 → `execute_request` chạy snippet `INFER` trên VM → snippet gọi `http://127.0.0.1:8000/v1/chat/completions` → mỗi dòng SSE phát lại thành marker trên stdout → local parse marker, stream ra client.
3. Không có cổng nào mở ra internet.

## Chính sách retry / mất kết nối

- Lỗi trước khi gửi được thử lại một lần. Lỗi trong lúc ghi có thể đã tới kernel; với exec hoặc inference, caller nhận lỗi kết quả thực thi chưa xác định. Thời gian chờ lock được tính trong deadline của request.
- SSE chỉ phát `[DONE]` sau khi server trên VM báo hoàn tất. Với lỗi upstream sau khi stream đã mở, proxy phát event lỗi khi client còn nhận dữ liệu được. Request mới sẽ mở lại channel sau khi WS đứt.
- `/health` của proxy local kiểm tra liveness. `/ready` kiểm tra kernel qua REST, execute qua WS, remote health và cấu hình phiên. Với Transformers, tiêu chí GPU là device CUDA. Mỗi instance mới có định danh riêng gắn với PID, thời điểm tạo tiến trình và tham số khởi động.
- Timeout: đóng channel (trạng thái stream không xác định), lần gọi sau mở channel mới.
- Kernel biến mất → `KernelGone` → `kgl connect` lại.
- Ctrl-C trong `kgl exec` → `POST /api/kernels/<id>/interrupt`.

## Mã thoát

0 OK · 1 lỗi chung · 2 sai cú pháp · 3 kết nối/phiên/timeout/chưa `connect` · 4 lỗi code phía remote · với `exec --shell`: mã thoát của lệnh (0–125).

## Hạn chế đã biết của thiết kế

- Một kernel = một lần chạy tại một thời điểm; shim tuần tự hóa yêu cầu.
- Server `transformers` đơn giản (không continuous batching, không đếm token `usage`). Muốn thông lượng cao cần vLLM/llama.cpp (ngoài MVP).
- WebSocket client tự viết: không hỗ trợ HTTP proxy, nén permessage-deflate. Nếu cần, thay `ws.py` bằng thư viện `websockets` (giao diện `send/recv/ping/close` đủ nhỏ để thay).
- `container_uptime_s` là ước lượng từ `/proc` của container; không phải số liệu chính thức của Kaggle.
- Chưa có checkpoint/resume, `.ipynb`, MCP.
