# Kế hoạch dự án: `kgl` – Kaggle GPU CLI & Model Serving

Tài liệu này là kế hoạch sống: nghiên cứu (Giai đoạn 1), đánh giá khả thi, rủi ro, phạm vi MVP, tiêu chí nghiệm thu và **trạng thái xác minh thực tế**. Đề bài gốc: [`BRIEF.md`](BRIEF.md). Kiến trúc: [`ARCHITECTURE.md`](ARCHITECTURE.md). Cách kiểm thử trên Kaggle: [`TESTING.md`](TESTING.md).

> **Trạng thái trung thực:** toàn bộ code đã được test tự động với một Jupyter server *giả* (kể cả TLS). **Chưa có bất kỳ bước nào được kiểm chứng trên Kaggle thật.** Mọi mục "giả thuyết" bên dưới vẫn là giả thuyết cho tới khi bạn điền kết quả vào `TESTING.md`.

## 1. Phạm vi nghiên cứu và giới hạn của nó

| Nguồn | Đã đọc | Chưa đọc |
|---|---|---|
| googlecolab/google-colab-cli | README (lệnh, tính năng, tài liệu kiến trúc được liệt kê) | Mã nguồn, thư mục `docs/` |
| Kaggle/kaggle-api (nay là `Kaggle/kaggle-cli`) | README (tính năng: competitions, datasets, models, kernels) | Mã nguồn, `docs/`, schema `kernel-metadata.json` |
| JoseGarciaMayen/remote-bridge | README đầy đủ | Mã nguồn, `docs/security.md` |
| Điều khoản / chính sách Kaggle | **Chưa đọc trực tiếp** | Cần đọc trước khi dùng rộng rãi |
| Giới hạn phiên Kaggle | Nguồn thứ cấp trích tài liệu Kaggle (12h CPU/GPU, 9h TPU) | Trang tài liệu gốc |

## 2. Phân loại phương án kết nối

| Phương án | Loại | Trạng thái | Ghi chú |
|---|---|---|---|
| Kaggle CLI: `kernels`, `datasets`, `models` | API chính thức | Có trong README | Batch (push/run/output). Không có phiên tương tác. |
| Jupyter REST + WebSocket tới kernel của notebook đang chạy | Không chính thức | **Giả thuyết** | Cách `remote-bridge` làm. Cần proxy URL có token, lấy từ trình duyệt. |
| Lấy proxy URL bằng Playwright (remote-bridge) | Không chính thức | Do tác giả repo mô tả, chưa kiểm chứng | Dễ vỡ khi giao diện Kaggle đổi; có thể trái điều khoản. |
| Tunnel đi ra (cloudflared/ngrok) để mở HTTP | Cộng đồng | Chạy được về kỹ thuật theo nguồn cộng đồng | **Chính sách chưa xác minh.** `kgl` không dùng mặc định. |
| SSH vào VM | Cộng đồng | Rủi ro cao | Một ghi chú (có thể đã cũ) nói Kaggle dừng phiên khi chạy SSH server. |
| Colab-style API cấp phát VM | — | **Không có bằng chứng Kaggle có** | Colab CLI có backend cấp phát; không giả định Kaggle tương tự. |

## 3. Đánh giá khả thi

- **Interactive inference (cá nhân):** khả thi *có điều kiện*. Điều kiện duy nhất chưa biết là Spike 1 (kết nối kernel). Phần còn lại (chạy model nhỏ trên GPU, gọi qua Jupyter messaging) dùng cơ chế chuẩn của Jupyter.
- **Production serving 24/7:** **không khả thi** trên Kaggle: phiên bị giới hạn (12h CPU/GPU, 9h TPU theo nguồn thứ cấp), có hạn mức GPU hàng tuần (con số không có trong tài liệu Kaggle), URL/phiên thay đổi mỗi lần chạy, chỉ một phiên đồng thời mỗi tài khoản theo kinh nghiệm cộng đồng.
- **Phương án thay thế hợp lệ nếu Spike 1 thất bại:** chế độ batch qua Kaggle CLI chính thức (`kaggle kernels push` → `status` → `output`). Không tương tác, nhưng được hỗ trợ.

## 4. Quyết định thiết kế chính

1. **Không dùng public tunnel mặc định.** Model server bind `127.0.0.1` trên VM; local gọi qua Jupyter `execute_request`. Lý do: không mở endpoint ra internet, không phụ thuộc dịch vụ thứ ba, không vướng chính sách tunnel chưa rõ.
2. **Shim OpenAI-compatible chạy ở máy local** (`kgl serve proxy`), dịch HTTP → Jupyter messaging. Ứng dụng/agent dùng `base_url=http://127.0.0.1:8001/v1`.
3. **Không phụ thuộc thư viện ngoài** (stdlib + WebSocket client tối giản tự viết, đã test). Đổi lại: cần thay bằng `websockets` nếu muốn hỗ trợ proxy HTTP, permessage-deflate… (xem mục hạn chế).
4. **Transport tách biệt** khỏi logic (`ws.py` → `jupyter.py` → `runtime.py`) để thêm backend Colab sau này.
5. **An toàn mặc định:** URL có token lưu file 0600, không in ra log (`redact_url`), không lưu credentials Kaggle; `kgl auth` chỉ báo có/không.

## 5. Rủi ro chính

| # | Rủi ro | Mức | Giảm thiểu / kiểm chứng |
|---|---|---|---|
| R1 | Không có API chính thức để lấy/gắn vào phiên tương tác; proxy URL lấy qua trình duyệt | Cao | Spike 1; phương án batch dự phòng |
| R2 | Có thể trái điều khoản Kaggle (tự động hóa trình duyệt, tunnel) | Cao | Đọc điều khoản; `kgl` không tự động hóa trình duyệt, không dùng tunnel mặc định |
| R3 | Kernel bận: các `execute_request` xếp hàng | Trung bình | Model chạy ở subprocess riêng; proxy tuần tự hóa yêu cầu (concurrency = 1) |
| R4 | WebSocket qua proxy bị cắt khi im lặng lâu / giới hạn kích thước message | Trung bình | Ping định kỳ; chunk upload cấu hình `KGL_CHUNK_KB`; Spike 5 đo |
| R5 | Hết quota / GPU không được gắn / phiên hết hạn | Trung bình | `kgl status` hiển thị GPU + thời gian container (xấp xỉ) |
| R6 | Model tải từ Hugging Face cần Internet bật trong notebook | Thấp–TB | Ghi trong `TESTING.md`; dùng dataset đính kèm thay thế |
| R7 | Token trong URL bị lộ | TB | 0600, redact, không log |

## 6. Phạm vi MVP (nhỏ nhất chứng minh mục tiêu)

Trong MVP: `auth`, `connect`, `status`, `exec` (Python/`--shell`), `upload`, `download` (có xác minh sha256), `serve start|stop|status|logs|proxy`, `infer` (stream), `stop`, thêm `doctor` để chạy các spike tự động.

Ngoài MVP (cố ý hoãn): chạy `.ipynb`, đồng bộ cả project/quản lý dependencies, checkpoint/resume, MCP server, backend Colab, tự động lấy proxy URL, đọc quota, hỗ trợ vLLM/llama.cpp, nhiều request đồng thời.

## 7. Kế hoạch triển khai & tiêu chí nghiệm thu

Cột "Local" = đã test với Jupyter giả. Cột "Kaggle" = bạn xác nhận trên Kaggle thật (điền ở `TESTING.md`).

| # | Hạng mục | Tiêu chí nghiệm thu | Local | Kaggle |
|---|---|---|---|---|
| 1 | Kết nối kernel | `kgl connect` liệt kê kernel, `print(1+1)` trả `2`; sai/hết hạn token → `AuthError` rõ ràng | ✅ | ⬜ |
| 2 | Thực thi + stream | Output hiện dần; lỗi remote → exit 4 + traceback; `--shell` trả đúng mã thoát | ✅ | ⬜ |
| 3 | GPU hiển thị | `kgl status` thấy tên GPU + VRAM | ✅ (nhánh "không có GPU") | ⬜ |
| 4 | Model nhỏ trên GPU | `kgl serve start --backend transformers` → `ready` | ❌ chưa chạy (cần GPU) | ⬜ |
| 5 | Inference từ local | `kgl infer` stream đúng; p50/p95 latency ghi lại | ✅ (backend `echo`) | ⬜ |
| 6 | OpenAI-compat local | SDK/curl gọi `/v1/chat/completions` (stream + non-stream), API key | ✅ (backend `echo`) | ⬜ |
| 7 | File I/O | Upload/download nhiều chunk, khớp sha256 | ✅ | ⬜ |
| 8 | Mất kết nối | Ngắt WS → retry an toàn; mất sau khi gửi → *không* chạy lại code không idempotent | ✅ (mô phỏng) | ⬜ (ngắt mạng thật) |
| 9 | Độ bền | 20+ request liên tiếp; phiên chạy ≥ 1h không treo | ⬜ | ⬜ |
| 10 | Checkpoint/resume | Tác vụ dài tiếp tục được sau khi phiên mới | ⬜ chưa làm | ⬜ |

Thứ tự ưu tiên tiếp theo: (1) Spike 1 → quyết định đi tiếp hay chuyển sang batch; (2) Spike 4–5 với model thật; (3) đo độ bền; (4) chỉ sau đó mới làm checkpoint/resume, MCP, `.ipynb`.

## 8. Tuân thủ

`kgl` không có tính năng né quota, kéo dài phiên hay giả lập hoạt động. `kgl stop` **không** kết thúc phiên Kaggle: hãy dừng phiên từ giao diện Kaggle để không tiêu hao quota. Người dùng chịu trách nhiệm đọc và tuân thủ điều khoản Kaggle.

## 9. Nguồn tham khảo

- https://github.com/googlecolab/google-colab-cli (README)
- https://github.com/Kaggle/kaggle-cli (README; trước đây `Kaggle/kaggle-api`)
- https://github.com/JoseGarciaMayen/remote-bridge (README)
- https://github.com/ARahim3/kaggle-tpu-lab, qua bài tóm tắt https://discuss.pytorch.kr/t/kaggle-tpu-lab-kaggle-tpu-320b-glm-5-3-flash-api/11944 (bài tóm tắt tự ghi là do GPT soạn nháp; chưa đọc repo gốc)
- https://gpuperhour.com/blog/free-cloud-gpus-and-credits (nguồn thứ cấp về giới hạn phiên)
