# Troubleshooting

| Triệu chứng | Nguyên nhân có thể | Cách xử lý |
|---|---|---|
| `connection error: ... cannot reach` | URL sai, mạng, proxy hết hạn | Kiểm tra phiên notebook còn chạy; lấy URL mới; `kgl -v connect ...` |
| `HTTP 403/401 ... expired or not accepted` | Token hết hạn/phiên mới | Khởi động lại phiên, lấy URL mới, `kgl connect` lại; thử `--header "Cookie: ..."` nếu proxy yêu cầu (chưa kiểm chứng) |
| `HTTP 404 ... does not look like a Jupyter server base URL` | URL không phải gốc Jupyter | Thử cắt bớt/đổi path; báo lại định dạng URL thực tế để tôi hỗ trợ |
| `WebSocket upgrade refused (HTTP xxx)` | Proxy chặn WS hoặc thiếu header | Ghi lại mã HTTP; xem `-v`; có thể cần header bổ sung |
| `kernel not found` / `KernelGone` | Phiên Kaggle đã kết thúc/khởi động lại | `kgl connect` lại; `kgl serve start` lại (VM mới) |
| `no GPU visible` | Accelerator chưa bật, hoặc Kaggle không gắn GPU | Bật GPU trong Settings, khởi động lại phiên |
| `serve start` kẹt ở `loading` | Tải model chậm / Internet Off | `kgl serve logs`; bật Internet hoặc đính kèm model dưới dạng dataset |
| `model failed to load: ... out of memory` | Model quá lớn cho VRAM | Dùng model nhỏ hơn hoặc `--dtype float16` |
| Proxy local trả 503 `kgl_connection_lost` | WS bị cắt | Thử lại; `kgl status`; kết nối lại nếu kernel mất |
| `kgl infer` chậm lần đầu | Khởi tạo CUDA/compile | Bỏ qua lần đầu khi đo p50/p95 (doctor dùng cả lần đầu) |
| Cần ghi log chi tiết | | `kgl -v <lệnh>` (URL luôn được redact) |

Quota và thời gian còn lại: `kgl status` chỉ hiển thị ước lượng từ uptime của container; quota hàng tuần **không đọc được** (chưa có API chính thức). Hãy xem trong cài đặt tài khoản Kaggle.
