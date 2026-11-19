# Troubleshooting

| Triệu chứng | Nguyên nhân có thể | Cách xử lý |
|---|---|---|
| Trang demo trả 404 | Proxy chạy thiếu cờ demo | Dừng proxy bằng Ctrl-C, chạy `kgl serve proxy --demo` rồi mở địa chỉ được in ra |
| Demo ảnh báo lỗi đầu vào | File hỏng, sai định dạng hoặc quá kích thước | Dùng JPEG hoặc PNG tối đa 350 KiB và 16 megapixel. Gửi lại một ảnh hợp lệ |
| Worker ảnh báo lỗi khi nạp | Thiếu thư viện, checkpoint hoặc kiến trúc không khớp | Xem `kgl serve logs`. Kiểm tra đường dẫn thực trong VM và dùng đúng kiến trúc đã lưu trọng số. [Hướng dẫn checkpoint riêng](../examples/demos.md#use-your-own-checkpoint) |
| Terminal giữ nguyên sau `kgl serve proxy` | Proxy đang phục vụ ở foreground | Giữ terminal A mở. Ở B chạy `curl --max-time 5 http://127.0.0.1:8001/health`, sau đó gửi một chat request ngắn theo [bước 6](TESTING.md). Ctrl-C ở A dừng proxy local |
| Start báo legacy server | Server còn chạy bằng phiên bản cũ | Khi sẵn sàng ngắt dịch vụ cũ, chạy `kgl serve stop`, sau đó `kgl serve start` lại và mở lại proxy local bằng CLI mới |
| Local health OK nhưng chat lỗi hoặc treo | Local health chỉ kiểm tra tiến trình proxy | Thử `curl -i --max-time 10 http://127.0.0.1:8001/ready`. Trong terminal dùng cùng KGL_HOME, chạy `kgl exec --timeout 30 'print(2)'`. Nếu thành công, kiểm tra `kgl serve status`, backend/model và log. Ghi rõ lần cuối nhận chunk và có nhận dấu kết thúc hay chưa |
| Stream đứt giữa câu | WS hoặc worker có thể đã mất sau khi phát nội dung | Giữ phần đã nhận cùng thời điểm lỗi. Kiểm tra kết nối và thử một request mới. Nếu cần thử lại, tạo phản hồi mới riêng biệt để tránh ghép trùng prefix |
| `connection error: ... cannot reach` | URL sai, mạng, proxy hết hạn | Kiểm tra phiên notebook còn chạy; lấy URL mới; `kgl -v connect ...` |
| `HTTP 403/401 ... expired or not accepted` | Token hết hạn/phiên mới | Kiểm tra phiên đang chạy, sao chép lại VS Code Compatible URL từ panel Kaggle Jupyter Server và connect qua stdin |
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
