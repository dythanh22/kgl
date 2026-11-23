# Kiểm tra đợt tái cấu trúc ngày 23/11/2026

## Lần chạy đầu trong sandbox

| Phép kiểm | Kết quả | Phạm vi |
|---|---|---|
| `python -m unittest discover -s tests -t . -q` | 20 đạt, 43 lỗi tạo socket, 1 thất bại khi khởi động Jupyter local, 1 bỏ qua | 65 test của kgl. Sandbox từ chối mở socket. |
| `python -m unittest discover -s examples/image-classifier -q` | 2 đạt | Giải mã JPEG/PNG bằng Pillow, kiểm tra dữ liệu lỗi và giới hạn byte. |
| `python -m unittest discover -s examples/agent -q` | Bị chặn | HTTP server của fixture cần socket loopback. |
| `python -m tests.wheel_smoke` | Đạt phần đóng gói | Build và cài wheel offline, kiểm tra tài nguyên, import với `python -S`, chạy CLI. Fake Jupyter bị chặn ở bước mở socket. |
| Gateway và CLI | Đạt phần kiểm tra local | Chạy `serve.pyz --help` trong Python cô lập. Dùng runtime giả kiểm tra preset tới argv, context 4096 và lưu session. Đọc được serve record cũ. |
| Demo và forward | Đạt phần kiểm tra trong bộ nhớ | Nội dung trang gốc khớp HTML trong example. Chạy snippet VM với HTTP upstream giả, kiểm tra dữ liệu nhị phân đi và về ở cả chế độ trực tiếp và nhiều chunk. |
| Kaggle | Chưa nghiệm thu | Phép thử `kgl exec` dừng khi phân giải DNS. Chưa xác minh SAM3 hoặc classifier trên GPU qua các đường gọi mới. |
| Test SAM3 | Chưa chạy | Môi trường local thiếu pytest và dependency riêng của example. |

Các mốc GPU trước đây thuộc phiên bản ghi trong báo cáo tương ứng. [Quy trình kiểm thử](../TESTING.md)

## Nghiệm thu lại trên checkout sau refactor

Python local 3.12.16 trong `myenv`. Dependency test của SAM3 được cài theo `uv.lock` vào thư mục tạm. Các phép thử cần socket chạy với quyền mở loopback và kết nối đến notebook. `kgl connect` và `kgl exec --timeout 15 'print(2)'` thành công với URL của phiên mới. VM chạy Python 3.13.16 và nhận hai Tesla T4, mỗi card 15.360 MiB.

| Phép kiểm | Kết quả quan sát |
|---|---|
| Suite kgl | 65/65 đạt, gồm Fake Jupyter, Jupyter/ipykernel thật, TLS, gateway, forward và client OpenAI. Lần đầu bỏ qua test SDK. Chạy lại toàn suite với SDK 3.57.0 trong thư mục tạm mất 76,7 giây. |
| Wheel | `python -m tests.wheel_smoke` đạt cả build, cài offline và tích hợp với Fake Jupyter. |
| Example | Agent đạt 1/1, classifier đạt 2/2. |
| SAM3 CPU | 111 đạt, 2 module bỏ qua (bộ test GPU và adapter CVAT/Nuclio ngoài checkout). FastAPI TestClient phát một cảnh báo deprecation về httpx. |
| Chat trên GPU | Qwen2.5-0.5B-Instruct, Transformers, `cuda:0`, float16. Qua HTTP local, 10 JSON và 10 SSE đều thành công. p50 0,436 giây, p95 0,443 giây. Stream bị cắt sau delta có nội dung, request tiếp theo thành công. Worker thử trả `stopped: true`. Lần chạy này dùng kernel chính. |
| Classifier với checkpoint đỏ/xanh | Firefox headless mở demo từ `kgl serve proxy --static examples/chat-demo`. Handler chạy trên CUDA. PNG đỏ có nhãn đầu red (95,3%). Gửi ảnh hỏng tới `POST /handle` nhận HTTP 400. JPEG xanh sau đó có nhãn đầu blue (95,2%) trên cùng instance. Lệnh dừng worker thành công. |
| SAM3 qua forward | Uvicorn nạp checkpoint đã gắn, Torch 2.10.0+cu128, Tesla T4, FP32. Với `KGL_REQUEST_BYTES=33554432`, forward từ loopback 8014 đến cổng VM 8013 trả HTTP 200 cho `/readyz`. Firefox mở `/demo`, gửi ảnh giao thông mẫu cùng prompt `car` (threshold 0,4, tối đa 10). Giao diện hiển thị 10 mask và thiết bị CUDA. Đã gửi TERM tới PID Uvicorn của lần thử. |
| Hủy qua kernel phụ | Chạy lại smoke với kernel điều khiển tạm. Đạt thêm 20/20 request, p50 0,430 giây, p95 0,432 giây. Sau khi cắt stream, `cancellation_confirmed: true`, request tiếp theo thành công. Đã dừng worker và thu hồi kernel phụ. |
| Classifier ViT mặc định | Cùng trình tự trong Firefox với `google/vit-base-patch16-224`. Mỗi ảnh hợp lệ trả ba nhãn và điểm, ảnh hỏng trả HTTP 400. Instance giữ nguyên qua ba lượt, sau đó dừng thành công. Hai ảnh đơn sắc dùng để kiểm tra đường phục vụ. |

Sau nghiệm thu, cổng SAM3 8013 đã đóng và cả hai T4 báo 0 MiB VRAM sử dụng. Notebook vẫn đang chạy. Thư mục runtime thử `/kaggle/working/accept-sam-oDwvCC` chiếm 7,3 GiB trên VM.
