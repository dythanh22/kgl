# Vận hành Qwen Q4_K_M trên hai Tesla T4 của Kaggle

## Kết nối và khởi động

Mở Bash ở gốc repo với Python của `myenv` và bộ `KGL_HOME` bạn định dùng cho phiên này. Chọn **GPU T4 ×2**, gắn GGUF Q4_K_M theo [preset Qwen](qwen3.8-35b-a3b.json), rồi mở **Run → Kaggle Jupyter Server → Manually Connect**. Dán **VS Code Compatible URL** vào lời nhắc ẩn của `connect`. Với phiên đã kết nối, bắt đầu từ `exec`.

```bash
python -m kgl connect
python -m kgl exec --timeout 15 'print(2)'
python examples/qwen35b-llama/ready.py
python examples/qwen35b-llama/proxy.py
python examples/qwen35b-llama/check_answer.py
```

`exec` phải in `2`. `ready.py` khôi phục dataset build ghim khi thiếu, kiểm manifest, binary và SHA-256 của GGUF, rồi xác nhận đúng worker và hai GPU. Trên Kaggle Input chỉ đọc, lần kiểm lại cùng artifact có thể dùng cache SHA trong VM. Khi chạy lại cùng cấu hình, instance và PID vẫn giữ nguyên. `check_answer.py` phải in `Answer: '4'`. [Build, khôi phục thủ công và xử lý lỗi CUDA](BUILD.md)

Mở [chat local](http://127.0.0.1:8001/) hoặc dùng Base URL `http://127.0.0.1:8001/v1` với alias `qwen-local`. Dùng cùng origin và cùng hồ sơ trình duyệt để mở lại lịch sử. [Ví dụ API và JSON](API.md) · [Hướng dẫn giao diện](../chat-demo/README.md)

## Khi đổi VM

1. Trên VM cũ, ghi instance từ `python -m kgl serve status`. Gửi một mã thử trong chat và lưu hội thoại. Chạy một ca API trong hướng dẫn để có đầu ra trước khi đổi.
2. Trên notebook mới, chọn hai T4, xác nhận input model và quyền truy cập bản build theo preset. Kết nối bằng `python -m kgl connect` và chạy lại chuỗi `exec`, `ready.py`, `proxy.py` ở trên. Ghi thời gian tới nội dung đầu nếu đang nghiệm thu.
3. Nếu dùng `KGL_HOME` mới và proxy cũ đã mất kết nối nhưng còn giữ cổng, truyền thư mục phiên trước vào lệnh bên dưới. Lệnh kiểm đúng chủ sở hữu, key, worker đích và tình trạng proxy trước khi bàn giao.
4. Mở lại hội thoại từ VM cũ, hỏi lại mã đã lưu và chạy lần nữa ca API với cùng alias và Base URL. `/ready` phải trả đúng instance mới. [Checklist nghiệm thu](../../docs/PLAN-qwen-daily.md)

```bash
python examples/qwen35b-llama/proxy.py --from-home /duong/dan/toi/session-cu
```

## Xử lý sự cố

| Dấu hiệu | Cách xử lý |
|---|---|
| `exec` báo mất kết nối | Kiểm tra notebook trong Kaggle, chạy lại `python -m kgl connect` qua lời nhắc ẩn rồi chạy `ready.py` và `proxy.py`. |
| Thiếu GGUF, lỗi SHA hoặc thiếu build | Kiểm tra quant, đường dẫn và version trong preset. Xem quy trình kiểm binary và library trong [BUILD.md](BUILD.md). |
| Worker đang bận | Chờ lượt hiện tại hoàn tất rồi chạy lại hoặc chọn **Check connection**. |
| Worker cũ thiếu alias hoặc báo xung đột cấu hình | Xem trạng thái với `python -m kgl serve status`. Khi có thể gián đoạn, dừng worker đúng session rồi chạy lại `ready.py` và `proxy.py`. |
| Cổng 8001 có listener khác | Kiểm tra tiến trình đang giữ cổng. Giải phóng cổng bằng công cụ của ứng dụng sở hữu nó để giữ cùng origin. |
| Proxy yêu cầu bearer key | Đặt cùng `KGL_PROXY_KEY` cho proxy và client. Xem [cách nhập key](API.md#xác-thực). |

Trình quản lý proxy nền cần Linux, `ss`, `/proc` và `pidfd`. Log nằm trong `qwen-proxy.log` ở session store. Khi cổng 8001 trống, cũng có thể chạy proxy foreground bằng `python -m kgl serve proxy --static examples/chat-demo` và dừng bằng Ctrl-C.

## Thử context theo nhu cầu

**32.768 token là mặc định của preset khi chạy trực tiếp hoặc qua `ready.py` cho chat, tóm tắt, biên tập và trích xuất văn bản.** [Bảng số đo context](../../docs/reports/2026-11-24-qwen-m3-5.md)

Khi worker hiện tại đang rảnh và bạn muốn đổi context, chạy lần lượt trong `myenv` với cùng `KGL_HOME`:

```bash
python -m kgl serve stop
python examples/qwen35b-llama/ready.py --ctx 32768
python examples/qwen35b-llama/proxy.py
```

Dùng `--ctx 65536` khi cần cửa sổ lớn hơn, hoặc `--ctx 4096` để trở về cấu hình ban đầu. Mốc 262.144 được ghi lại trong báo cáo benchmark. Khi worker đang chạy ở context khác mặc định, truyền lại cùng giá trị `--ctx` để dùng lại worker. GUI lấy context thực tế qua `/ready`. Client gửi prompt rất dài cần chừa thời gian nhận token đầu tiên và kiểm tra giới hạn kích thước HTTP của proxy.

## Dừng phục vụ

Chờ lượt đang chạy kết thúc. Dùng `python -m kgl serve stop` để dừng worker và giữ kết nối notebook. Khi kết thúc cả phiên, dùng `python -m kgl stop` để xóa kết nối local, rồi dừng notebook trong giao diện Kaggle và xác nhận đã dừng. Proxy foreground thoát bằng Ctrl-C. Proxy nền có thể giữ origin cho lần kết nối sau và được quản lý bằng `proxy.py` khi phiên mới đã ready.

[Kết quả khởi động, khôi phục và benchmark](../../docs/reports/2026-11-25-qwen-l9-l10.md)
