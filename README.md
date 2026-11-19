# kgl – Kaggle GPU CLI

Chạy Python, truyền file và hỏi model AI trên GPU của một Kaggle notebook đang chạy, từ terminal máy bạn.

> **Đã kiểm chứng PoC trên một phiên Kaggle GPU.** Qua proxy local, 20/20 request tuần tự (10 JSON, 10 SSE) nhận được phản hồi từ Qwen trên Tesla T4. Bản sửa đã phục hồi request kế tiếp sau khi WS bị ngắt giữa stream. Độ bền một giờ và phục hồi qua phiên mới còn cần nghiệm thu. [Bằng chứng và thứ tự nâng cấp](docs/PLAN.md) · [Tự kiểm thử](docs/TESTING.md)

- Không phụ thuộc thư viện ngoài (Python ≥ 3.10).
- Không dùng public tunnel: model server chỉ bind `127.0.0.1` trên VM; request đi qua Jupyter messaging.
- Phù hợp *interactive inference* cá nhân. **Không** dành cho serving 24/7.

## Cài đặt

```bash
pip install -e .
```

## Dùng nhanh

Trong một notebook Kaggle đang chạy với GPU, mở **Run → Kaggle Jupyter Server → Manually Connect** và sao chép **VS Code Compatible URL**. Chạy trên máy cá nhân:

```bash
kgl connect                       # dán URL khi được hỏi, ký tự được ẩn; nhấn Enter
kgl exec 'print(1+1)'             # kỳ vọng 2
kgl serve start --model Qwen/Qwen2.5-0.5B-Instruct
kgl infer "Xin chào" --timing
```

Khi dùng xong, chạy `kgl stop` rồi **dừng session trong giao diện Kaggle**.

### Gọi từ ứng dụng hoặc agent

Sau khi model sẵn sàng, tại terminal A chạy `kgl serve proxy` và giữ terminal này mở. Tại terminal B dùng cùng session:

```bash
curl --fail http://127.0.0.1:8001/ready
python examples/agent_infer.py "Xin chào"
```

Ví dụ kiểm tra session và readiness rồi in một JSON khi inference hoàn tất. Script đọc stdout để lấy dữ liệu, stderr để chẩn đoán và kiểm tra mã thoát trước khi parse. Mã 3 biểu thị lỗi kết nối/phiên/timeout; nếu trạng thái thực thi chưa xác định, kiểm tra trạng thái remote trước khi gửi lại lệnh có tác dụng phụ. Với pipeline headless, truyền URL qua `kgl connect --proxy-url -`. Khi xong, Ctrl-C ở A, chạy `kgl stop` rồi dừng phiên trên Kaggle.

Dùng với client OpenAI:

```python
from openai import OpenAI
c = OpenAI(base_url="http://127.0.0.1:8001/v1", api_key="unused")
print(c.chat.completions.create(model="x", messages=[{"role": "user", "content": "hi"}]).choices[0].message.content)
```

## Biến môi trường

`KGL_HOME` (mặc định `~/.config/kgl`), `KGL_WORKDIR` (`/kaggle/working`), `KGL_CHUNK_KB` (256), `KGL_PROXY_URL`, `KGL_PROXY_KEY`, `KGL_SESSION_LIMIT_HOURS` (12), `KGL_PING_INTERVAL` (20), `KGL_CA_FILE` (CA/chứng chỉ tùy chỉnh cho https/wss, ví dụ proxy công ty).

## Tài liệu

[Kế hoạch nâng cấp v2](docs/PLAN_V2.md)

[`docs/PLAN.md`](docs/PLAN.md) · [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) · [`docs/TESTING.md`](docs/TESTING.md) · [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) · [`docs/BRIEF.md`](docs/BRIEF.md)

## Chưa có

Checkpoint/resume, chạy `.ipynb`, MCP server, đọc quota, backend Colab, vLLM/llama.cpp, nhiều request đồng thời.
