# kgl – Kaggle GPU CLI

Chạy Python, truyền file và hỏi model AI trên GPU của một Kaggle notebook đang chạy, từ terminal máy bạn.

> **Trạng thái: PoC chưa được xác minh trên Kaggle thật.** Code đã qua 27 test tự động với Jupyter server giả. Hãy đọc [`docs/PLAN.md`](docs/PLAN.md) và làm theo [`docs/TESTING.md`](docs/TESTING.md) để kiểm chứng.

- Không phụ thuộc thư viện ngoài (Python ≥ 3.10).
- Không dùng public tunnel: model server chỉ bind `127.0.0.1` trên VM; request đi qua Jupyter messaging.
- Phù hợp *interactive inference* cá nhân. **Không** dành cho serving 24/7.

## Cài đặt

```bash
pip install -e .
```

## Dùng nhanh

```bash
echo "$PROXY_URL" | kgl connect --proxy-url -     # xem docs/TESTING.md để biết cách lấy URL
kgl status
kgl exec "print('hello')"
kgl exec --shell "nvidia-smi"
kgl upload data.csv data.csv && kgl download out/result.txt ./
kgl serve start --model Qwen/Qwen2.5-0.5B-Instruct
kgl infer "Giải thích GPU là gì" --timing
kgl serve proxy                    # OpenAI-compatible tại http://127.0.0.1:8001/v1
kgl doctor --backend echo          # kiểm tra đường truyền không cần model
kgl stop                           # nhớ dừng phiên trên giao diện Kaggle
```

Dùng với client OpenAI:

```python
from openai import OpenAI
c = OpenAI(base_url="http://127.0.0.1:8001/v1", api_key="unused")
print(c.chat.completions.create(model="x", messages=[{"role": "user", "content": "hi"}]).choices[0].message.content)
```

## Biến môi trường

`KGL_HOME` (mặc định `~/.config/kgl`), `KGL_WORKDIR` (`/kaggle/working`), `KGL_CHUNK_KB` (256), `KGL_PROXY_URL`, `KGL_PROXY_KEY`, `KGL_SESSION_LIMIT_HOURS` (12), `KGL_PING_INTERVAL` (20), `KGL_CA_FILE` (CA/chứng chỉ tùy chỉnh cho https/wss, ví dụ proxy công ty).

## Tài liệu

[`docs/PLAN.md`](docs/PLAN.md) · [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) · [`docs/TESTING.md`](docs/TESTING.md) · [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) · [`docs/BRIEF.md`](docs/BRIEF.md)

## Chưa có

Checkpoint/resume, chạy `.ipynb`, MCP server, đọc quota, backend Colab, vLLM/llama.cpp, nhiều request đồng thời.
