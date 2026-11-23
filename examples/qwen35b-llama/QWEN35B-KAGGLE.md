# Chạy Qwen3.8-35B-A3B Q4_K_M trên hai Tesla T4 của Kaggle

Mở terminal Bash ở thư mục gốc repo. llama.cpp được build và lưu trên Kaggle. Nếu phiên Kaggle còn thư mục build của đúng commit bên dưới, kiểm tra binary rồi chuyển thẳng đến bước thử model. Lần build mới trên 4 CPU mất khoảng 35 phút.

## 1. Kiểm tra phiên, GGUF và build sẵn có

```bash
# Dùng môi trường và KGL_HOME của phiên đã kết nối.
kgl exec --timeout 15 'print(2)'
```

Nếu kernel đã mất, chạy `kgl connect` và dán VS Code Compatible URL vào lời nhắc ẩn. Chạy lại `print(2)` trước khi tiếp tục.

```bash
REV=f4f6f2e7db76e98c95cc9b7fc608ea669c806044
BUILD_ROOT=/tmp/kgl/llama
PACKAGE="$BUILD_ROOT/$REV"
MANIFEST="$PACKAGE/manifest.json"
GGUF=/kaggle/input/models/thanhduy1022/qwen38-35b-a3b/gguf/q4-k-m/1/Qwen3.8-35B-A3B-Q4_K_M.gguf

kgl exec --shell --timeout 20 "test -r '$GGUF' && ls -lh '$GGUF' && nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader"
kgl exec --shell --timeout 20 "test -s '$MANIFEST' && '$PACKAGE/llama' serve --version"
```

Nếu thiếu GGUF, gắn model qua **Add Input** trong Kaggle. File thử nghiệm khoảng 20,2 GiB và được đọc trực tiếp từ mount; không cần copy sang `/kaggle/working` hay tải lại. Nếu lệnh thứ hai in đúng commit `f4f6f2e`, bỏ qua bước 2. Nếu manifest có nhưng binary không chạy, kiểm tra thư viện thiếu bằng `ldd` trên Kaggle trước khi quyết định build lại.

## 2. Build khi thiếu binary tương thích

Phiên đã thử có `git`, `cmake`, GCC 13.3, CUDA toolkit 13.0.88 và 4 CPU. Kiểm tra công cụ và đĩa tạm trước khi bắt đầu.

```bash
kgl exec --shell --timeout 20 'command -v git && command -v cmake && command -v nvcc && nproc && df -h /tmp && test -r /usr/local/nvidia/lib64/libcuda.so'
```

Lần chạy `kgl model install llama` trên ảnh Kaggle này dừng ở bước tạo cấu hình CMake với lỗi `Target "ggml-cuda" links to CUDA::cuda_driver but the target was not found`. File driver đã có trong `/usr/local/nvidia/lib64`. Dùng trình cài của repo trong một tiến trình Kaggle riêng với `CMAKE_LIBRARY_PATH` trỏ tới thư mục đó. Bước này không sửa toolkit hoặc cài thêm gì vào môi trường local.

```bash
kgl upload src/kgl/vm/llama_tools.py /tmp/kgl/llama-tools-smoke.py
kgl exec --shell --timeout 3600 <<'SH'
set -e
export CMAKE_LIBRARY_PATH="/usr/local/nvidia/lib64${CMAKE_LIBRARY_PATH:+:$CMAKE_LIBRARY_PATH}"
python -u - <<'PY'
import json
import runpy

tools = runpy.run_path("/tmp/kgl/llama-tools-smoke.py")
result = tools["install"](
    "f4f6f2e7db76e98c95cc9b7fc608ea669c806044",
    "/tmp/kgl/llama",
    jobs=4,
)
print(json.dumps(result, indent=2))
PY
SH
```

Trình cài dùng `GGML_CUDA=ON`, `CMAKE_CUDA_ARCHITECTURES=75` và build Release. Với commit này, nó chọn target `llama-app` và xác nhận lệnh `llama serve` trước khi ghi manifest. Dùng lại đường dẫn `MANIFEST` từ bước 1 khi khởi động server.

**Nếu local hết thời gian chờ:** trong lần thử, timeout 1800 giây xảy ra sau khi thư viện CUDA đã link xong. Build vẫn chạy trên Kaggle và manifest xuất hiện sau khoảng 2108 giây kể từ lúc bắt đầu. Đợi notebook xử lý xong lệnh đang chạy, rồi kiểm tra lại manifest và `llama serve --version` như bước 1 trước khi chạy một build khác. Timeout 3600 giây trong lệnh trên có dư thời gian so với lần đã đo.

### Dùng lại build khi đổi phiên Kaggle

Trong cùng phiên, dùng thẳng thư mục `$PACKAGE`. Để thử dùng lại trên một phiên mới có cùng kiến trúc GPU, hệ điều hành và thư viện CUDA tương thích, lưu **cả thư mục**, gồm binary, các file `.so` và manifest, dưới dạng tar để giữ quyền executable.

```bash
kgl exec --shell --timeout 120 "tar -C '$BUILD_ROOT' -cf /kaggle/working/llama-f4f6f2e-sm75.tar '$REV'"
```

Lưu file tar vào bộ lưu trữ riêng hoặc một Kaggle dataset trước khi kết thúc phiên. Ở phiên mới, gắn tar qua **Add Input**, đặt `BUILD_ROOT` và `MANIFEST` như ở bước 1, rồi thay `TAR_INPUT` bằng đường dẫn thật trên VM.

```bash
TAR_INPUT=/kaggle/input/your-build-dataset/llama-f4f6f2e-sm75.tar
kgl exec --shell --timeout 120 "test -r '$TAR_INPUT' && test ! -e '$PACKAGE' && mkdir -p '$BUILD_ROOT' && tar -C '$BUILD_ROOT' -xf '$TAR_INPUT'"
kgl exec --shell --timeout 20 "test -s '$MANIFEST' && '$PACKAGE/llama' serve --version"
```

Nếu binary chạy và in đúng commit, chuyển sang bước 3. Tiếp tục dùng GGUF từ mount riêng của model.

## 3. Nạp model và chạy hai request

Dùng notebook thử riêng. Chạy `kgl status` trước để kiểm tra phiên local chưa ghi nhận một server khác cần giữ lại. Có thể dùng một `KGL_HOME` riêng đã kết nối cho ca này. `model plan` giúp xác nhận tên file, kiến trúc và VRAM trước khi nạp model.

```bash
kgl model plan --model "$GGUF" --ctx 4096
TEST_PORT=$(kgl exec --timeout 15 'import socket
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])')

kgl serve start --backend llama --port "$TEST_PORT" \
  --model "$GGUF" --preset examples/qwen35b-llama/qwen3.8-35b-a3b.json --llama-manifest "$MANIFEST" \
  --ctx 4096 --tensor-split 1,1 --require-gpu --wait-timeout 360
kgl serve status
```

Đạt bước nạp khi `state` là `ready`, `health.n_ctx` là `4096`, `health.devices` có cả `cuda:0` và `cuda:1`, và `health.gpu_memory` ghi cùng PID trên hai card. Khi lỗi hoặc nạp quá lâu, xem `kgl serve logs --lines 50` và log llama.cpp ở đường dẫn `health.llama_log` nếu có. Có thể dùng `kgl model plan --model "$GGUF" --ctx 4096 --log "$LLAMA_LOG"` với đường dẫn log trên VM để đọc số buffer khi phiên bản llama.cpp ghi chúng.

```bash
PROMPT='What is the capital of France? Answer with the city name.'
kgl infer "$PROMPT" --no-stream --json --temperature 0 --max-tokens 512 --timeout 120
kgl infer "$PROMPT" --json --temperature 0 --max-tokens 512 --timeout 120
kgl serve stop
kgl exec --shell --timeout 20 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader'
```

Cả hai phản hồi cần có `text` chứa `Paris` và `finish_reason: stop`. Nếu output chỉ có reasoning trước khi hết token, tăng `--max-tokens` và chạy lại. Khi so sánh độ trễ, ghi `timings.cache_n` vì request lặp có thể dùng prompt cache. `ttft_first_token_s` tính từ token đầu tiên, gồm reasoning; `ttft_first_content_s` tính tới nội dung câu trả lời đầu tiên.

## Kết quả lần chạy trên Kaggle

| Kiểm tra | Kết quả |
|---|---|
| Build CUDA trên 4 CPU | 2107,91 giây, gói build khoảng 305 MiB |
| VRAM của tiến trình khi ready | GPU 0 dùng 10.568 MiB, GPU 1 dùng 10.018 MiB |
| Không streaming | `Paris`, 2,53 giây |
| Streaming | `Paris`, 2,03 giây, 1,85 giây tới nội dung đầu tiên, dùng lại 19 token prompt cache |
| Sau khi dừng server | Hai GPU về 0 MiB sử dụng |

Health báo `offloaded_layers: null`, nên số lớp offload chưa được xác nhận. Phép thử này kiểm chứng cấu hình context 4096 với prompt ngắn. Context dài, tải kéo dài và khôi phục gói tar trên một phiên mới cần phép thử riêng.

## Thử context theo nhu cầu

Dùng cùng GGUF Q4_K_M, bản build và tensor split `1,1` ở các mốc **4096 → 8192 → 16384 → 32768 → 65536**. Chỉ tăng khi hội thoại hoặc tài liệu thực tế cần thêm chỗ. 65536 là giới hạn của kế hoạch thử. Mốc đã chạy thành công trong tài liệu này là 4096.

1. Lưu kết quả ở mốc hiện tại. Khi có thể gián đoạn chat, dừng proxy và chạy `kgl serve stop` để giải phóng trọng số trước khi đo lại bằng `model plan`. Nếu đo khi model cũ còn chạy, planner sẽ tính khả năng nạp thêm một bản model vào VRAM còn trống.
2. Đặt `CTX=8192` cho lần tăng đầu tiên. Dùng `--ctx "$CTX"` trong cả lệnh plan và start ở bước 3. Xác nhận `health.n_ctx` đúng mốc mới, rồi mở lại proxy để demo kết nối đúng instance.
3. Chạy lại một prompt ngắn làm mốc. Tiếp theo, đưa một mã nhận dạng vào đầu hội thoại, gửi đủ văn bản để vượt cửa sổ trước đó rồi yêu cầu model nhắc lại mã. Chừa chỗ cho câu trả lời và kiểm tra số token đầu vào từ `usage.prompt_tokens` của request không streaming khi cần xác nhận độ dài.
4. Ghi độ trễ tới nội dung đầu tiên, tốc độ sinh token, VRAM từng card trong lúc xử lý prompt dài và kết quả nhớ mã. Nếu gặp OOM hoặc độ trễ vượt mức chấp nhận được, khởi động lại ở mốc gần nhất đã đạt.

Kiểm tra câu trả lời cố định của model đang chạy từ thư mục gốc repo bằng `python examples/qwen35b-llama/check_answer.py`.
