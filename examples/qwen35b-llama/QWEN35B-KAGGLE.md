# Chạy Qwen3.8-35B-A3B Q4_K_M trên hai Tesla T4 của Kaggle

Mở terminal Bash ở thư mục gốc repo. llama.cpp được build và lưu trên Kaggle. Nếu phiên Kaggle còn thư mục build của đúng commit bên dưới, kiểm tra binary rồi chuyển thẳng đến bước thử model. Lần build mới trên 4 CPU mất khoảng 35 phút.

## Đưa worker về ready bằng một lệnh

Trong `myenv`, tại gốc repo, dùng cùng `KGL_HOME` đã kết nối với notebook. Gắn GGUF Q4_K_M và chọn GPU T4 ×2, rồi chạy:

```bash
python examples/qwen35b-llama/ready.py
```

Lệnh khôi phục dataset build ghim khi thiếu, kiểm tra manifest và binary, rồi gọi `kgl serve start` với cấu hình ở bước 3. Nếu đã có worker, kgl so khớp cấu hình trên cổng đã ghi và báo xung đột khi cấu hình khác. Dòng `Qwen ready` hiển thị instance, PID và cổng nội bộ sau khi xác minh context và cả hai GPU. Nếu phiên hết hạn, chạy `kgl connect` rồi thử lại.

### Mở chat và API ở cổng 8001

Sau khi worker đã ready, dùng cùng môi trường và session store trên máy Linux:

```bash
python examples/qwen35b-llama/proxy.py
```

Mở [chat local](http://127.0.0.1:8001/) hoặc cấu hình Base URL `http://127.0.0.1:8001/v1` cho script. Lệnh khởi chạy proxy ở nền, xác nhận instance qua `/ready` và in PID. Chạy lại sau khi đổi phiên: lệnh dùng lại proxy đúng đích hoặc kiểm tra worker mới trước khi thay proxy cũ. Nếu lệnh báo bận, chờ lượt đang chạy xong rồi thử lại.

Lệnh dùng `ss` và `/proc` để nhận diện proxy của cùng tài khoản, Python, thư mục chat và `KGL_HOME`. Hỗ trợ cả proxy cũ khởi chạy theo ví dụ `python -m kgl serve proxy --static examples/chat-demo`. Cổng do ứng dụng khác giữ sẽ được báo xung đột. Nếu đặt `KGL_PROXY_KEY`, dùng cùng key khi quản lý lại proxy. Log của proxy do lệnh khởi chạy nằm trong `qwen-proxy.log` ở session store.

Ví dụ curl và Python cho tác vụ trích xuất đầu việc, bearer key và xử lý lỗi nằm trong [hướng dẫn API](API.md).

### Nghiệm thu qua một lần đổi VM

1. Trên VM cũ, ghi instance từ `kgl serve status`. Trong chat tại `http://127.0.0.1:8001/`, gửi một mã thử riêng cho lần đo và giữ lại hội thoại trong cùng hồ sơ trình duyệt. Chạy `python examples/agent/agent_infer.py "What is 2+2? Reply with only the digit."` và ghi câu trả lời.
2. Trên VM mới có hai T4 và input GGUF, chạy `kgl connect` trong cùng `myenv` và `KGL_HOME`, dán URL của phiên mới tại lời nhắc ẩn. Kiểm tra `kgl exec --timeout 15 'print(2)'`. Trước khi chạy ready, kiểm tra thư mục build của commit ở bước 1 bên dưới để ghi nhận build đã có hay cần khôi phục.
3. Chạy `python examples/qwen35b-llama/ready.py`, ghi thời gian và instance. Chạy `python examples/qwen35b-llama/proxy.py` rồi xác nhận `/ready` trên cổng 8001 trả về instance mới.
4. Mở lại đúng hội thoại trong cùng hồ sơ trình duyệt, chọn gửi đủ lịch sử và yêu cầu nhắc lại mã thử. Chạy lại nguyên lệnh script ở mục 1. Ghi kết quả GUI và script riêng trong báo cáo. Nếu VM cũ đã dừng trước khi tạo hội thoại thử, ghi rõ phần bằng chứng còn thiếu.

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

### Khôi phục từ Dataset trên phiên Kaggle mới

Chọn accelerator **GPU T4 ×2**, gắn model GGUF ở bước 1 qua **Add Input**, rồi kết nối `kgl` với URL của phiên mới từ môi trường `myenv`. Bản build CUDA cho T4 nằm trong dataset riêng tư của tài khoản `thanhduy1022`. Lệnh dưới đây gắn đúng phiên bản bằng `kagglehub` trong notebook. [Dataset llama.cpp SM75, phiên bản 1](https://www.kaggle.com/datasets/thanhduy1022/llama-cpp-f4f6f2e-sm75-214105c7/versions/1)

Kaggle đã giải nén file tar khi tạo dataset. Thư mục gốc của input chứa thư mục mang tên commit. Từ terminal ở bước 1, chạy:

```bash
kgl exec --timeout 180 <<'PYTHON'
from pathlib import Path
from tempfile import TemporaryDirectory
import shutil
import subprocess
import kagglehub

rev = "f4f6f2e7db76e98c95cc9b7fc608ea669c806044"
package = Path("/tmp/kgl/llama") / rev
if not package.exists():
    dataset = kagglehub.dataset_download(
        "thanhduy1022/llama-cpp-f4f6f2e-sm75-214105c7/versions/1"
    )
    package.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=package.parent) as tmp:
        restored = Path(tmp) / rev
        shutil.copytree(Path(dataset) / rev, restored)
        binary = restored / "llama"
        binary.chmod(binary.stat().st_mode | 0o100)
        subprocess.run([str(binary), "serve", "--version"], check=True)
        restored.rename(package)
PYTHON
kgl exec --shell --timeout 20 "test -s '$MANIFEST' && '$PACKAGE/llama' serve --version"
```

Khi binary in commit `f4f6f2e`, chuyển thẳng sang bước 3, dùng `--llama-manifest "$MANIFEST"` và `--model "$GGUF"`. Để tiếp tục chat sau hai request thử, giữ server đang chạy và chỉ dùng `kgl serve stop` khi muốn kết thúc. Nếu thư mục input hoặc dataset riêng tư không truy cập được, kiểm tra tài khoản và phiên bản dataset. Nếu binary báo thiếu thư viện CUDA, kiểm tra bằng `ldd "$PACKAGE/llama"` trên VM trước khi build lại.

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

### Lưu trữ bản build

Để lưu một bản build mới, đóng gói cả thư mục commit (binary, thư viện `.so` và `manifest.json`). Ví dụ với các biến ở bước 1:

```bash
kgl exec --shell --timeout 120 "tar -C '$BUILD_ROOT' -cf /kaggle/working/llama-f4f6f2e-sm75.tar '$REV'"
```

Đưa riêng file tar vào một Kaggle Dataset riêng tư. Bản dùng trong hướng dẫn đã được lưu ở phiên bản 1. File tar gốc có SHA-256 `214105c7da0501b68bbc49746c7095b67f2510e66ac9a658cf4922c288bb272f`.

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

Health báo `offloaded_layers: null`, nên số lớp offload chưa được xác nhận. Phép thử này kiểm chứng cấu hình context 4096 với prompt ngắn. Đã đối chiếu SHA-256 từng file từ dataset với bản build gốc và chạy binary khôi phục trong thư mục tạm trên cùng VM. Trên VM mới, đã khôi phục build, nạp model và kiểm tra cả hai chế độ inference. Context dài đã được đo riêng ở mục dưới. Tải kéo dài vẫn cần phép thử riêng. [Kết quả khôi phục trên VM mới](../../docs/reports/2026-11-23-qwen-m1.md)

## Thử context theo nhu cầu

**32.768 token là mặc định của `ready.py` cho chat, tóm tắt, biên tập và trích xuất văn bản.** Mốc này xử lý prompt thử 25.459 token, cho nội dung đầu tiên sau 17,23 giây và còn 4.375 MiB trên GPU có ít bộ nhớ trống hơn. Với prompt đối chứng khoảng 2.000 token, thời gian tới nội dung đầu tiên là 2,61 giây. [Bảng số đo context](../../docs/reports/2026-11-24-qwen-m3-5.md)

Khi worker hiện tại đang rảnh và bạn muốn đổi context, chạy lần lượt trong `myenv` với cùng `KGL_HOME`:

```bash
kgl serve stop
python examples/qwen35b-llama/ready.py --ctx 32768
python examples/qwen35b-llama/proxy.py
```

Dùng `--ctx 65536` khi cần cửa sổ lớn hơn, hoặc `--ctx 4096` để trở về cấu hình ban đầu. Mốc 262.144 được ghi lại trong báo cáo benchmark. Khi worker đang chạy ở context khác mặc định, truyền lại cùng giá trị `--ctx` để dùng lại worker. GUI lấy context thực tế qua `/ready`. Client gửi prompt rất dài cần chừa thời gian nhận token đầu tiên và kiểm tra giới hạn kích thước HTTP của proxy.

Kiểm tra câu trả lời cố định của model đang chạy từ thư mục gốc repo bằng `python examples/qwen35b-llama/check_answer.py`.
