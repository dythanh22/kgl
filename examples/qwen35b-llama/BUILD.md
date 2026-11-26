# Build và khôi phục llama.cpp cho Qwen trên Kaggle

Chạy các lệnh Bash từ gốc repo, trong `myenv`, với `KGL_HOME` của notebook đích. Bản build dùng hằng ngày được ghim trong [preset](qwen3.8-35b-a3b.json). Dùng tài liệu này khi cần kiểm binary, khôi phục riêng bản build hoặc chuẩn bị dataset build mới.

## 1. Kiểm tra phiên, GGUF và build sẵn có

```bash
# Dùng môi trường và KGL_HOME của phiên đã kết nối.
kgl exec --timeout 15 'print(2)'
```

Nếu kernel đã mất, chạy `kgl connect` và dán VS Code Compatible URL vào lời nhắc ẩn. Chạy lại `print(2)` trước khi tiếp tục.

```bash
PRESET=examples/qwen35b-llama/qwen3.8-35b-a3b.json
mapfile -t qwen < <(python - "$PRESET" <<'PYTHON'
import sys

from kgl.client.model_fetch import INSTALL_DIR, read_preset

preset = read_preset(sys.argv[1])
print(preset["build"]["revision"])
print(INSTALL_DIR)
print(preset["inputs"][preset["default_quant"]])
PYTHON
)
REV=${qwen[0]}
BUILD_ROOT=${qwen[1]}
GGUF=${qwen[2]}
PACKAGE="$BUILD_ROOT/$REV"
MANIFEST="$PACKAGE/manifest.json"

kgl exec --shell --timeout 20 "test -r '$GGUF' && ls -lh '$GGUF' && nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader"
kgl exec --shell --timeout 20 "test -s '$MANIFEST' && '$PACKAGE/llama' serve --version"
```

Nếu thiếu GGUF, gắn model qua **Add Input** trong Kaggle. File thử nghiệm khoảng 20,2 GiB và được đọc trực tiếp từ mount; không cần copy sang `/kaggle/working` hay tải lại. Khi lệnh kiểm build in đúng commit ghim trong `REV`, dùng bản build hiện có. Nếu manifest có nhưng binary không chạy, kiểm tra thư viện thiếu bằng `ldd` trên Kaggle trước khi quyết định build lại.

### Khôi phục từ Dataset trên phiên Kaggle mới

Với phiên Kaggle mới, dùng [hướng dẫn vận hành](QWEN35B-KAGGLE.md) để chuẩn bị hai T4 và gắn GGUF. Preset ghim dataset build riêng tư và version. Notebook dùng `kagglehub` để đọc đúng version. Nếu chưa có bản build, `ready.py` chạy cả khôi phục và nạp model.

Để chỉ chạy bước chuẩn bị và khôi phục, chạy lệnh dưới bằng cùng Python của `myenv` và `KGL_HOME`. Lệnh dùng chung hàm kiểm manifest, SM75 và binary của `ready.py`. Nó in cổng phục vụ đã ghi hoặc một cổng trống để kiểm tra. [Mã khôi phục](restore_worker.py)

```bash
python - <<'PYTHON'
import subprocess
import sys
from pathlib import Path

from kgl.client.model_fetch import INSTALL_DIR, read_preset
from kgl.core.config import Settings

folder = Path("examples/qwen35b-llama")
preset = read_preset(folder / "qwen3.8-35b-a3b.json")
quant = preset["default_quant"]
build = preset["build"]
source = (folder / "restore_worker.py").read_text(encoding="utf-8")
arguments = (preset["inputs"][quant], str(Path(INSTALL_DIR) / build["revision"]),
             Settings.from_env().workdir)
call = ", ".join(f"Path({path!r})" for path in arguments)
source += f"\nprint(prepare({call}, {build['dataset']!r}, {quant!r}))"
subprocess.run([sys.executable, "-m", "kgl", "exec", "--timeout", "180", source], check=True)
PYTHON
```

Nếu thiếu quyền truy cập input hoặc dataset riêng tư, kiểm tra tài khoản và version trong preset. Nếu binary báo thiếu thư viện CUDA, kiểm tra bằng `ldd "$PACKAGE/llama"` trên VM trước khi build lại. Bước 3 là ca thủ công độc lập khi build đã có và chưa chạy worker.

## 2. Build khi thiếu binary tương thích

Phiên đã thử có `git`, `cmake`, GCC 13.3, CUDA toolkit 13.0.88 và 4 CPU. Kiểm tra công cụ và đĩa tạm trước khi bắt đầu.

```bash
kgl exec --shell --timeout 20 'command -v git && command -v cmake && command -v nvcc && nproc && df -h /tmp && test -r /usr/local/nvidia/lib64/libcuda.so'
```

Lần chạy `kgl model install llama` trên ảnh Kaggle này dừng ở bước tạo cấu hình CMake với lỗi `Target "ggml-cuda" links to CUDA::cuda_driver but the target was not found`. File driver đã có trong `/usr/local/nvidia/lib64`. Dùng trình cài của repo trong một tiến trình Kaggle riêng với `CMAKE_LIBRARY_PATH` trỏ tới thư mục đó. Bước này không sửa toolkit hoặc cài thêm gì vào môi trường local.

```bash
kgl upload src/kgl/vm/llama_tools.py /tmp/kgl/llama-tools-smoke.py
kgl exec --shell --timeout 3600 <<SH
set -e
export CMAKE_LIBRARY_PATH="/usr/local/nvidia/lib64\${CMAKE_LIBRARY_PATH:+:\$CMAKE_LIBRARY_PATH}"
python -u - <<'PY'
import json
import runpy

tools = runpy.run_path("/tmp/kgl/llama-tools-smoke.py")
result = tools["install"](
    "$REV",
    "$BUILD_ROOT",
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
kgl exec --shell --timeout 120 "tar -C '$BUILD_ROOT' -cf /kaggle/working/llama-${REV:0:7}-sm75.tar '$REV'"
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

Cả hai phản hồi cần có `text` chứa `Paris` và `finish_reason: stop`. Nếu lượt chạm giới hạn trước khi có nội dung, ghi lần trượt rồi thử lại với giới hạn phù hợp hoặc thử tắt Thinking qua proxy. Khi so sánh độ trễ, ghi `timings.cache_n` vì request lặp có thể dùng prompt cache. `ttft_first_token_s` tính từ token đầu tiên, gồm reasoning; `ttft_first_content_s` tính tới nội dung câu trả lời đầu tiên.

[Kết quả gốc của ca build và thử 4K](../../docs/reports/2026-11-25-qwen-l9-l10.md#kết-quả-gốc-của-ca-build-4k) · [Báo cáo các mức context](../../docs/reports/2026-11-24-qwen-m3-5.md)
