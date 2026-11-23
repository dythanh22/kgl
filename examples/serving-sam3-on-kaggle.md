# Chạy SAM3 trên Kaggle với kgl

Ví dụ này chạy nguyên service FastAPI của `sam3-service` trên GPU Kaggle. Bạn có thể gọi API từ terminal local để tải mask PNG về, hoặc dùng giao diện trình duyệt để xem mask chồng lên ảnh.

```text
Trình duyệt → cầu nối local :8014 → Jupyter qua kgl → SAM3 :8013 trên Kaggle → GPU
```

Các lệnh dưới đây chạy trong Bash trên máy local. Nội dung bên trong `kgl exec --shell` chạy trên VM Kaggle. Dùng cùng `KGL_HOME` cho các lệnh `kgl` và cầu nối demo nếu bạn lưu phiên ở thư mục riêng.

## 1. Kết nối và đưa source lên Kaggle

Trong notebook Kaggle, bật GPU, bật Internet để cài gói và bật **Kaggle Jupyter Server**. Qua **Add Input**, gắn model hoặc dataset chứa checkpoint `sam3.pt`. Quy trình này dùng revision `3c879f39826c281e95690f02c7821c4de09afae7` và SHA-256 có sẵn trong `.env.example`. File checkpoint đã kiểm có dung lượng khoảng 3,45 GB. [Thông tin checkpoint](sam3-service/DEPLOYMENT.md#2-checkpoint-và-api-key)

Từ thư mục gốc repo `kgl`, kích hoạt môi trường local của bạn. Ví dụ này dùng `myenv`. Khi `kgl connect` nhắc nhập URL, dán **VS Code Compatible URL** từ mục **Manually Connect** của Kaggle.

```bash
conda activate myenv
kgl connect
kgl exec --shell 'nvidia-smi && df -h /kaggle/working'

ARCHIVE="$(mktemp /tmp/sam3-service.XXXXXX.tar.gz)"
tar -czf "$ARCHIVE" -C examples/sam3-service \
  src pyproject.toml uv.lock README.md .env.example \
  test-images-seg/traffic-in-vietnam-banner.jpg &&
kgl upload "$ARCHIVE" /kaggle/working/sam3-service.tar.gz &&
kgl exec --shell '
  set -e
  mkdir -p /kaggle/working/sam3-service
  tar -xzf /kaggle/working/sam3-service.tar.gz -C /kaggle/working/sam3-service
'
```

Danh sách file trong archive chỉ gồm source, cấu hình mẫu và ảnh dùng để nghiệm thu. API key được tạo trực tiếp trên Kaggle ở bước 3.

## 2. Tạo môi trường Python 3.12 và cài runtime đã pin

Service dùng NumPy 1.26.4 theo bộ pin đã nghiệm thu. Dùng Python 3.12 riêng trên VM, kể cả khi kernel Kaggle đang chạy Python 3.13. Trên phiên thử, Python 3.12 của hệ thống có executable nhưng thiếu `ensurepip`. `uv venv --managed-python --python 3.12 --seed` xử lý trường hợp này và cài `pip` vào venv. [Bộ pin gốc](sam3-service/DEPLOYMENT.md#4-cách-b--python-venv-không-cần-docker-daemon) · [Python do uv quản lý](https://docs.astral.sh/uv/concepts/python-versions/)

```bash
kgl exec --shell --timeout 600 <<'SH'
set -euo pipefail
cd /kaggle/working/sam3-service

if command -v uv >/dev/null 2>&1; then
  UV="$(command -v uv)"
else
  curl -fsSL https://astral.sh/uv/install.sh \
    | env UV_INSTALL_DIR="$PWD/.uv-bin" UV_NO_MODIFY_PATH=1 sh
  UV="$PWD/.uv-bin/uv"
fi

if [ -d .venv ] && ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
  BACKUP="$(mktemp -d .venv-backup.XXXXXX)"
  mv .venv "$BACKUP/"
fi
if [ ! -e .venv ]; then
  "$UV" venv --managed-python --python 3.12 --seed .venv
fi

.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version; print(sys.version)'
.venv/bin/python -m pip --version
SH
```

Kiểm tra hai dòng cuối in Python 3.12 và `pip` trong `/kaggle/working/sam3-service/.venv`. Sau đó cài dependencies từ `uv.lock`, PyTorch 2.10.0 và torchvision 0.25.0 cho CUDA 12.8, cùng source SAM3 tại commit đã pin.

```bash
kgl exec --shell --timeout 1800 <<'SH'
set -euo pipefail
cd /kaggle/working/sam3-service
PY=.venv/bin/python
C=.venv/runtime-constraints.txt

"$PY" - <<'PYCODE'
from pathlib import Path
import tomllib

lock = tomllib.loads(Path("uv.lock").read_text())
pins = [f"{p['name']}=={p['version']}" for p in lock["package"]]
pins += [
    "numpy==1.26.4", "scipy==1.13.1", "scikit-image==0.24.0",
    "torch==2.10.0", "torchvision==0.25.0", "setuptools<81",
]
Path(".venv/runtime-constraints.txt").write_text("\n".join(pins) + "\n")
PYCODE

"$PY" -m pip install 'setuptools<81' wheel
"$PY" -m pip install torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128
"$PY" -m pip install -c "$C" einops psutil scipy scikit-image pycocotools
"$PY" -m pip install -c "$C" \
  'git+https://github.com/facebookresearch/sam3.git@2345a4ad109ac29c569da749c91d84f10dc08c40'
"$PY" -m pip install -c "$C" .
"$PY" -m pip check

CUDA_VISIBLE_DEVICES=0 "$PY" - <<'PYCODE'
import torch
import torchvision
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3_service.schemas import PredictResponse

assert torch.cuda.is_available(), "Torch không truy cập được CUDA"
assert "batch" in PredictResponse.model_fields
print("Torch:", torch.__version__, "| torchvision:", torchvision.__version__)
print("GPU:", torch.cuda.get_device_name(0))
PYCODE
SH
```

Chuyển bước khi `pip check` thành công, các import chạy được và dòng cuối in tên GPU.

## 3. Khởi động SAM3 trên Kaggle

Tìm đường dẫn checkpoint trong notebook.

```bash
kgl exec --shell 'find -L /kaggle/input -type f -name sam3.pt -print'
```

Thay `CHECKPOINT` trong khối sau bằng kết quả của bạn. Giá trị bên dưới là đường dẫn đã dùng trong lần chạy thử. Lần đầu, lệnh tạo `.env` từ mẫu, giữ revision và checksum, rồi sinh Bearer key. Những lần sau dùng lại cùng key. Service dùng một worker trên GPU 0 và chọn FP32 trên Tesla T4 khi `SAM3_PRECISION=auto`. [Cấu hình mẫu](sam3-service/.env.example)

```bash
kgl exec --shell --timeout 120 <<'SH'
set -euo pipefail
cd /kaggle/working/sam3-service
umask 077

CHECKPOINT=/kaggle/input/models/thanhduy1022/sam3/pytorch/default/1/sam3.pt
test -r "$CHECKPOINT" || { echo "Không đọc được checkpoint: $CHECKPOINT"; exit 1; }

if [ -s api.pid ] && kill -0 "$(cat api.pid)" 2>/dev/null; then
  echo "PID trong api.pid đang chạy. Kiểm tra /readyz trước khi khởi động lại."
  exit 1
fi

.venv/bin/python - "$CHECKPOINT" <<'PYCODE'
from pathlib import Path
import secrets
import sys

from dotenv import dotenv_values

path = Path(".env")
checkpoint = sys.argv[1]
if not path.exists():
    config = Path(".env.example").read_text()
    config = config.replace("SAM3_API_KEY=\n",
                            f"SAM3_API_KEY={secrets.token_hex(32)}\n", 1)
    config = config.replace("SAM3_CHECKPOINT_PATH=/weights/sam3.pt",
                            f"SAM3_CHECKPOINT_PATH={checkpoint}", 1)
    config += "\nSAM3_SOURCE_SHA=2345a4ad109ac29c569da749c91d84f10dc08c40\n"
    path.write_text(config)
    path.chmod(0o600)
elif dotenv_values(path).get("SAM3_CHECKPOINT_PATH") != checkpoint:
    raise SystemExit("Cập nhật SAM3_CHECKPOINT_PATH trong .env cho khớp checkpoint.")
PYCODE

CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 \
  nohup .venv/bin/python -m uvicorn sam3_service.app:create_app --factory \
    --env-file .env --host 127.0.0.1 --port 8013 --workers 1 \
    --timeout-keep-alive 5 --no-server-header \
    > api.log 2>&1 < /dev/null &
echo "$!" > api.pid
SH

kgl exec --shell 'curl -sS --max-time 10 -w "\nHTTP %{http_code}\n" http://127.0.0.1:8013/readyz'
```

Đợi `HTTP 200` và `{"status":"ready"}`. Trạng thái `loading` nghĩa là model chưa sẵn sàng. Nếu thấy `failed`, `fatal` hoặc cổng từ chối kết nối, xem log. Nếu SHA-256 không khớp, kiểm lại file và revision đã gắn.

```bash
kgl exec --shell 'tail -n 60 /kaggle/working/sam3-service/api.log'
```

## 4. Gọi API và tải mask về từ terminal

Chạy hai request trên ảnh mẫu đã upload. Tọa độ điểm và thông số concept lấy từ hai fixture GPU của service. Mỗi ca kiểm tra CUDA và kích thước PNG trước khi lưu mask đầu tiên. [Fixture instance](sam3-service/tests/fixtures/p1-offline-20261008/cases/instance_point.json) · [Fixture concept](sam3-service/tests/fixtures/p1-offline-20261008/cases/concept_car.json)

```bash
kgl exec --shell --timeout 180 <<'SH'
set -euo pipefail
cd /kaggle/working/sam3-service

.venv/bin/python - <<'PYCODE'
import base64
import io
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from dotenv import dotenv_values
from PIL import Image

image = Path("test-images-seg/traffic-in-vietnam-banner.jpg").read_bytes()
key = dotenv_values(".env")["SAM3_API_KEY"]
common = {"schema_version": "1", "image_base64": base64.b64encode(image).decode("ascii")}
headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
cases = {
    "instance": {"points": [{"x": 1250, "y": 820, "label": 1}]},
    "concept": {"text": "car", "threshold": 0.4, "max_instances": 10},
}

for mode, prompt in cases.items():
    request = Request(
        "http://127.0.0.1:8013/v1/predict",
        data=json.dumps({**common, "mode": mode, **prompt}).encode(),
        headers=headers,
    )
    try:
        with urlopen(request, timeout=75) as response:
            result = json.load(response)
    except HTTPError as exc:
        raise SystemExit(f"{mode}: HTTP {exc.code}: {exc.read().decode()}") from None

    assert result["mode"] == mode and result["model"]["device"] == "cuda"
    instances = result["instances"]
    if not instances:
        raise SystemExit(f"{mode}: HTTP 200 nhưng fixture không có mask")
    mask = base64.b64decode(instances[0]["mask_png_base64"], validate=True)
    with Image.open(io.BytesIO(mask)) as png:
        assert png.format == "PNG"
        assert png.size == (result["image"]["width"], result["image"]["height"])
        png.verify()
    Path(f"{mode}-mask.png").write_bytes(mask)
    print(f"{mode}: {len(instances)} mask | CUDA | {result['timing']['inference_ms']:.0f} ms")
PYCODE
SH

kgl download /kaggle/working/sam3-service/instance-mask.png ./instance-mask.png
kgl download /kaggle/working/sam3-service/concept-mask.png ./concept-mask.png
```

Chạy hai lệnh tải xuống sau khi cả hai ca in số mask thành công.

## 5. Mở demo local và nhập API key

Giữ Uvicorn chạy trên Kaggle. Trong một terminal local tại thư mục gốc repo, khởi động cầu nối:

```bash
KGL_REQUEST_BYTES=33554432 kgl forward --remote-port 8013 --listen 127.0.0.1:8014
```

Mở [http://127.0.0.1:8014/demo](http://127.0.0.1:8014/demo). Cầu nối phục vụ trang demo từ cùng origin và chuyển `/healthz`, `/readyz` và `/v1/predict` qua Jupyter. Dùng `--listen` để đổi địa chỉ local và `--remote-port` nếu Uvicorn nghe cổng khác.

Trong một terminal local khác dùng cùng phiên `kgl`, chạy đoạn sau để lấy key và chép vào clipboard. Nếu thiếu công cụ clipboard, đoạn lệnh in key để bạn sao chép trực tiếp vào demo.

```bash
if SAM3_DEMO_KEY="$(kgl exec --shell --timeout 20 <<'SH'
set -e
cd /kaggle/working/sam3-service
.venv/bin/python - <<'PYCODE'
from dotenv import dotenv_values

key = dotenv_values(".env").get("SAM3_API_KEY")
if not key:
    raise SystemExit("Không tìm thấy SAM3_API_KEY trong .env")
print(key)
PYCODE
SH
)"; then
  if [ -n "${WAYLAND_DISPLAY:-}" ] && command -v wl-copy >/dev/null; then
    printf '%s' "$SAM3_DEMO_KEY" | wl-copy
  elif command -v xclip >/dev/null; then
    printf '%s' "$SAM3_DEMO_KEY" | xclip -selection clipboard
  else
    printf '%s\n' "$SAM3_DEMO_KEY"
  fi
fi
unset SAM3_DEMO_KEY
```

Dán vào ô **Bearer API key**, chọn `examples/sam3-service/test-images-seg/traffic-in-vietnam-banner.jpg`, rồi thử từng chế độ:

- **Instance**. Đặt một điểm dương lên chiếc xe hoặc vẽ box, rồi bấm **Chạy instance**.
- **Concept**. Nhập `car`, đặt threshold `0.4` và số instance tối đa `10`, rồi bấm **Chạy concept**.

Trong lần thử bằng Firefox headless qua Kaggle, giao diện báo **Sẵn sàng**, ca point trả **1 mask** và ca concept trả **10 mask**. Cả hai phản hồi ghi thiết bị CUDA. Gửi Bearer key sai nhận HTTP 401. Ảnh lớn và nhiều mask có thể mất thêm thời gian do cầu nối truyền request và response theo chunk.

## 6. Dừng sau khi dùng xong

Nhấn Ctrl+C ở terminal cầu nối local. Để dừng Uvicorn trên Kaggle, xác nhận PID trong `api.pid` thuộc đúng tiến trình SAM3 trước khi gửi TERM:

```bash
kgl exec --shell '
  set -e
  cd /kaggle/working/sam3-service
  pid=$(cat api.pid)
  cmd=$(ps -p "$pid" -o args=)
  case "$cmd" in
    *"-m uvicorn sam3_service.app:create_app"* ) kill -TERM "$pid" ;;
    * ) echo "PID trong api.pid không khớp Uvicorn SAM3"; exit 1 ;;
  esac
'
```

Dừng notebook trong giao diện Kaggle khi không còn dùng GPU. Demo phụ thuộc vào phiên notebook đang kết nối. [Triển khai SAM3 trên máy GPU riêng](sam3-service/DEPLOYMENT.md)
