# Triển khai SAM3 API trên máy GPU

**Cập nhật 16/10/2026.** Áp dụng workstation, server nội bộ hoặc GPU cloud có NVIDIA
GPU; không yêu cầu RunPod. Chọn **một** cách: Docker (mục 3) hoặc Python venv
(mục 4). Chạy trên **máy B/GPU**; CVAT và Nuclio CPU nằm trên máy A theo
[hướng dẫn cài mới CVAT](https://github.com/dythanh22/cvat-sam3-custom/blob/sam3-custom/deployment/sam3/README.md).

## 1. Chuẩn bị và lấy source

Cần Linux x86_64, Git, curl, Python 3 (tạo key/config), driver NVIDIA hoạt động (`nvidia-smi`), mạng tải gói và
đủ chỗ cho Torch/CUDA, model cùng cache. Bộ runtime đã pin: Python 3.12,
Torch 2.10.0 + torchvision 0.25.0 CUDA 12.8; source SAM3 pin
`2345a4ad109ac29c569da749c91d84f10dc08c40`. Driver phải tương thích runtime CUDA
này; kiểm thực tế bằng Torch, không chỉ nhìn dòng CUDA của `nvidia-smi`.
GPU/VRAM khác máy đã thử cần smoke trên ảnh đại diện; không có cam kết mọi GPU đều phù hợp.

```bash
nvidia-smi
git clone --branch sam3-custom https://github.com/dythanh22/cvat-sam3-custom.git
cd cvat-sam3-custom
git rev-parse HEAD
# Nếu máy A đã chốt một commit, checkout đúng commit đó trên bản clone mới này.
cd sam3-service
```

Từ đây chạy trong `sam3-service/`. Có thể clone vào thư mục của bạn, không bắt buộc
`/workspace` hoặc `/srv`. Nếu có source/venv/checkpoint sẵn, xác minh phiên bản
trước khi dùng; không copy đè `.env` hoặc tạo venv chồng lên môi trường đang chạy.

## 2. Checkpoint và API key

Checkpoint **không có trong Git hoặc image Docker**. Nhận `sam3.pt` từ bản lưu hợp lệ
hoặc [model chính thức facebook/sam3](https://huggingface.co/facebook/sam3), theo
quyền truy cập model của bạn. Dùng revision đã kiểm
`3c879f39826c281e95690f02c7821c4de09afae7`, không mặc định lấy bản `main` mới nhất.
Đặt vào thư mục `sam3-checkpoint/` dưới service; thư mục này được ignore:

```bash
mkdir -p sam3-checkpoint
# Chuyển file sam3.pt vào sam3-checkpoint/ bằng kênh riêng của bạn.
printf '%s\n' \
  '9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e  sam3-checkpoint/sam3.pt' \
  | sha256sum --check
```

Chỉ tiếp tục khi báo `OK`. Bản đã kiểm có kích thước 3.450.062.241 bytes;
checksum là tiêu chí chính. Nếu dùng checkpoint khác, cần đánh giá lại và thay
revision/checksum có chủ đích; không tắt kiểm để bỏ qua file sai.

**Chỉ lần đầu**, tạo env và key ngẫu nhiên mà không in key ra terminal:

```bash
cp .env.example .env
chmod 600 .env
python3 - <<'PY'
from pathlib import Path
import secrets
p = Path('.env')
s = p.read_text()
assert 'SAM3_API_KEY=\n' in s, 'File đã có key; không tự ghi đè'
p.write_text(s.replace('SAM3_API_KEY=\n', 'SAM3_API_KEY=' + secrets.token_hex(32) + '\n', 1))
p.chmod(0o600)
PY
```

Nếu đang nhận key từ quản trị viên hoặc máy A đã có key, điền **key đã thống nhất**
bằng editor thay cho tạo key mới. Đặt `SAM3_AUTH_DISABLED=false`, giữ checksum,
revision và một worker. Key máy B phải được quản trị viên nhập vào
`SAM3_API_TOKEN` trong env máy A qua kênh riêng; không phải GitHub token/CVAT password.

## 3. Cách A — Docker trên máy có Docker daemon

Cần Docker Engine/Compose và [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
đã cấu hình cho Docker. Việc cài driver/runtime do quản trị máy thực hiện, không
chạy restart Docker trên host dùng chung mà chưa phối hợp. Kiểm GPU trong container:

```bash
docker info >/dev/null
docker compose version
docker run --rm --gpus all nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 nvidia-smi
```

Sửa `.env` bằng editor, **thay đường dẫn ví dụ bằng đường dẫn tuyệt đối trên máy B**:

```dotenv
SAM3_WEIGHTS_DIR=/DUONG_DAN_THAT/cvat-sam3-custom/sam3-service/sam3-checkpoint
SAM3_CHECKPOINT_PATH=/weights/sam3.pt
SAM3_BIND_ADDRESS=127.0.0.1
SAM3_PUBLISH_PORT=8000
SAM3_IMAGE_TAG=custom-v2
```

Lấy đường dẫn thật bằng `pwd` rồi thêm `/sam3-checkpoint`. `SAM3_WEIGHTS_DIR` là
thư mục trên host; `/weights/sam3.pt` là đường dẫn **trong container**, không đổi
thành đường dẫn host. File/thư mục phải đọc được bởi user UID 10001 của container.

```bash
docker compose config --quiet
docker compose build
docker compose up -d
docker compose ps
docker compose logs --tail=80 sam3
docker compose exec -T sam3 python -c \
  'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'
curl --fail --silent --show-error http://127.0.0.1:8000/readyz
```

Image tự đặt `SAM3_SOURCE_SHA`/offline runtime; weights mount chỉ đọc, API một worker
ở port 8000 trong container. Port host mặc định 8000 chỉ bind loopback; bước 5 sẽ
công bố HTTPS. Model nạp lần đầu có thể mất vài phút, `/readyz` trả 503 khi loading.
`restart: unless-stopped` áp dụng cho container API; không cần `nohup`/venv host.
Nếu đã chọn Docker thì bỏ qua mục 4.

## 4. Cách B — Python venv, không cần Docker daemon

Cần **Python 3.12 kèm venv/pip**, Git, driver NVIDIA và thư viện hệ thống cần cho
runtime. Cách này phù hợp cả máy vật lý lẫn container GPU cloud không chạy được
Docker lồng nhau. Không cài gói vào Python hệ thống hoặc dùng venv từ máy khác.
Pin gói dưới đây khớp Dockerfile; không tự đổi Torch sang CPU nếu CUDA lỗi.

```bash
python3.12 --version
python3.12 -m venv .venv
.venv/bin/python - <<'PY'
from pathlib import Path
import tomllib
lock = tomllib.loads(Path('uv.lock').read_text())
pins = [f"{item['name']}=={item['version']}" for item in lock['package']]
pins += ['numpy==1.26.4', 'scipy==1.13.1', 'scikit-image==0.24.0',
         'torch==2.10.0', 'torchvision==0.25.0', 'setuptools<81']
Path('.venv/runtime-constraints.txt').write_text('\n'.join(pins) + '\n')
PY
.venv/bin/python -m pip install 'setuptools<81' wheel
.venv/bin/python -m pip install torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -c .venv/runtime-constraints.txt \
  einops psutil scipy scikit-image pycocotools
.venv/bin/python -m pip install -c .venv/runtime-constraints.txt \
  'git+https://github.com/facebookresearch/sam3.git@2345a4ad109ac29c569da749c91d84f10dc08c40'
.venv/bin/python -m pip install -c .venv/runtime-constraints.txt .
.venv/bin/python -m pip check
```

Nếu Python/venv/Git chưa có, quản trị viên chuẩn bị theo hệ điều hành của máy đích
trước. Gói `sam3-service` không tự cài Torch/SAM3 từ `pyproject.toml`, vì chúng cần
đúng GPU/runtime; không bỏ các bước cài Torch và SAM3.

Sửa `.env`, thay `/DUONG_DAN_THAT/...` bằng đường dẫn thật từ `pwd`:

```dotenv
SAM3_CHECKPOINT_PATH=/DUONG_DAN_THAT/cvat-sam3-custom/sam3-service/sam3-checkpoint/sam3.pt
SAM3_SOURCE_SHA=2345a4ad109ac29c569da749c91d84f10dc08c40
```

Không dùng `/weights/sam3.pt` của Docker khi chạy venv. Kiểm import/CUDA/contract:

```bash
.venv/bin/python - <<'PY'
import torch, torchvision, sam3
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3_service.schemas import PredictResponse
assert torch.cuda.is_available(), 'Torch chưa truy cập được CUDA'
assert 'batch' in PredictResponse.model_fields, 'API source chưa có contract batch'
print('Torch:', torch.__version__, 'torchvision:', torchvision.__version__)
print('GPU:', torch.cuda.get_device_name(0))
PY
```

Tạo launcher **cục bộ** trong thư mục service, không cần đường dẫn máy tác giả:

```bash
umask 077
cat > run-api.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail
sam3_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$sam3_dir"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
exec "$sam3_dir/.venv/bin/python" -m uvicorn sam3_service.app:create_app --factory \
  --env-file "$sam3_dir/.env" \
  --host "${SAM3_LISTEN_HOST:-127.0.0.1}" --port "${SAM3_LISTEN_PORT:-8000}" \
  --workers 1 --timeout-keep-alive 5 --no-server-header
SH
chmod 700 run-api.sh
```

Chạy foreground để thấy lỗi khởi động, mặc định ở `127.0.0.1:8000`:

```bash
bash ./run-api.sh
```

Mở terminal GPU thứ hai, vào đúng service và kiểm:

```bash
curl --fail --silent --show-error http://127.0.0.1:8000/readyz
```

Sau khi thử xong có thể Ctrl+C để dừng foreground. Nếu cần chạy qua phiên SSH,
trước hết kiểm chưa có process API/cổng đang nạp rồi mới chạy nền:

```bash
umask 077
nohup bash ./run-api.sh > api.log 2>&1 < /dev/null &
echo "$!" > api.pid
chmod 600 api.pid api.log
tail -n 30 api.log
```

`nohup` không tự khởi động sau reboot/crash; môi trường lâu dài nên để quản trị viên
quản lý launcher bằng service manager, hoặc chọn Docker ở mục 3.
`SAM3_LISTEN_HOST`/`SAM3_LISTEN_PORT` là biến **shell của launcher**, không đọc từ
`.env` bởi Uvicorn; nếu đổi chúng, đặt trước lệnh chạy như mục 6.

## 5. Công bố API và kiểm từ máy A

Cách mặc định: giữ API loopback và đặt TLS reverse proxy trên **máy B**. Ví dụ nếu
đã có Caddy và domain `sam3.example.org` trỏ tới máy B, cấu hình Caddyfile:

```caddyfile
sam3.example.org {
    reverse_proxy 127.0.0.1:8000 {
        transport http {
            response_header_timeout 90s
        }
    }
}
```

Thay domain/port thật; để Caddy cấp chứng chỉ cần DNS và đường vào 80/443 phù hợp.
Caddy chạy trên host ở ví dụ này; Caddy trong container khác không dùng loopback
để trỏ host. Quản trị viên cài/kích hoạt Caddy theo [tài liệu Caddy](https://caddyserver.com/docs/caddyfile/directives/reverse_proxy).
Có thể dùng nginx hoặc proxy HTTPS sẵn có của nhà cung cấp thay cho Caddy.
Timeout proxy phải lớn hơn timeout request service (60 giây mặc định); không cắt
request đang chờ sớm hơn. Nếu máy A/B cùng máy và CVAT Traefik đã dùng 80/443,
cần một front proxy chung hoặc cấu hình route API riêng; không bật thêm Caddy
cùng chiếm 80/443. Hai máy riêng có thể dùng 80/443 độc lập.

**Máy A**, dùng URL thật:

```bash
curl --fail --silent --show-error --max-time 20 \
  -A 'SAM3-CVAT/0.1' https://sam3.example.org/healthz
curl --fail --silent --show-error --max-time 20 \
  -A 'SAM3-CVAT/0.1' https://sam3.example.org/readyz
```

`/healthz` chỉ báo process sống; **`/readyz` mới báo model đã nạp**. Thử `/demo`
qua HTTPS bằng API key, upload ảnh và Run một bbox. Demo không ghi nhãn CVAT.
Sau đó quay lại [mục nối adapter](https://github.com/dythanh22/cvat-sam3-custom/blob/sam3-custom/deployment/sam3/README.md#6-đăng-ký-hai-adapter-và-nối-api).
Nếu dùng HTTP trong mạng riêng tin cậy, máy A cần `register-functions.py --allow-http`
và API phải bind/publish vào địa chỉ mà container Nuclio truy cập được.

## 6. Ví dụ tùy chọn — RunPod hoặc GPU cloud có HTTP proxy

Các mục 1–5 vẫn là quy trình chính. Riêng RunPod:

1. Chọn template có driver/CUDA dùng được, Python 3.12 + venv/Git cho cách B, hoặc
   Docker daemon + NVIDIA runtime nếu chọn cách A. Không mặc định mọi Pod có Docker.
2. Đăng ký **public key của riêng bạn**, dùng SSH user/host/port của server mới và
   private key tương ứng sau `-i`. Không dùng file `.pub` làm identity private key.
   Basic SSH gateway có thể không hỗ trợ SCP/SFTP; dùng Full SSH/TCP hoặc file manager
   theo [hướng dẫn SSH RunPod](https://docs.runpod.io/pods/configuration/use-ssh).
3. Clone vào storage bền vững đã chọn, ví dụ `/workspace/cvat-sam3-custom`, và làm
   mục 2–4. Storage/đường dẫn thực tế tùy template; xem [loại storage RunPod](https://docs.runpod.io/pods/storage/types).
4. Expose HTTP port **9012** nếu chọn port này cho API. Với cách B, thay lệnh nền bằng:

```bash
# Chạy trong thư mục sam3-service, khi chưa có API khác đang chạy.
umask 077
SAM3_LISTEN_HOST=0.0.0.0 SAM3_LISTEN_PORT=9012 \
  nohup bash ./run-api.sh > api.log 2>&1 < /dev/null &
echo "$!" > api.pid
chmod 600 api.pid api.log
curl --fail --silent --show-error http://127.0.0.1:9012/readyz
```

Với Docker, đặt `SAM3_BIND_ADDRESS=0.0.0.0`, `SAM3_PUBLISH_PORT=9012` trong `.env`,
giữ port trong container là 8000, rồi `docker compose up -d` để áp dụng.
Dùng URL HTTPS proxy nhà cung cấp cấp, ví dụ
`https://POD_ID-9012.proxy.runpod.net`, kiểm theo mục 5 và điền vào máy A.
Cổng 8080 không bắt buộc cho SAM3; cổng CVAT trên máy A là một hệ thống khác.
Cloud khác cũng áp dụng nguyên tắc: port publish/bind → proxy HTTPS → URL API gốc.
Khi tạo server mới, kiểm lại storage/checkpoint/venv và cập nhật endpoint máy A.

## 7. Mở lại, cập nhật và dừng API

| Việc | Docker (trong `sam3-service/`) | Venv |
| --- | --- | --- |
| Mở lại | `docker compose up -d` | Chạy lại launcher với đúng host/port sau khi kiểm không có process cũ |
| Xem log | `docker compose logs --tail=80 sam3` | `tail -n 80 api.log` |
| Đổi `.env` | `docker compose up -d --force-recreate sam3` | Dừng process cũ rồi chạy launcher lại |
| Source mới đã review | `docker compose build` rồi `docker compose up -d` | Dừng API; cài lại package như dưới rồi chạy launcher lại |
| Dừng | `docker compose stop sam3` | Foreground: Ctrl+C; chạy nền: kiểm PID rồi TERM |

Cài lại **chỉ package service** nếu dependencies/pins không đổi:

```bash
.venv/bin/python -m pip install --no-deps --force-reinstall .
.venv/bin/python -m pip check
```

Nếu dependencies/pins đổi, áp dụng lại bước cài tương ứng trong mục 4 hoặc tạo môi
trường mới; chép `src/` không cập nhật package đã cài. Không mặc định mang venv từ
máy/template khác sang dùng được.

Kiểm PID trước khi dừng API nền:

```bash
ps -p "$(cat api.pid)" -o pid,args
```

Chỉ khi đúng process Uvicorn `sam3_service.app:create_app` của thư mục này:

```bash
kill -TERM "$(cat api.pid)"
```

Chờ process/cổng đóng trước khi mở lại. Save nhãn và thông báo người đang dùng trước
khi tắt GPU. Không dùng `pkill python` trên host có nhiều công việc.
Sau mỗi restart kiểm readiness và Run → Accept → Save trên task thử ở CVAT.

## 8. Lỗi thường gặp và giới hạn kiểm chứng

| Lỗi | Kiểm tra |
| --- | --- |
| Không thấy GPU trong Docker | Driver host và NVIDIA Container Toolkit; kiểm `docker run --gpus all ... nvidia-smi` |
| `nvidia-smi` được, Torch CUDA false | Bộ wheel CUDA/driver, quyền GPU của process/container; chạy kiểm import ở mục 4 |
| Checkpoint không đọc được | Đúng path host/container? Quyền UID 10001 (Docker)? Checksum đúng? |
| `/readyz` loading/failed | Xem log model load/warm-up; đừng khởi chạy thêm worker |
| HTTPS 502/404 nhưng loopback ready | Domain/proxy upstream, port publish, firewall; cloud proxy cần bind địa chỉ truy cập được |
| 401 khi inference | Key API trùng với token hai adapter; readiness public không kiểm auth |
| 503/504 | Queue đầy, timeout, mạng hoặc GPU OOM; kiểm log, đợi rồi retry chủ động |
| Cổng đã bị chiếm | Xác định process trước; không chạy đồng thời Docker và venv trên cùng port/GPU |

Docker/GPU và venv đã có bằng chứng lịch sử trên một số GPU trong
[baseline](https://github.com/dythanh22/cvat-sam3-custom/blob/sam3-custom/docs-vi/baseline.md). Bộ lệnh venv ở đây dùng Python venv/pip chuẩn,
pin theo Dockerfile; đợt rà tài liệu này không cài lại hàng GB dependency hoặc thuê
GPU để chạy lại. Máy/driver/proxy mới vẫn cần kiểm mục 3/4/5 và nghiệm thu CVAT.
Contract request/response nằm ở [README API](README.md).
