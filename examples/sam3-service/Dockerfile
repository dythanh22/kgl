# syntax=docker/dockerfile:1
#
# Image chạy trên máy B (GPU). Build context = thư mục sam3-service/.
# Checkpoint KHÔNG nằm trong image: mount chỉ đọc vào /weights.
#
# Bộ CUDA/Torch này đã được Docker build và GPU smoke-test trên Tesla T4; xem
# docs-vi/baseline.md. Rebuild trên máy đích vẫn phải xác minh driver/NVIDIA runtime.

ARG CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

# ---------------------------------------------------------------- builder
FROM ${CUDA_IMAGE} AS builder

# SHA source SAM3 đã pin và smoke-test; notebook Kaggle cũ dùng một commit khác.
ARG SAM3_SHA=2345a4ad109ac29c569da749c91d84f10dc08c40
ARG TORCH_VERSION=2.10.0
ARG TORCHVISION_VERSION=0.25.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3.12 python3.12-venv git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# Pin dependency Python khai báo cho service theo uv.lock. Torch/CUDA và một số
# runtime extras của SAM3 được cài riêng bên dưới; các gói có ràng buộc chốt rõ.
COPY uv.lock /tmp/uv.lock
RUN python -c 'import tomllib; data = tomllib.load(open("/tmp/uv.lock", "rb")); \
open("/tmp/uv-constraints.txt", "w").write("\n".join( \
    "{}=={}".format(p["name"], p["version"]) for p in data["package"]))'

# SAM3 import `pkg_resources` -> cần setuptools < 81.
RUN pip install "setuptools<81" wheel

RUN pip install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
        --index-url "${TORCH_INDEX_URL}"

# Ràng buộc: giữ numpy < 2 và không để pip đổi torch.
RUN printf '%s\n' \
      "numpy==1.26.4" "scipy==1.13.1" "scikit-image==0.24.0" \
      "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
      "setuptools<81" > /tmp/constraints.txt \
 && cat /tmp/uv-constraints.txt >> /tmp/constraints.txt

# Ngoài dependency khai báo của SAM3, các import thực tế ở đường inference cần
# thêm: einops (sam/rope.py), psutil (sam3_video_predictor, được nạp bởi
# model_builder), scipy, scikit-image (hole-filling ở instance mode khi không có
# cc_torch). Không cài extras notebook/train.
RUN pip install -c /tmp/constraints.txt einops psutil scipy scikit-image pycocotools \
 && pip install -c /tmp/constraints.txt "git+https://github.com/facebookresearch/sam3.git@${SAM3_SHA}"

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install -c /tmp/constraints.txt .

# Kiểm import ngay lúc build để thiếu dependency lộ ra sớm (không cần GPU).
RUN python -c "import sam3; \
from sam3.model_builder import build_sam3_image_model; \
from sam3.model.sam3_image_processor import Sam3Processor; \
import sam3_service.app; print('import ok')"

# ---------------------------------------------------------------- runtime
FROM ${CUDA_IMAGE} AS runtime
ARG SAM3_SHA=2345a4ad109ac29c569da749c91d84f10dc08c40

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3.12 \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --system --uid 10001 --home-dir /tmp --shell /usr/sbin/nologin sam3

COPY --from=builder /opt/venv /opt/venv

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp/.cache \
    HF_HUB_OFFLINE=1 \
    SAM3_CHECKPOINT_PATH=/weights/sam3.pt \
    SAM3_SOURCE_SHA=${SAM3_SHA}

USER 10001
EXPOSE 8000

# readyz trả 503 trong lúc nạp model -> cho start-period dài.
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=4)"

# MỘT worker: mỗi worker nhân bản model lên GPU.
CMD ["uvicorn", "sam3_service.app:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1", \
     "--timeout-keep-alive", "5", "--no-server-header"]
