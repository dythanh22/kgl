"""Nạp SAM 3 và chạy inference instance/concept.

Module này **không import Torch/SAM3 ở mức module** để test API chạy được mà
không cần GPU. Mọi lời gọi vào lớp ``Sam3Model`` phải diễn ra trong luồng của
``InferenceEngine`` (kể cả ``load``).

Hai đường suy luận (đối chiếu với source SAM3 SHA 2345a4ad…):

* ``concept``: ``Sam3Processor.set_image`` + ``set_text_prompt``. Ngưỡng là
  ``processor.confidence_threshold`` (điểm = sigmoid(logit) * presence).
* ``instance``: ``Sam3Processor.set_image`` + ``model.predict_inst`` (cần
  ``enable_inst_interactivity=True``). Box là XYXY pixel, điểm là XY pixel.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
from PIL import Image

from .config import Settings
from .schemas import ApiError, ModelMeta, ScoreType

log = logging.getLogger("sam3_service.model")

MIN_TORCH = "2.7"

_FATAL_CUDA_MARKERS = (
    "cuda error",
    "device-side assert",
    "illegal memory access",
    "unspecified launch failure",
)


@dataclass
class RawInstance:
    mask: np.ndarray  # bool, HxW, đúng kích thước ảnh đầu vào
    score: float
    score_type: ScoreType
    prompt_id: int | None = None


class SegmentationModel(Protocol):
    """Giao diện mà app dùng; test đưa đối tượng giả thỏa giao diện này vào."""

    @property
    def meta(self) -> ModelMeta: ...

    def load(self) -> None: ...

    def predict_instance(
        self,
        image: Image.Image,
        points: Sequence[tuple[float, float, int]],
        box: tuple[float, float, float, float] | None,
    ) -> list[RawInstance]: ...

    def predict_instances(
        self,
        image: Image.Image,
        boxes: Sequence[tuple[float, float, float, float]],
    ) -> list[RawInstance]: ...

    def predict_concept(
        self, image: Image.Image, text: str, threshold: float, max_instances: int
    ) -> list[RawInstance]: ...

    def close(self) -> None: ...


def sha256_file(path: Path, chunk: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


class Sam3Model:
    NAME = "sam3-image"

    def __init__(self, cfg: Settings) -> None:
        self._cfg = cfg
        self._torch = None
        self._model = None
        self._processor = None
        self._autocast_dtype = None
        self._precision = "unloaded"
        self._verified_sha: str | None = None

    # ------------------------------------------------------------------ meta
    @property
    def meta(self) -> ModelMeta:
        return ModelMeta(
            name=self.NAME,
            source_sha=self._cfg.source_sha,
            checkpoint_revision=self._cfg.checkpoint_revision,
            checkpoint_sha256=self._verified_sha,
            precision=self._precision,
            device=self._cfg.device,
        )

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        cfg = self._cfg
        import torch  # noqa: PLC0415 - import trễ có chủ đích
        from packaging.version import Version  # noqa: PLC0415

        self._torch = torch
        if Version(torch.__version__.split("+")[0]) < Version(MIN_TORCH):
            raise RuntimeError(f"Cần PyTorch >= {MIN_TORCH}, đang có {torch.__version__}")

        ckpt = cfg.checkpoint_path
        if not ckpt.is_file():
            raise FileNotFoundError(f"Không thấy checkpoint: {ckpt}")
        if cfg.checkpoint_sha256:
            log.info("Đang xác minh SHA-256 checkpoint (%s)...", ckpt)
            actual = sha256_file(ckpt)
            if actual.lower() != cfg.checkpoint_sha256.lower():
                raise RuntimeError(
                    f"SHA-256 checkpoint không khớp: mong đợi {cfg.checkpoint_sha256}, thực tế {actual}"
                )
            self._verified_sha = actual.lower()

        use_bf16 = False
        if cfg.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("SAM3_DEVICE=cuda nhưng CUDA không khả dụng trong container")
            torch.cuda.set_device(0)
            major, _ = torch.cuda.get_device_capability(0)
            bf16_ok = major >= 8 and torch.cuda.is_bf16_supported()
            if cfg.precision == "bf16" and not bf16_ok:
                raise RuntimeError("SAM3_PRECISION=bf16 nhưng GPU không hỗ trợ BF16 gốc")
            use_bf16 = cfg.precision == "bf16" or (cfg.precision == "auto" and bf16_ok)
            torch.backends.cuda.matmul.allow_tf32 = major >= 8
            torch.backends.cudnn.allow_tf32 = major >= 8
        elif cfg.precision == "bf16":
            raise RuntimeError("BF16 chỉ hỗ trợ khi SAM3_DEVICE=cuda")
        self._autocast_dtype = torch.bfloat16 if use_bf16 else None
        self._precision = "bf16-autocast" if use_bf16 else "fp32"

        bpe = cfg.bpe_path
        if bpe is None:
            from importlib.resources import files  # noqa: PLC0415

            bpe = Path(str(files("sam3") / "assets" / "bpe_simple_vocab_16e6.txt.gz"))
        if not bpe.is_file():
            raise FileNotFoundError(f"Không thấy file BPE: {bpe}")

        from sam3.model.sam3_image_processor import Sam3Processor  # noqa: PLC0415
        from sam3.model_builder import build_sam3_image_model  # noqa: PLC0415

        log.info("Dựng kiến trúc SAM3 (device=%s, precision=%s)...", cfg.device, self._precision)
        # checkpoint_path=None + load_from_HF=False: không tải/nạp gì ở đây. Ta tự
        # nạp bên dưới để kiểm tra key thiếu (upstream dùng strict=False).
        model = build_sam3_image_model(
            bpe_path=str(bpe),
            device=cfg.device,
            eval_mode=True,
            checkpoint_path=None,
            load_from_HF=False,
            enable_segmentation=True,
            enable_inst_interactivity=True,
            compile=False,
        )
        if getattr(model, "inst_interactive_predictor", None) is None:
            raise RuntimeError("Model không có inst_interactive_predictor (mode instance không dùng được)")
        self._load_weights(model, ckpt)
        model.eval()

        self._model = model
        self._processor = Sam3Processor(
            model, device=cfg.device, confidence_threshold=cfg.default_threshold
        )
        if cfg.warmup:
            self._warmup()
        if cfg.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        log.info("SAM3 sẵn sàng: %s", self.meta.model_dump())

    def _load_weights(self, model, path: Path) -> None:
        torch = self._torch
        with open(path, "rb") as f:
            ckpt = torch.load(f, map_location="cpu", weights_only=True)
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            ckpt = ckpt["model"]
        # Giữ đúng phép ánh xạ key của SAM3 ``_load_checkpoint`` (SHA đã pin).
        state = {k.replace("detector.", ""): v for k, v in ckpt.items() if "detector" in k}
        state.update(
            {
                k.replace("tracker.", "inst_interactive_predictor.model."): v
                for k, v in ckpt.items()
                if "tracker" in k
            }
        )
        if not state:
            raise RuntimeError("Checkpoint không chứa key 'detector.*' nào: sai file checkpoint?")
        missing, unexpected = model.load_state_dict(state, strict=False)
        del ckpt, state
        if unexpected:
            log.warning("Checkpoint có %d key không dùng (mẫu: %s)", len(unexpected), unexpected[:5])
        if len(missing) > self._cfg.max_missing_keys:
            raise RuntimeError(
                f"Thiếu {len(missing)} key khi nạp checkpoint (cho phép "
                f"{self._cfg.max_missing_keys}). Mẫu: {missing[:10]}. Nếu đã xác nhận "
                "đây là các key không cần thiết, đặt SAM3_MAX_MISSING_KEYS cho khớp."
            )
        if missing:
            log.warning("Thiếu %d key (được phép): %s", len(missing), missing[:10])

    def _warmup(self) -> None:
        img = Image.new("RGB", (256, 256), (127, 127, 127))
        self.predict_concept(img, "object", self._cfg.default_threshold, 1)
        self.predict_instance(img, [(128.0, 128.0, 1)], None)
        log.info("Warm-up xong")

    def close(self) -> None:
        self._processor = None
        self._model = None
        if self._torch is not None and self._cfg.device == "cuda":
            with contextlib.suppress(Exception):
                self._torch.cuda.empty_cache()

    # --------------------------------------------------------------- helpers
    def _autocast(self):
        if self._autocast_dtype is None:
            return contextlib.nullcontext()
        return self._torch.autocast("cuda", dtype=self._autocast_dtype)

    def _require_loaded(self) -> None:
        if self._model is None or self._processor is None:
            raise ApiError(503, "not_ready", "Model chưa được nạp.")

    def _after_request(self) -> None:
        torch = self._torch
        # Predictor SAM1-task giữ _features/_is_image_set giữa các lần gọi nếu
        # predict_inst bị ngắt bởi lỗi: luôn đặt lại.
        self._model.inst_interactive_predictor.reset_predictor()
        self._processor.confidence_threshold = self._cfg.default_threshold
        if self._cfg.empty_cache_each_request and self._cfg.device == "cuda":
            torch.cuda.empty_cache()

    def _translate(self, exc: Exception) -> ApiError | None:
        torch = self._torch
        if torch is not None and self._cfg.device == "cuda":
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                with contextlib.suppress(Exception):
                    torch.cuda.empty_cache()
                return ApiError(
                    503,
                    "gpu_out_of_memory",
                    "Hết bộ nhớ GPU với yêu cầu này (thử ảnh nhỏ hơn hoặc threshold cao hơn).",
                    headers={"Retry-After": "5"},
                )
            if isinstance(exc, RuntimeError) and any(
                m in str(exc).lower() for m in _FATAL_CUDA_MARKERS
            ):
                return ApiError(
                    500, "gpu_fatal_error", "Lỗi CUDA không phục hồi được.", fatal=True
                )
        return None

    # -------------------------------------------------------------- instance
    def predict_instance(
        self,
        image: Image.Image,
        points: Sequence[tuple[float, float, int]],
        box: tuple[float, float, float, float] | None,
    ) -> list[RawInstance]:
        self._require_loaded()
        torch = self._torch
        w, h = image.size
        pts = np.array([[x, y] for x, y, _ in points], dtype=np.float32) if points else None
        lbl = np.array([lab for _, _, lab in points], dtype=np.int32) if points else None
        bx = np.array(box, dtype=np.float32) if box is not None else None
        # Một điểm đơn lẻ thì mơ hồ -> lấy 3 ứng viên rồi chọn điểm IoU dự đoán
        # cao nhất. Box hoặc nhiều prompt thì không mơ hồ -> một mask.
        multimask = bx is None and pts is not None and len(pts) == 1

        state = None
        try:
            with torch.inference_mode(), self._autocast():
                state = self._processor.set_image(image)
                masks, scores, _ = self._model.predict_inst(
                    state,
                    point_coords=pts,
                    point_labels=lbl,
                    box=bx,
                    multimask_output=multimask,
                )
        except Exception as exc:
            mapped = self._translate(exc)
            if mapped is not None:
                raise mapped from exc
            raise
        finally:
            state = None
            self._after_request()

        masks = np.asarray(masks)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        if masks.ndim == 2:
            masks = masks[None]
        if masks.shape[0] != scores.shape[0] or masks.shape[1:] != (h, w):
            raise RuntimeError(
                f"Kích thước mask bất thường: masks={masks.shape}, scores={scores.shape}, ảnh={(h, w)}"
            )
        best = int(np.argmax(scores))
        return [RawInstance(masks[best] > 0.5, float(scores[best]), "predicted_iou")]

    def predict_instances(
        self,
        image: Image.Image,
        boxes: Sequence[tuple[float, float, float, float]],
    ) -> list[RawInstance]:
        """Chạy nhiều box trên cùng state ảnh, giữ prompt_id theo thứ tự đầu vào."""
        self._require_loaded()
        torch = self._torch
        w, h = image.size
        state = None
        instances = []
        try:
            with torch.inference_mode(), self._autocast():
                # SAM3 image backbone chỉ chạy một lần; từng box dùng lại state này.
                state = self._processor.set_image(image)
                for prompt_id, box in enumerate(boxes):
                    masks, scores, _ = self._model.predict_inst(
                        state,
                        point_coords=None,
                        point_labels=None,
                        box=np.asarray(box, dtype=np.float32),
                        multimask_output=False,
                    )
                    masks = np.asarray(masks)
                    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
                    if masks.ndim == 2:
                        masks = masks[None]
                    if masks.shape[0] != scores.shape[0] or masks.shape[1:] != (h, w):
                        raise RuntimeError(
                            f"Kích thước mask bất thường: masks={masks.shape}, "
                            f"scores={scores.shape}, ảnh={(h, w)}"
                        )
                    best = int(np.argmax(scores))
                    instances.append(
                        RawInstance(
                            masks[best] > 0.5,
                            float(scores[best]),
                            "predicted_iou",
                            prompt_id,
                        )
                    )
        except Exception as exc:
            mapped = self._translate(exc)
            if mapped is not None:
                raise mapped from exc
            raise
        finally:
            state = None
            self._after_request()
        return instances

    # --------------------------------------------------------------- concept
    def predict_concept(
        self, image: Image.Image, text: str, threshold: float, max_instances: int
    ) -> list[RawInstance]:
        self._require_loaded()
        torch = self._torch
        w, h = image.size
        state = None
        try:
            with torch.inference_mode(), self._autocast():
                state = self._processor.set_image(image)
                self._processor.confidence_threshold = threshold
                state = self._processor.set_text_prompt(prompt=text, state=state)
                scores = state["scores"]
                if scores.numel() == 0:
                    return []
                order = torch.argsort(scores, descending=True)[:max_instances]
                masks = state["masks"][order].squeeze(1).cpu().numpy()
                top = scores[order].float().cpu().numpy()
        except Exception as exc:
            mapped = self._translate(exc)
            if mapped is not None:
                raise mapped from exc
            raise
        finally:
            state = None
            self._after_request()

        if masks.ndim != 3 or masks.shape[1:] != (h, w):
            raise RuntimeError(f"Kích thước mask bất thường: {masks.shape}, ảnh={(h, w)}")
        return [
            RawInstance(masks[i].astype(bool), float(top[i]), "detection_confidence")
            for i in range(masks.shape[0])
        ]
