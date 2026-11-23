"""Decode ảnh, kiểm kích thước/prompt, encode mask. Không phụ thuộc Torch."""

from __future__ import annotations

import base64
import binascii
import io
import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFile, ImageOps, UnidentifiedImageError

from .config import Settings
from .schemas import ApiError, Box, InstanceOut, OutBox, Point, ScoreType

ALLOWED_FORMATS = ("JPEG", "PNG", "WEBP")

# Ảnh cắt cụt phải bị từ chối, không được "đoán" nốt phần còn lại.
ImageFile.LOAD_TRUNCATED_IMAGES = False


def decode_image(b64: str, cfg: Settings) -> Image.Image:
    """base64 -> ảnh RGB đã áp dụng EXIF orientation.

    Chạy trong threadpool (CPU), không chạy trong luồng GPU.
    """
    if b64.startswith("data:"):
        raise ApiError(400, "invalid_image", "Không hỗ trợ data URI; gửi base64 thuần.")
    if len(b64) > cfg.max_b64_chars:
        raise ApiError(413, "payload_too_large", "Ảnh vượt quá dung lượng cho phép.")
    try:
        raw = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError):
        raise ApiError(400, "invalid_base64", "image_base64 không phải base64 hợp lệ.")
    if len(raw) > cfg.max_image_bytes:
        raise ApiError(413, "payload_too_large", "Ảnh vượt quá dung lượng cho phép.")

    try:
        img = Image.open(io.BytesIO(raw), formats=list(ALLOWED_FORMATS))
        # Mới đọc header: kiểm kích thước trước khi giải nén toàn bộ.
        w, h = img.size
        if w < 1 or h < 1:
            raise ApiError(400, "invalid_image", "Ảnh có kích thước không hợp lệ.")
        if w * h > cfg.max_image_pixels or max(w, h) > cfg.max_image_side:
            raise ApiError(
                413,
                "image_too_large",
                f"Ảnh {w}x{h} vượt giới hạn (tối đa {cfg.max_image_pixels} pixel, "
                f"cạnh dài tối đa {cfg.max_image_side}).",
            )
        if getattr(img, "n_frames", 1) > 1:
            raise ApiError(400, "invalid_image", "Không hỗ trợ ảnh nhiều khung (animated).")
        img.load()
        # exif_transpose trả bản sao đã xoay theo EXIF; ảnh không có EXIF giữ nguyên.
        img = ImageOps.exif_transpose(img)
        # Kênh alpha (nếu có) bị loại bỏ bởi convert; model chỉ nhận RGB.
        return img.convert("RGB")
    except ApiError:
        raise
    except Image.DecompressionBombError:
        raise ApiError(413, "image_too_large", "Ảnh vượt giới hạn số pixel.")
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, EOFError):
        raise ApiError(
            400, "invalid_image", "Không giải mã được ảnh (hỏng, cắt cụt hoặc sai định dạng)."
        )


def validate_points_and_box(
    points: list[Point], box: Box | None, width: int, height: int, cfg: Settings
) -> tuple[list[tuple[float, float, int]], tuple[float, float, float, float] | None]:
    """Kiểm prompt hình học theo kích thước ảnh sau xử lý orientation."""
    if len(points) > cfg.max_points:
        raise ApiError(422, "invalid_prompt", f"Tối đa {cfg.max_points} điểm.")
    pts: list[tuple[float, float, int]] = []
    for i, p in enumerate(points):
        if not (0 <= p.x < width and 0 <= p.y < height):
            raise ApiError(
                422,
                "invalid_prompt",
                f"points[{i}]=({p.x}, {p.y}) nằm ngoài ảnh {width}x{height} "
                "(yêu cầu 0 <= x < width, 0 <= y < height).",
            )
        pts.append((float(p.x), float(p.y), int(p.label)))

    bx = None
    if box is not None:
        if not (0 <= box.x1 < box.x2 <= width and 0 <= box.y1 < box.y2 <= height):
            raise ApiError(
                422,
                "invalid_prompt",
                f"box ({box.x1}, {box.y1}, {box.x2}, {box.y2}) không hợp lệ trên ảnh "
                f"{width}x{height} (yêu cầu 0 <= x1 < x2 <= width, 0 <= y1 < y2 <= height).",
            )
        bx = (float(box.x1), float(box.y1), float(box.x2), float(box.y2))
    return pts, bx


def validate_boxes(
    boxes: list[Box], width: int, height: int
) -> list[tuple[float, float, float, float]]:
    """Kiểm hình học và giữ thứ tự các box của một batch instance."""
    checked = []
    for i, box in enumerate(boxes):
        if not (0 <= box.x1 < box.x2 <= width and 0 <= box.y1 < box.y2 <= height):
            raise ApiError(
                422,
                "invalid_prompt",
                f"boxes[{i}] không hợp lệ trên ảnh {width}x{height} "
                "(yêu cầu 0 <= x1 < x2 <= width, 0 <= y1 < y2 <= height).",
            )
        checked.append((float(box.x1), float(box.y1), float(box.x2), float(box.y2)))
    return checked


def validate_concept_params(
    text: str, threshold: float | None, max_instances: int | None, cfg: Settings
) -> tuple[str, float, int]:
    if len(text) > cfg.max_text_chars:
        raise ApiError(422, "invalid_prompt", f"text tối đa {cfg.max_text_chars} ký tự.")
    thr = cfg.default_threshold if threshold is None else float(threshold)
    if not math.isfinite(thr) or not (cfg.min_threshold <= thr < 1.0):
        raise ApiError(
            422,
            "invalid_prompt",
            f"threshold phải trong [{cfg.min_threshold}, 1.0).",
        )
    cap = cfg.max_instances
    n = cap if max_instances is None else int(max_instances)
    if n < 1 or n > cap:
        raise ApiError(422, "invalid_prompt", f"max_instances phải trong [1, {cap}].")
    return text, thr, n


def mask_to_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Hộp bao chặt (x1, y1, x2, y2), x2/y2 exclusive; None nếu mask rỗng."""
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(mask.any(axis=0))
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def encode_mask_png(mask: np.ndarray) -> str:
    """Mask bool HxW -> PNG 8-bit (0/255) -> base64."""
    arr = np.ascontiguousarray(mask, dtype=np.uint8) * np.uint8(255)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG", compress_level=6)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build_instance(
    mask: np.ndarray, score: float, score_type: ScoreType, prompt_id: int | None = None
) -> InstanceOut | None:
    """Mask -> InstanceOut. Mask rỗng trả None (bị loại khỏi kết quả)."""
    box = mask_to_box(mask)
    if box is None:
        return None
    return InstanceOut(
        mask_png_base64=encode_mask_png(mask),
        box=OutBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3]),
        area=int(mask.sum()),
        score=float(score),
        score_type=score_type,
        prompt_id=prompt_id,
    )


@dataclass(frozen=True)
class ImageSize:
    width: int
    height: int
