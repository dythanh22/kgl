"""Schema request/response/lỗi. Độc lập với framework suy luận.

Quy ước hình học (áp dụng cho request lẫn response):

* Tọa độ tính trên ảnh **sau khi áp dụng EXIF orientation**, đã chuyển RGB.
  ``image.width/height`` trong response là kích thước của ảnh đó; mọi mask
  trả về có đúng kích thước này.
* Hệ tọa độ liên tục: pixel ``i`` chiếm đoạn ``[i, i+1)``. Điểm hợp lệ có
  ``0 <= x < width`` và ``0 <= y < height``.
* Box: ``x1, y1`` inclusive (trái/trên), ``x2, y2`` exclusive (phải/dưới),
  ``0 <= x1 < x2 <= width``, ``0 <= y1 < y2 <= height``.
* Box trong response được suy ra từ mask (hộp bao chặt, số nguyên), cùng quy
  ước trên, nên khớp tuyệt đối với mask.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from . import SCHEMA_VERSION

RequestId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9._:-]{1,64}$")]
Coord = Annotated[float, Field(allow_inf_nan=False)]
# strict: không nhận True/1.0/"1" cho nhãn điểm.
PointLabel = Annotated[int, Field(strict=True, ge=0, le=1)]

MAX_POINTS_HARD = 64
MAX_BATCH_BOXES_HARD = 12
MAX_TEXT_HARD = 256


class ApiError(Exception):
    """Lỗi có mã trạng thái HTTP và mã lỗi máy đọc được."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        *,
        headers: dict[str, str] | None = None,
        fatal: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.headers = headers or {}
        # fatal=True: model không còn dùng được (vd. lỗi CUDA sticky).
        self.fatal = fatal


class _Strict(BaseModel):
    # extra="forbid": trường không được hỗ trợ ở một mode (vd. threshold ở
    # mode instance) bị từ chối thay vì bị bỏ qua âm thầm.
    model_config = ConfigDict(extra="forbid")


class Point(_Strict):
    x: Coord
    y: Coord
    label: PointLabel  # 1 = dương (thuộc đối tượng), 0 = âm (loại trừ)


class Box(_Strict):
    x1: Coord
    y1: Coord
    x2: Coord
    y2: Coord


class _CommonRequest(_Strict):
    schema_version: Literal["1"]
    request_id: RequestId | None = None
    # Base64 thuần của byte file ảnh (JPEG/PNG/WEBP); không có tiền tố data:.
    image_base64: str = Field(min_length=1)


class InstanceRequest(_CommonRequest):
    """Điểm/box đơn hoặc nhiều box độc lập -> instance."""

    mode: Literal["instance"]
    points: list[Point] = Field(default_factory=list, max_length=MAX_POINTS_HARD)
    box: Box | None = None
    boxes: list[Box] | None = Field(default=None, min_length=1, max_length=MAX_BATCH_BOXES_HARD)

    @model_validator(mode="after")
    def _need_prompt(self) -> "InstanceRequest":
        if self.boxes is not None:
            if self.box is not None or self.points:
                raise ValueError("boxes không thể kết hợp với box hoặc points")
            return self
        if self.box is None and not any(p.label == 1 for p in self.points):
            raise ValueError(
                "mode=instance cần ít nhất một box hoặc một điểm dương (label=1)"
            )
        return self


class ConceptRequest(_CommonRequest):
    """Text -> danh sách instance."""

    mode: Literal["concept"]
    text: str = Field(min_length=1, max_length=MAX_TEXT_HARD)
    threshold: float | None = Field(default=None, gt=0.0, lt=1.0)
    max_instances: int | None = Field(default=None, ge=1)

    @field_validator("text")
    @classmethod
    def _clean_text(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("text không được rỗng")
        if any(ord(c) < 32 or ord(c) == 127 for c in v):
            raise ValueError("text chứa ký tự điều khiển")
        return v


PredictRequest = Annotated[
    Union[InstanceRequest, ConceptRequest], Field(discriminator="mode")
]

ScoreType = Literal["predicted_iou", "detection_confidence"]


class ImageInfo(BaseModel):
    width: int
    height: int


class ModelMeta(BaseModel):
    name: str
    source_sha: str
    checkpoint_revision: str
    # None: chưa được xác minh (không đặt SAM3_CHECKPOINT_SHA256).
    checkpoint_sha256: str | None
    precision: str
    device: str


class Timing(BaseModel):
    queue_ms: float
    inference_ms: float
    total_ms: float


class OutBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class InstanceOut(BaseModel):
    # PNG 8-bit grayscale, kích thước đúng bằng ảnh; 255 = đối tượng, 0 = nền.
    mask_png_base64: str
    box: OutBox
    area: int
    score: float
    score_type: ScoreType
    # Chỉ có trong batch boxes; chỉ số 0-based theo thứ tự box đầu vào.
    prompt_id: int | None = None


class BatchInfo(BaseModel):
    count: int
    empty_prompt_ids: list[int]


class PredictResponse(BaseModel):
    schema_version: str = SCHEMA_VERSION
    request_id: str
    mode: Literal["instance", "concept"]
    image: ImageInfo
    model: ModelMeta
    timing: Timing
    # [] khi không có kết quả; lỗi inference trả lỗi HTTP riêng.
    instances: list[InstanceOut]
    batch: BatchInfo | None = None


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str | None = None
    details: list[dict] | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody
