"""Đọc và kiểm tra cấu hình từ biến môi trường (tiền tố ``SAM3_``)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

MIN_API_KEY_LENGTH = 24


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SAM3_", extra="ignore")

    # --- Xác thực ---------------------------------------------------------
    api_key: SecretStr | None = None
    # Chỉ dùng khi phát triển cục bộ. Production phải đặt API key.
    auth_disabled: bool = False

    # --- Model ------------------------------------------------------------
    checkpoint_path: Path = Path("/weights/sam3.pt")
    # None: dùng file BPE đi kèm gói ``sam3``.
    bpe_path: Path | None = None
    # Builder của SAM3 chỉ chuyển model lên GPU khi device == "cuda" đúng
    # nguyên văn, nên không nhận "cuda:0". Chọn GPU bằng CUDA_VISIBLE_DEVICES.
    device: Literal["cuda", "cpu"] = "cuda"
    # auto: BF16 nếu GPU compute capability >= 8 và hỗ trợ BF16, ngược lại FP32.
    precision: Literal["auto", "bf16", "fp32"] = "auto"
    # Số key thiếu tối đa khi nạp checkpoint (SAM3 nạp strict=False và chỉ in
    # cảnh báo). 0 = bất kỳ key thiếu nào cũng làm khởi động thất bại.
    max_missing_keys: int = Field(default=0, ge=0)
    warmup: bool = True
    empty_cache_each_request: bool = False

    # Thông tin truy vết, trả về trong response. Đặt khi build/deploy.
    source_sha: str = "unknown"
    checkpoint_revision: str = "unknown"
    # Nếu đặt: xác minh SHA-256 file checkpoint lúc khởi động (sai thì dừng).
    checkpoint_sha256: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{64}$")

    # --- Giới hạn đầu vào -------------------------------------------------
    max_image_bytes: int = Field(default=20 * 1024 * 1024, ge=1024)
    max_image_pixels: int = Field(default=12_000_000, ge=1024)
    max_image_side: int = Field(default=8192, ge=16)
    max_points: int = Field(default=16, ge=1, le=64)
    max_text_chars: int = Field(default=128, ge=1, le=256)
    max_instances: int = Field(default=100, ge=1)
    default_threshold: float = Field(default=0.5, gt=0.0, lt=1.0)
    # Chặn ngưỡng quá thấp: processor nội suy mọi mask vượt ngưỡng lên độ phân
    # giải ảnh gốc, nên ngưỡng thấp có thể làm bộ nhớ GPU tăng vọt.
    min_threshold: float = Field(default=0.1, gt=0.0, lt=1.0)

    # --- Thực thi ---------------------------------------------------------
    queue_size: int = Field(default=8, ge=1, le=1024)
    request_timeout_s: float = Field(default=60.0, gt=0.0)
    decode_concurrency: int = Field(default=2, ge=1, le=16)
    # Lỗi CUDA không phục hồi được / nạp model thất bại -> tự dừng process để
    # Docker restart (restart: unless-stopped).
    exit_on_fatal: bool = True
    log_level: str = "INFO"

    @field_validator("checkpoint_sha256", "bpe_path", mode="before")
    @classmethod
    def _empty_is_none(cls, v):
        # `VAR=` trong .env cho chuỗi rỗng: coi như chưa đặt.
        return None if isinstance(v, str) and not v.strip() else v

    @field_validator("source_sha", "checkpoint_revision", mode="before")
    @classmethod
    def _empty_is_unknown(cls, v):
        return "unknown" if isinstance(v, str) and not v.strip() else v

    @property
    def max_b64_chars(self) -> int:
        return (self.max_image_bytes + 2) // 3 * 4

    @property
    def max_body_bytes(self) -> int:
        # base64 + phần JSON còn lại (prompt, request_id, ...).
        return self.max_b64_chars + 64 * 1024

    @model_validator(mode="after")
    def _check(self) -> "Settings":
        if not self.auth_disabled:
            key = self.api_key.get_secret_value() if self.api_key else ""
            if len(key) < MIN_API_KEY_LENGTH:
                raise ValueError(
                    f"SAM3_API_KEY bắt buộc và dài tối thiểu {MIN_API_KEY_LENGTH} "
                    "ký tự (hoặc đặt SAM3_AUTH_DISABLED=true chỉ khi phát triển)."
                )
        if self.min_threshold > self.default_threshold:
            raise ValueError("SAM3_MIN_THRESHOLD phải <= SAM3_DEFAULT_THRESHOLD")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
