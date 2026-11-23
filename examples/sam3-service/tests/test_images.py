import base64
import io

import numpy as np
import pytest
from PIL import Image

from sam3_service.config import Settings
from sam3_service.images import (
    build_instance,
    decode_image,
    encode_mask_png,
    mask_to_box,
    validate_concept_params,
    validate_points_and_box,
)
from sam3_service.schemas import ApiError, Box, Point


def cfg(**kw) -> Settings:
    return Settings(auth_disabled=True, **kw)


def b64_image(w=64, h=48, fmt="PNG", mode="RGB", **save_kw) -> str:
    img = Image.new(mode, (w, h), (10, 20, 30) if mode == "RGB" else 0)
    buf = io.BytesIO()
    img.save(buf, format=fmt, **save_kw)
    return base64.b64encode(buf.getvalue()).decode()


def code_of(fn, *a, **kw) -> tuple[int, str]:
    with pytest.raises(ApiError) as ei:
        fn(*a, **kw)
    return ei.value.status_code, ei.value.code


# ------------------------------------------------------------------ decode
@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "WEBP"])
def test_decode_formats(fmt):
    img = decode_image(b64_image(fmt=fmt), cfg())
    assert img.mode == "RGB" and img.size == (64, 48)


@pytest.mark.parametrize("size", [(1, 1), (31, 17), (33, 1), (257, 255)])
def test_decode_odd_sizes(size):
    assert decode_image(b64_image(*size), cfg()).size == size


def test_decode_rgba_and_palette_become_rgb():
    assert decode_image(b64_image(mode="RGBA"), cfg()).mode == "RGB"
    assert decode_image(b64_image(mode="P"), cfg()).mode == "RGB"
    assert decode_image(b64_image(mode="L"), cfg()).mode == "RGB"


def test_decode_applies_exif_orientation():
    img = Image.new("RGB", (60, 40), (200, 0, 0))
    exif = Image.Exif()
    exif[0x0112] = 6  # xoay 90 độ: ảnh hiển thị là 40x60
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    out = decode_image(base64.b64encode(buf.getvalue()).decode(), cfg())
    assert out.size == (40, 60)


def test_invalid_base64():
    assert code_of(decode_image, "!!!not-base64!!!", cfg()) == (400, "invalid_base64")


def test_data_uri_rejected():
    assert code_of(decode_image, "data:image/png;base64," + b64_image(), cfg()) == (
        400,
        "invalid_image",
    )


def test_not_an_image():
    junk = base64.b64encode(b"hello world, definitely not an image").decode()
    assert code_of(decode_image, junk, cfg()) == (400, "invalid_image")


def test_truncated_image_rejected():
    raw = base64.b64decode(b64_image(300, 300, fmt="PNG"))
    cut = base64.b64encode(raw[: len(raw) // 2]).decode()
    assert code_of(decode_image, cut, cfg()) == (400, "invalid_image")


def test_disallowed_format_rejected():
    assert code_of(decode_image, b64_image(fmt="GIF", mode="P"), cfg()) == (400, "invalid_image")


def test_too_many_pixels_rejected_before_decode():
    c = cfg(max_image_pixels=1024, max_image_side=4096)
    assert code_of(decode_image, b64_image(100, 100), c) == (413, "image_too_large")


def test_side_limit():
    c = cfg(max_image_side=64)
    assert code_of(decode_image, b64_image(65, 10), c) == (413, "image_too_large")


def test_payload_size_limit():
    c = cfg(max_image_bytes=2048)
    big = base64.b64encode(np.random.default_rng(0).bytes(5000)).decode()
    assert code_of(decode_image, big, c) == (413, "payload_too_large")


# ----------------------------------------------------------------- hình học
def test_mask_to_box_empty():
    assert mask_to_box(np.zeros((5, 7), bool)) is None


def test_mask_to_box_is_tight_with_exclusive_right_bottom():
    m = np.zeros((10, 12), bool)
    m[2:5, 3:9] = True  # rows 2..4, cols 3..8
    assert mask_to_box(m) == (3, 2, 9, 5)


def test_mask_to_box_touching_borders_and_single_pixel():
    m = np.zeros((6, 8), bool)
    m[:, :] = True
    assert mask_to_box(m) == (0, 0, 8, 6)
    m = np.zeros((6, 8), bool)
    m[5, 7] = True
    assert mask_to_box(m) == (7, 5, 8, 6)


def test_mask_with_hole_keeps_outer_box():
    m = np.zeros((20, 20), bool)
    m[2:18, 2:18] = True
    m[8:12, 8:12] = False
    assert mask_to_box(m) == (2, 2, 18, 18)
    inst = build_instance(m, 0.9, "predicted_iou")
    assert inst.area == 16 * 16 - 16


def test_encode_mask_roundtrip_odd_size():
    m = np.zeros((17, 31), bool)
    m[3:9, 4:20] = True
    png = base64.b64decode(encode_mask_png(m))
    out = Image.open(io.BytesIO(png))
    assert out.mode == "L" and out.size == (31, 17)  # (w, h)
    arr = np.asarray(out)
    assert set(np.unique(arr)) <= {0, 255}
    assert np.array_equal(arr == 255, m)


def test_build_instance_drops_empty_mask():
    assert build_instance(np.zeros((4, 4), bool), 0.5, "predicted_iou") is None


# ------------------------------------------------------------ validate prompt
def test_points_inside_ok_and_boundaries():
    c = cfg()
    pts, bx = validate_points_and_box([Point(x=0, y=0, label=1), Point(x=9.99, y=4.5, label=0)], None, 10, 5, c)
    assert pts == [(0.0, 0.0, 1), (9.99, 4.5, 0)] and bx is None


@pytest.mark.parametrize("x,y", [(10, 2), (2, 5), (-0.1, 1), (1, -1), (50, 50)])
def test_point_outside_rejected(x, y):
    assert code_of(validate_points_and_box, [Point(x=x, y=y, label=1)], None, 10, 5, cfg()) == (
        422,
        "invalid_prompt",
    )


def test_box_right_bottom_exclusive_may_equal_size():
    _, bx = validate_points_and_box([], Box(x1=0, y1=0, x2=10, y2=5), 10, 5, cfg())
    assert bx == (0.0, 0.0, 10.0, 5.0)


@pytest.mark.parametrize(
    "b",
    [
        (5, 1, 5, 3),  # rộng 0
        (6, 1, 5, 3),  # đảo ngược
        (1, 3, 4, 3),  # cao 0
        (-1, 0, 4, 3),
        (0, 0, 11, 3),
        (0, 0, 4, 6),
    ],
)
def test_bad_boxes_rejected(b):
    assert code_of(
        validate_points_and_box, [], Box(x1=b[0], y1=b[1], x2=b[2], y2=b[3]), 10, 5, cfg()
    ) == (422, "invalid_prompt")


def test_too_many_points():
    pts = [Point(x=1, y=1, label=1)] * 5
    assert code_of(validate_points_and_box, pts, None, 10, 5, cfg(max_points=4))[1] == "invalid_prompt"


def test_concept_params_defaults_and_limits():
    c = cfg(max_instances=10, min_threshold=0.2, default_threshold=0.5)
    assert validate_concept_params("car", None, None, c) == ("car", 0.5, 10)
    assert validate_concept_params("car", 0.3, 3, c) == ("car", 0.3, 3)
    for thr in (0.1, 1.0):
        assert code_of(validate_concept_params, "car", thr, None, c)[1] == "invalid_prompt"
    assert code_of(validate_concept_params, "car", None, 11, c)[1] == "invalid_prompt"
    assert code_of(validate_concept_params, "x" * 200, None, None, cfg(max_text_chars=100))[1] == "invalid_prompt"
