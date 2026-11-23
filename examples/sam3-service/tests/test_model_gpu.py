"""Test model THẬT trên GPU. Mặc định bị bỏ qua.

Chạy trên máy B (trong container hoặc venv đã cài torch + sam3):

    SAM3_TEST_CHECKPOINT=/srv/sam3/weights/sam3.pt pytest -m gpu -s tests/test_model_gpu.py

Tùy chọn:
    SAM3_TEST_CHECKPOINT_SHA256  xác minh SHA-256 checkpoint
    SAM3_TEST_IMAGE, SAM3_TEST_TEXT  ảnh thật + text (vd. ảnh có xe, text "car")

CHƯA ĐƯỢC CHẠY khi viết: các ngưỡng IoU bên dưới là kiểm tra hợp lý trên ảnh
tổng hợp, có thể cần chỉnh sau lần chạy đầu. Mock ở test_api.py không thay thế
được các test cô lập state ở đây.
"""

from __future__ import annotations

import asyncio
import base64
import io
import os
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
CKPT = os.environ.get("SAM3_TEST_CHECKPOINT")
if not CKPT or not Path(CKPT).is_file() or not torch.cuda.is_available():
    pytest.skip("cần GPU + SAM3_TEST_CHECKPOINT trỏ tới sam3.pt", allow_module_level=True)

from fastapi.testclient import TestClient  # noqa: E402

from sam3_service.app import create_app  # noqa: E402
from sam3_service.config import Settings  # noqa: E402
from sam3_service.schemas import ApiError  # noqa: E402


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 1.0


def scene(w=640, h=480):
    """Nền xám + một elip đỏ đặc. Trả về (ảnh, mask elip, box elip)."""
    box = (int(w * 0.30), int(h * 0.25), int(w * 0.60), int(h * 0.75))
    img = Image.new("RGB", (w, h), (90, 90, 90))
    ImageDraw.Draw(img).ellipse(box, fill=(220, 30, 30))
    gt = Image.new("L", (w, h), 0)
    ImageDraw.Draw(gt).ellipse(box, fill=255)
    return img, np.asarray(gt) > 0, box


def b64(img: Image.Image, fmt="PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture(scope="module")
def svc():
    settings = Settings(
        auth_disabled=True,
        checkpoint_path=Path(CKPT),
        checkpoint_sha256=os.environ.get("SAM3_TEST_CHECKPOINT_SHA256"),
        exit_on_fatal=False,
        warmup=True,
        max_image_pixels=40_000_000,
        max_image_side=8192,
    )
    app = create_app(settings)  # model thật, nạp trong luồng suy luận
    with TestClient(app) as client:
        end = time.time() + 900
        while time.time() < end:
            r = client.get("/readyz")
            if r.status_code == 200:
                break
            assert r.json()["status"] != "failed", "nạp model thất bại (xem log)"
            time.sleep(1)
        else:
            pytest.fail("model không sẵn sàng sau 900s")
        yield client, app.state.ctx


def on_gpu(ctx, fn, *a):
    """Chạy ``fn`` trong luồng suy luận duy nhất (như production)."""
    value, _ = asyncio.run(ctx.engine.run(lambda: fn(*a), timeout_s=300))
    return value


# -------------------------------------------------------------------- model
def test_meta_and_ready(svc):
    _, ctx = svc
    meta = ctx.model.meta
    assert meta.name == "sam3-image" and meta.precision in ("fp32", "bf16-autocast")
    assert ctx.model._model.inst_interactive_predictor is not None


@pytest.mark.parametrize("size", [(640, 480), (333, 251), (97, 61)])
def test_instance_box_matches_image_size_and_object(svc, size):
    _, ctx = svc
    img, gt, box = scene(*size)
    (inst,) = on_gpu(ctx, ctx.model.predict_instance, img, [], tuple(map(float, box)))
    assert inst.mask.shape == (size[1], size[0]) and inst.mask.dtype == bool
    assert inst.score_type == "predicted_iou" and np.isfinite(inst.score)
    assert iou(inst.mask, gt) > 0.7, f"IoU thấp: {iou(inst.mask, gt):.3f}"


def test_instance_single_point_returns_one_candidate(svc):
    _, ctx = svc
    img, gt, box = scene()
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    out = on_gpu(ctx, ctx.model.predict_instance, img, [(cx, cy, 1)], None)
    assert len(out) == 1 and out[0].mask.shape == gt.shape


def test_instance_negative_point_changes_result(svc):
    _, ctx = svc
    img, gt, box = scene()
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    a = on_gpu(ctx, ctx.model.predict_instance, img, [(cx, cy, 1)], None)[0].mask
    b = on_gpu(ctx, ctx.model.predict_instance, img, [(cx, cy, 1), (5.0, 5.0, 0)], None)[0].mask
    assert a.shape == b.shape  # chỉ kiểm hợp lệ hình học; kết quả có thể giống nhau


def test_concept_structure(svc):
    _, ctx = svc
    path = os.environ.get("SAM3_TEST_IMAGE")
    if path:
        img = Image.open(path).convert("RGB")
        text = os.environ.get("SAM3_TEST_TEXT", "car")
    else:
        img, _, _ = scene()
        text = "red ellipse"
    out = on_gpu(ctx, ctx.model.predict_concept, img, text, 0.3, 5)
    assert len(out) <= 5
    scores = [o.score for o in out]
    assert scores == sorted(scores, reverse=True)
    for o in out:
        assert o.mask.shape == (img.height, img.width) and 0.3 < o.score <= 1.0
        assert o.score_type == "detection_confidence"
    if path:
        assert out, "ảnh/text thật nhưng không có kết quả"


def test_no_state_leak_when_interleaving_modes_and_prompts(svc):
    _, ctx = svc
    m = ctx.model
    imgA, _, boxA = scene(640, 480)
    imgB, _, _ = scene(300, 200)
    boxB = (30.0, 20.0, 120.0, 150.0)
    a1 = on_gpu(ctx, m.predict_instance, imgA, [], tuple(map(float, boxA)))[0].mask
    on_gpu(ctx, m.predict_concept, imgB, "ellipse", 0.2, 3)
    b = on_gpu(ctx, m.predict_instance, imgB, [], boxB)[0].mask
    on_gpu(ctx, m.predict_concept, imgA, "something else", 0.9, 3)
    a2 = on_gpu(ctx, m.predict_instance, imgA, [], tuple(map(float, boxA)))[0].mask
    assert b.shape == (200, 300) and a2.shape == (480, 640)
    assert iou(a1, a2) > 0.99, f"kết quả A thay đổi sau request khác: IoU={iou(a1, a2):.4f}"
    # Ngưỡng của request concept không được rò sang request sau.
    assert m._processor.confidence_threshold == ctx.settings.default_threshold
    pred = m._model.inst_interactive_predictor
    assert pred._is_image_set is False and pred._features is None


def test_recovers_after_error_and_cleans_predictor(svc):
    _, ctx = svc
    m = ctx.model
    img, gt, box = scene()
    real = m._model.predict_inst

    def boom(*a, **k):
        raise RuntimeError("lỗi giả lập")

    def faulty():
        m._model.predict_inst = boom
        try:
            m.predict_instance(img, [], tuple(map(float, box)))
        finally:
            m._model.predict_inst = real

    with pytest.raises(ApiError) as ei:  # engine đổi lỗi lạ thành inference_failed
        on_gpu(ctx, faulty)
    assert ei.value.code == "inference_failed"
    pred = m._model.inst_interactive_predictor
    assert pred._is_image_set is False and pred._features is None
    (inst,) = on_gpu(ctx, m.predict_instance, img, [], tuple(map(float, box)))
    assert iou(inst.mask, gt) > 0.7


def test_oom_is_translated_and_service_continues(svc):
    _, ctx = svc
    m = ctx.model
    img, _, box = scene()
    real = m._model.predict_inst

    def oom(*a, **k):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory (giả lập)")

    def faulty():
        m._model.predict_inst = oom
        try:
            m.predict_instance(img, [], tuple(map(float, box)))
        finally:
            m._model.predict_inst = real

    with pytest.raises(ApiError) as ei:
        on_gpu(ctx, faulty)
    assert ei.value.code == "gpu_out_of_memory" and not ei.value.fatal
    assert on_gpu(ctx, m.predict_instance, img, [], tuple(map(float, box)))


# ---------------------------------------------------------------- end-to-end
def test_http_instance_and_concept_end_to_end(svc):
    client, _ = svc
    img, gt, box = scene()
    r = client.post("/v1/predict", json={
        "schema_version": "1", "mode": "instance", "image_base64": b64(img),
        "box": dict(zip(("x1", "y1", "x2", "y2"), map(float, box))),
    })
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["image"] == {"width": 640, "height": 480} and len(d["instances"]) == 1
    mask = np.asarray(Image.open(io.BytesIO(base64.b64decode(d["instances"][0]["mask_png_base64"])))) == 255
    assert mask.shape == (480, 640) and iou(mask, gt) > 0.7
    b = d["instances"][0]["box"]
    ys, xs = np.nonzero(mask)
    assert (b["x1"], b["y1"], b["x2"], b["y2"]) == (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)

    r = client.post("/v1/predict", json={
        "schema_version": "1", "mode": "concept", "image_base64": b64(img), "text": "ellipse", "threshold": 0.2,
    })
    assert r.status_code == 200 and r.json()["timing"]["inference_ms"] > 0


def test_large_image_does_not_break_service(svc):
    """Ảnh lớn: hoặc thành công, hoặc lỗi rõ ràng (OOM/413) và service vẫn sống."""
    client, _ = svc
    img, _, box = scene(4000, 3000)  # 12 MP
    r = client.post("/v1/predict", json={
        "schema_version": "1", "mode": "instance", "image_base64": b64(img, "JPEG"),
        "box": dict(zip(("x1", "y1", "x2", "y2"), map(float, box))),
    })
    assert r.status_code in (200, 503, 413), r.text
    assert client.get("/readyz").status_code == 200


def test_report_latency_and_vram(svc):
    """In số đo để ghi vào baseline (chạy với -s). Không có assert về hiệu năng."""
    client, ctx = svc
    rows = []
    for w, h in [(512, 512), (1280, 720), (1920, 1080), (3840, 2160)]:
        img, _, box = scene(w, h)
        body = {"schema_version": "1", "mode": "instance", "image_base64": b64(img, "JPEG"),
                "box": dict(zip(("x1", "y1", "x2", "y2"), map(float, box)))}
        torch.cuda.reset_peak_memory_stats()
        times = []
        status = None
        for _ in range(3):
            r = client.post("/v1/predict", json=body)
            status = r.status_code
            if status == 200:
                times.append(r.json()["timing"]["inference_ms"])
        rows.append((f"{w}x{h}", status, times, torch.cuda.max_memory_allocated() / 2**30))
    print("\nảnh | status | inference_ms (3 lần) | peak allocated GiB")
    for name, status, times, peak in rows:
        print(f"{name} | {status} | {[round(t) for t in times]} | {peak:.2f}")
