"""Test hợp đồng HTTP với model GIẢ (đưa vào create_app). Không cần Torch/GPU.

Lưu ý: mock KHÔNG chứng minh được cô lập state của SAM3; việc đó nằm ở
test_model_gpu.py và cần GPU thật.
"""

import base64
import io
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from sam3_service.app import create_app
from sam3_service.config import Settings
from sam3_service.model import RawInstance
from sam3_service.schemas import ApiError, ModelMeta

KEY = "k" * 32
AUTH = {"Authorization": f"Bearer {KEY}"}


class FakeModel:
    """Hình học giả lập: instance = box (hoặc đĩa quanh điểm dương); concept = N ô."""

    def __init__(self, *, delay=0.0, load_gate=None, load_error=None):
        self.delay = delay
        self.load_gate = load_gate
        self.load_error = load_error
        self.calls: list[tuple] = []
        self.active = 0
        self.max_active = 0
        self.raise_next: Exception | None = None
        self._lock = threading.Lock()

    @property
    def meta(self):
        return ModelMeta(
            name="fake", source_sha="s", checkpoint_revision="r",
            checkpoint_sha256=None, precision="fp32", device="cpu",
        )

    def load(self):
        if self.load_gate is not None:
            self.load_gate.wait(10)
        if self.load_error:
            raise RuntimeError(self.load_error)

    def close(self):
        pass

    def _enter(self):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(self.delay)
        if self.raise_next is not None:
            exc, self.raise_next = self.raise_next, None
            raise exc

    def _exit(self):
        with self._lock:
            self.active -= 1

    def predict_instance(self, image, points, box):
        self.calls.append(("instance", image.size, list(points), box))
        self._enter()
        try:
            w, h = image.size
            m = np.zeros((h, w), bool)
            if box is not None:
                x1, y1, x2, y2 = box
                m[int(np.floor(y1)):int(np.ceil(y2)), int(np.floor(x1)):int(np.ceil(x2))] = True
            else:
                x, y, _ = next(p for p in points if p[2] == 1)
                yy, xx = np.ogrid[:h, :w]
                m = (xx - x) ** 2 + (yy - y) ** 2 <= 4
            return [RawInstance(m, 0.9, "predicted_iou")]
        finally:
            self._exit()

    def predict_instances(self, image, boxes):
        self.calls.append(("instances", image.size, list(boxes)))
        self._enter()
        try:
            w, h = image.size
            results = []
            for prompt_id, (x1, y1, x2, y2) in enumerate(boxes):
                mask = np.zeros((h, w), bool)
                mask[int(y1):int(y2), int(x1):int(x2)] = True
                results.append(RawInstance(mask, 0.9, "predicted_iou", prompt_id))
            return results
        finally:
            self._exit()

    def predict_concept(self, image, text, threshold, max_instances):
        self.calls.append(("concept", image.size, text, threshold, max_instances))
        self._enter()
        try:
            w, h = image.size
            n = {"none": 0, "empty": 1}.get(text, 5)
            out = []
            for i in range(n):
                m = np.zeros((h, w), bool)
                if text != "empty":
                    m[: max(1, h // 4), i : i + 1] = True
                out.append(RawInstance(m, 0.99 - 0.1 * i, "detection_confidence"))
            return out[:max_instances]
        finally:
            self._exit()


def b64_png(w=64, h=48, fmt="PNG", **kw) -> str:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (1, 2, 3)).save(buf, format=fmt, **kw)
    return base64.b64encode(buf.getvalue()).decode()


def make_settings(**kw) -> Settings:
    kw.setdefault("api_key", KEY)
    kw.setdefault("exit_on_fatal", False)
    return Settings(**kw)


def wait_ready(client, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if client.get("/readyz").status_code == 200:
            return
        time.sleep(0.02)
    raise AssertionError("service không ready")


@pytest.fixture
def fake():
    return FakeModel()


@pytest.fixture
def client(fake):
    with TestClient(create_app(make_settings(), model=fake)) as c:
        wait_ready(c)
        yield c


def test_demo_page_is_served_without_api_auth(client):
    response = client.get("/demo")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "SAM 3 Lab" in response.text
    assert "/v1/predict" in response.text


def test_root_redirects_to_demo(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/demo"


def inst_body(**kw):
    body = {"schema_version": "1", "mode": "instance", "image_base64": b64_png(),
            "box": {"x1": 4, "y1": 5, "x2": 20, "y2": 30}}
    body.update(kw)
    return body


def concept_body(**kw):
    body = {"schema_version": "1", "mode": "concept", "image_base64": b64_png(), "text": "car"}
    body.update(kw)
    return body


def decode_mask(inst) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(inst["mask_png_base64"]))))


# ------------------------------------------------------------ health/ready
def test_healthz_needs_no_auth(client):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_readyz_loading_then_ready():
    gate = threading.Event()
    with TestClient(create_app(make_settings(), model=FakeModel(load_gate=gate))) as c:
        r = c.get("/readyz")
        assert r.status_code == 503 and r.json()["status"] == "loading"
        assert c.get("/healthz").status_code == 200  # process vẫn trả lời khi đang nạp
        r = c.post("/v1/predict", json=inst_body(), headers=AUTH)
        assert r.status_code == 503 and r.json()["error"]["code"] == "not_ready"
        gate.set()
        wait_ready(c)
        assert c.post("/v1/predict", json=inst_body(), headers=AUTH).status_code == 200


def test_load_failure_reports_failed_and_requests_terminate_hook():
    stopped = []
    app = create_app(
        make_settings(exit_on_fatal=True),
        model=FakeModel(load_error="no checkpoint"),
        terminate=lambda: stopped.append(1),
    )
    with TestClient(app) as c:
        end = time.time() + 5
        while not stopped and time.time() < end:
            time.sleep(0.02)
        r = c.get("/readyz")
        assert r.status_code == 503 and r.json()["status"] == "failed"
        r = c.post("/v1/predict", json=inst_body(), headers=AUTH)
        assert r.status_code == 503
    assert stopped == [1]


# --------------------------------------------------------------------- auth
def test_auth_required_and_checked_before_body(client):
    assert client.post("/v1/predict", json=inst_body()).status_code == 401
    r = client.post("/v1/predict", json=inst_body(), headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"
    assert r.json()["error"]["code"] == "unauthorized"
    # Body hỏng nhưng chưa xác thực -> vẫn 401, không phải 422.
    assert client.post("/v1/predict", content=b"{not json").status_code == 401
    assert client.get("/v1/openapi.json").status_code == 401
    assert client.get("/v1/openapi.json", headers=AUTH).status_code == 200


def test_settings_require_api_key_unless_explicitly_disabled():
    with pytest.raises(ValueError):
        Settings()
    with pytest.raises(ValueError):
        Settings(api_key="short")
    Settings(auth_disabled=True)


def test_settings_from_env_example_style_empty_values(monkeypatch):
    """`VAR=` để trống trong .env không được làm hỏng khởi động."""
    for k, v in {
        "SAM3_API_KEY": "x" * 32,
        "SAM3_CHECKPOINT_SHA256": "",
        "SAM3_CHECKPOINT_REVISION": "",
        "SAM3_SOURCE_SHA": "",
        "SAM3_BPE_PATH": "",
    }.items():
        monkeypatch.setenv(k, v)
    s = Settings()
    assert s.checkpoint_sha256 is None and s.bpe_path is None
    assert s.checkpoint_revision == "unknown" and s.source_sha == "unknown"


def test_env_example_file_is_loadable(monkeypatch, tmp_path):
    """Mọi biến trong .env.example phải được Settings chấp nhận (chỉ thêm API key)."""
    import pathlib

    text = (pathlib.Path(__file__).parent.parent / ".env.example").read_text()
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            monkeypatch.setenv(k, v)
    monkeypatch.setenv("SAM3_API_KEY", "y" * 32)
    s = Settings()
    assert s.queue_size == 8 and s.device == "cuda" and s.max_missing_keys == 0


# ------------------------------------------------------------ instance mode
def test_instance_box_roundtrip_matches_image_and_box(client, fake):
    r = client.post("/v1/predict", json=inst_body(request_id="abc-1"), headers=AUTH)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["request_id"] == "abc-1" and r.headers["x-request-id"] == "abc-1"
    assert d["schema_version"] == "1" and d["mode"] == "instance"
    assert d["image"] == {"width": 64, "height": 48}
    assert d["model"]["name"] == "fake"
    assert set(d["timing"]) == {"queue_ms", "inference_ms", "total_ms"}
    (inst,) = d["instances"]
    mask = decode_mask(inst)
    assert mask.shape == (48, 64)  # (h, w) khớp ảnh
    assert inst["box"] == {"x1": 4, "y1": 5, "x2": 20, "y2": 30}  # trái/trên incl, phải/dưới excl
    assert inst["area"] == int((mask == 255).sum()) == 16 * 25
    assert inst["score_type"] == "predicted_iou"
    assert fake.calls[-1][3] == (4.0, 5.0, 20.0, 30.0)


def test_instance_points_only(client):
    body = inst_body(points=[{"x": 10, "y": 10, "label": 1}, {"x": 50, "y": 40, "label": 0}])
    del body["box"]
    r = client.post("/v1/predict", json=body, headers=AUTH)
    assert r.status_code == 200 and len(r.json()["instances"]) == 1


def test_instance_multiple_boxes_return_ordered_masks_in_one_model_call(client, fake):
    boxes = [
        {"x1": 4, "y1": 5, "x2": 20, "y2": 30},
        {"x1": 30, "y1": 10, "x2": 45, "y2": 40},
    ]
    body = inst_body()
    body.pop("box")
    body["boxes"] = boxes
    r = client.post("/v1/predict", json=body, headers=AUTH)
    assert r.status_code == 200, r.text
    instances = r.json()["instances"]
    assert [item["prompt_id"] for item in instances] == [0, 1]
    assert [item["box"] for item in instances] == boxes
    assert [item["area"] for item in instances] == [16 * 25, 15 * 30]
    assert fake.calls[-1] == ("instances", (64, 48), [(4.0, 5.0, 20.0, 30.0), (30.0, 10.0, 45.0, 40.0)])


def test_instance_empty_result_is_empty_list(client, fake):
    original = fake.predict_instance
    fake.predict_instance = lambda *a: [RawInstance(np.zeros((48, 64), bool), 0.1, "predicted_iou")]
    try:
        r = client.post("/v1/predict", json=inst_body(), headers=AUTH)
    finally:
        fake.predict_instance = original
    assert r.status_code == 200 and r.json()["instances"] == []
    assert "batch" not in r.json()


def test_batch_explicit_empty_ids_and_reordered_results(client, fake, monkeypatch):
    body = inst_body()
    body.pop("box")
    body["boxes"] = [{"x1": 1, "y1": 1, "x2": 4, "y2": 4}] * 3
    empty = np.zeros((48, 64), bool)
    mask = empty.copy()
    mask[1:4, 1:4] = True
    monkeypatch.setattr(fake, "predict_instances", lambda *args: [
        RawInstance(mask, .9, "predicted_iou", 2),
        RawInstance(empty, .1, "predicted_iou", 1),
        RawInstance(mask, .9, "predicted_iou", 0),
    ])
    response = client.post("/v1/predict", json=body, headers=AUTH)
    assert response.status_code == 200, response.text
    assert response.json()["batch"] == {"count": 3, "empty_prompt_ids": [1]}
    assert [item["prompt_id"] for item in response.json()["instances"]] == [2, 0]


@pytest.mark.parametrize("ids", [[0], [0, 0], [0, 2], [None, 1], [False, 1]])
def test_batch_model_missing_duplicate_or_invalid_ids_fail_closed(client, fake, monkeypatch, ids):
    body = inst_body()
    body.pop("box")
    body["boxes"] = [{"x1": 1, "y1": 1, "x2": 4, "y2": 4}] * 2
    monkeypatch.setattr(fake, "predict_instances", lambda *args: [
        RawInstance(np.zeros((48, 64), bool), .1, "predicted_iou", i) for i in ids
    ])
    # TestClient normally re-raises server exceptions; assert the actual contract violation.
    with pytest.raises(RuntimeError, match="batch prompt IDs"):
        client.post("/v1/predict", json=body, headers=AUTH)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda b: b.pop("box"),  # không có prompt
        lambda b: (b.pop("box"), b.update(points=[{"x": 1, "y": 1, "label": 0}])),  # chỉ điểm âm
        lambda b: b.update(threshold=0.5),  # trường không hỗ trợ ở mode instance
        lambda b: b.update(text="car"),
        lambda b: b.update(points=[{"x": 1, "y": 1, "label": 2}]),
        lambda b: b.update(points=[{"x": 1, "y": 1, "label": True}]),
        lambda b: b.update(schema_version="2"),
        lambda b: b.update(request_id="has space"),
        lambda b: b.update(mode="nope"),
    ],
)
def test_invalid_requests_are_422_without_echoing_input(client, mutate):
    body = inst_body()
    mutate(body)
    r = client.post("/v1/predict", json=body, headers=AUTH)
    assert r.status_code == 422, r.text
    err = r.json()["error"]
    assert err["code"] == "invalid_request" and err["details"]
    assert body["image_base64"] not in r.text  # không phản hồi lại chuỗi base64


@pytest.mark.parametrize(
    "box",
    [
        {"x1": 20, "y1": 5, "x2": 4, "y2": 30},
        {"x1": 0, "y1": 0, "x2": 65, "y2": 10},
        {"x1": -1, "y1": 0, "x2": 10, "y2": 10},
        {"x1": 5, "y1": 5, "x2": 5, "y2": 9},
    ],
)
def test_bad_box_geometry_is_422(client, box):
    r = client.post("/v1/predict", json=inst_body(box=box), headers=AUTH)
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_prompt"


def test_bad_batch_box_geometry_is_422(client):
    body = inst_body()
    body.pop("box")
    body["boxes"] = [
        {"x1": 1, "y1": 1, "x2": 10, "y2": 10},
        {"x1": 20, "y1": 5, "x2": 4, "y2": 30},
    ]
    r = client.post("/v1/predict", json=body, headers=AUTH)
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_prompt"


def test_batch_boxes_cannot_mix_with_single_box_or_points(client):
    body = inst_body(boxes=[{"x1": 1, "y1": 1, "x2": 10, "y2": 10}])
    r = client.post("/v1/predict", json=body, headers=AUTH)
    assert r.status_code == 422
    body.pop("box")
    body["points"] = [{"x": 2, "y": 2, "label": 1}]
    r = client.post("/v1/predict", json=body, headers=AUTH)
    assert r.status_code == 422


def test_batch_boxes_are_limited_to_twelve(client):
    body = inst_body()
    body.pop("box")
    body["boxes"] = [{"x1": 1, "y1": 1, "x2": 10, "y2": 10}] * 13
    r = client.post("/v1/predict", json=body, headers=AUTH)
    assert r.status_code == 422


def test_point_outside_image_is_422_and_model_not_called(client, fake):
    n = len(fake.calls)
    r = client.post("/v1/predict", json=inst_body(points=[{"x": 64, "y": 10, "label": 1}]), headers=AUTH)
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_prompt"
    assert len(fake.calls) == n


def test_box_may_touch_right_bottom_edge(client):
    r = client.post("/v1/predict", json=inst_body(box={"x1": 0, "y1": 0, "x2": 64, "y2": 48}), headers=AUTH)
    assert r.status_code == 200
    assert r.json()["instances"][0]["box"] == {"x1": 0, "y1": 0, "x2": 64, "y2": 48}


def test_odd_image_size_and_exif_coordinate_system(client, fake):
    r = client.post("/v1/predict", json=inst_body(image_base64=b64_png(31, 17), box={"x1": 1, "y1": 1, "x2": 31, "y2": 17}), headers=AUTH)
    assert r.status_code == 200 and decode_mask(r.json()["instances"][0]).shape == (17, 31)

    img = Image.new("RGB", (60, 40))
    exif = Image.Exif()
    exif[0x0112] = 6
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    b64 = base64.b64encode(buf.getvalue()).decode()
    # Sau orientation ảnh là 40x60: box x2=50 phải bị từ chối, x2=40 hợp lệ.
    bad = client.post("/v1/predict", json=inst_body(image_base64=b64, box={"x1": 0, "y1": 0, "x2": 50, "y2": 10}), headers=AUTH)
    assert bad.status_code == 422
    ok = client.post("/v1/predict", json=inst_body(image_base64=b64, box={"x1": 0, "y1": 0, "x2": 40, "y2": 10}), headers=AUTH)
    assert ok.status_code == 200 and ok.json()["image"] == {"width": 40, "height": 60}
    assert decode_mask(ok.json()["instances"][0]).shape == (60, 40)


# ------------------------------------------------------------- concept mode
def test_concept_returns_sorted_instances_and_respects_limit(client, fake):
    r = client.post("/v1/predict", json=concept_body(threshold=0.3, max_instances=3), headers=AUTH)
    assert r.status_code == 200, r.text
    inst = r.json()["instances"]
    assert [i["score_type"] for i in inst] == ["detection_confidence"] * 3
    scores = [i["score"] for i in inst]
    assert scores == sorted(scores, reverse=True)
    assert fake.calls[-1][2:] == ("car", 0.3, 3)


def test_concept_defaults_come_from_config(client, fake):
    client.post("/v1/predict", json=concept_body(), headers=AUTH)
    assert fake.calls[-1][3] == 0.5 and fake.calls[-1][4] == 100


def test_concept_none_found_and_empty_masks_dropped(client):
    r = client.post("/v1/predict", json=concept_body(text="none"), headers=AUTH)
    assert r.status_code == 200 and r.json()["instances"] == []
    r = client.post("/v1/predict", json=concept_body(text="empty"), headers=AUTH)
    assert r.status_code == 200 and r.json()["instances"] == []


@pytest.mark.parametrize(
    "extra",
    [{"text": ""}, {"text": "   "}, {"text": "a\x00b"}, {"threshold": 0.0}, {"threshold": 1.0},
     {"max_instances": 0}, {"points": []}, {"box": {"x1": 0, "y1": 0, "x2": 5, "y2": 5}}],
)
def test_concept_invalid_inputs(client, extra):
    r = client.post("/v1/predict", json=concept_body(**extra), headers=AUTH)
    assert r.status_code == 422


def test_concept_limits_from_config(client):
    assert client.post("/v1/predict", json=concept_body(threshold=0.05), headers=AUTH).status_code == 422
    assert client.post("/v1/predict", json=concept_body(max_instances=101), headers=AUTH).status_code == 422


# ------------------------------------------------------------- ảnh/payload
@pytest.mark.parametrize(
    "b64,status,code",
    [("###", 400, "invalid_base64"),
     (base64.b64encode(b"not an image").decode(), 400, "invalid_image"),
     ("data:image/png;base64,AAAA", 400, "invalid_image")],
)
def test_bad_images(client, b64, status, code):
    r = client.post("/v1/predict", json=inst_body(image_base64=b64), headers=AUTH)
    assert r.status_code == status and r.json()["error"]["code"] == code


def test_body_too_large_streaming_and_content_length():
    s = make_settings(max_image_bytes=1024)
    with TestClient(create_app(s, model=FakeModel())) as c:
        wait_ready(c)
        big = inst_body(image_base64="A" * (s.max_body_bytes + 10))
        r = c.post("/v1/predict", json=big, headers=AUTH)
        assert r.status_code == 413 and r.json()["error"]["code"] == "payload_too_large"
        # Không có Content-Length (chunked): vẫn bị chặn khi đọc.
        r = c.post("/v1/predict", content=(chunk for chunk in [b"{" + b" " * (s.max_body_bytes + 10)]), headers={**AUTH, "Content-Type": "application/json"})
        assert r.status_code == 413


def test_image_pixel_limit_via_api():
    with TestClient(create_app(make_settings(max_image_pixels=1024), model=FakeModel())) as c:
        wait_ready(c)
        r = c.post("/v1/predict", json=inst_body(image_base64=b64_png(100, 100)), headers=AUTH)
        assert r.status_code == 413 and r.json()["error"]["code"] == "image_too_large"


# ---------------------------------------------------- lỗi, quá tải, timeout
def test_request_after_inference_error_still_works(client, fake):
    fake.raise_next = ValueError("boom with secret path /srv/x")
    r = client.post("/v1/predict", json=inst_body(), headers=AUTH)
    assert r.status_code == 500 and r.json()["error"]["code"] == "inference_failed"
    assert "secret" not in r.text
    assert client.post("/v1/predict", json=inst_body(), headers=AUTH).status_code == 200


def test_gpu_oom_maps_to_503_and_service_continues(client, fake):
    fake.raise_next = ApiError(503, "gpu_out_of_memory", "oom", headers={"Retry-After": "5"})
    r = client.post("/v1/predict", json=inst_body(), headers=AUTH)
    assert r.status_code == 503 and r.json()["error"]["code"] == "gpu_out_of_memory"
    assert r.headers["retry-after"] == "5"
    assert client.get("/readyz").status_code == 200
    assert client.post("/v1/predict", json=inst_body(), headers=AUTH).status_code == 200


def test_fatal_gpu_error_makes_service_not_ready_and_triggers_exit():
    stopped = []
    fake = FakeModel()
    app = create_app(make_settings(exit_on_fatal=True), model=fake, terminate=lambda: stopped.append(1))
    with TestClient(app) as c:
        wait_ready(c)
        fake.raise_next = ApiError(500, "gpu_fatal_error", "dead", fatal=True)
        assert c.post("/v1/predict", json=inst_body(), headers=AUTH).status_code == 500
        assert c.get("/readyz").status_code == 503
        assert c.post("/v1/predict", json=inst_body(), headers=AUTH).status_code == 503
    assert stopped == [1]


def test_queue_full_returns_503_with_retry_after_and_requests_never_overlap():
    fake = FakeModel(delay=0.4)
    with TestClient(create_app(make_settings(queue_size=1, request_timeout_s=30), model=fake)) as c:
        wait_ready(c)
        results = []

        def call():
            results.append(c.post("/v1/predict", json=inst_body(), headers=AUTH))

        threads = [threading.Thread(target=call) for _ in range(5)]
        for t in threads:
            t.start()
            time.sleep(0.05)
        for t in threads:
            t.join()
    codes = sorted(r.status_code for r in results)
    assert 200 in codes and 503 in codes
    rejected = [r for r in results if r.status_code == 503]
    assert all(r.json()["error"]["code"] == "queue_full" and r.headers["retry-after"] for r in rejected)
    assert fake.max_active == 1


def test_timeout_returns_504_and_next_request_waits_for_gpu():
    fake = FakeModel(delay=0.6)
    with TestClient(create_app(make_settings(request_timeout_s=0.3), model=fake)) as c:
        wait_ready(c)
        r = c.post("/v1/predict", json=inst_body(), headers=AUTH)
        assert r.status_code == 504 and r.json()["error"]["code"] == "timeout"
        # Lượt trước vẫn đang giữ model: request kế tiếp hết hạn khi chờ, không chen vào.
        r2 = c.post("/v1/predict", json=inst_body(), headers=AUTH)
        assert r2.status_code == 504
        time.sleep(0.7)  # chờ lượt chậm kết thúc
        fake.delay = 0.0
        assert c.post("/v1/predict", json=inst_body(), headers=AUTH).status_code == 200
    assert fake.max_active == 1


def test_unknown_route_uses_error_format(client):
    r = client.get("/nope")
    assert r.status_code == 404 and r.json()["error"]["code"] == "http_error"


def test_request_id_header_is_used_when_body_has_none(client):
    r = client.post("/v1/predict", json=inst_body(), headers={**AUTH, "X-Request-ID": "hdr-77"})
    assert r.json()["request_id"] == "hdr-77" and r.headers["x-request-id"] == "hdr-77"
    r = client.post("/v1/predict", json=inst_body(), headers=AUTH)
    assert len(r.json()["request_id"]) == 32
