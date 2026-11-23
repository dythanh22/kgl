"""Batch adapter contract tests; no Nuclio, CVAT database or GPU required."""
import base64
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from PIL import Image
import pytest

ADAPTER = Path(__file__).resolve().parents[2] / "serverless/pytorch/meta/sam3/nuclio"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


if not ADAPTER.is_dir():
    pytest.skip("external CVAT/Nuclio adapter checkout is unavailable", allow_module_level=True)

single = load("main", ADAPTER / "main.py")
batch = load("sam3_batch_adapter", ADAPTER / "batch.py")


def png(mask=False):
    stream = io.BytesIO()
    image = Image.new("L" if mask else "RGB", (8, 6), 255)
    image.save(stream, format="PNG")
    return base64.b64encode(stream.getvalue()).decode()


def result():
    return {"image": {"width": 8, "height": 6},
            "batch": {"count": 2, "empty_prompt_ids": [1]},
            "instances": [{"prompt_id": 0, "mask_png_base64": png(True)}]}


def test_results_keep_stable_ids_and_explicit_empty():
    output = batch._results(result(), ["box-9", "box-2"])
    assert output[0]["id"] == "box-9" and output[0]["status"] == "mask"
    assert output[0]["shapes"][0]["points"] == [0, 48, 0, 0, 7, 5]
    assert output[1] == {"id": "box-2", "status": "empty", "shapes": []}


@pytest.mark.parametrize("mutate", [
    lambda r: r.pop("batch"),
    lambda r: r["batch"].update(count=3),
    lambda r: r["batch"].update(empty_prompt_ids=[]),
    lambda r: r["batch"].update(empty_prompt_ids=[1, 1]),
    lambda r: r["batch"].update(empty_prompt_ids=[0]),
    lambda r: r["batch"].update(empty_prompt_ids=[True]),
    lambda r: r["instances"].append(copy.deepcopy(r["instances"][0])),
    lambda r: r["instances"][0].update(prompt_id=2),
    lambda r: r["instances"][0].update(prompt_id=False),
    lambda r: r["instances"].clear(),
])
def test_incomplete_or_duplicate_results_fail_closed(mutate):
    response = result()
    mutate(response)
    with pytest.raises(single.AdapterError) as error:
        batch._results(response, ["a", "b"])
    assert error.value.status == 502


@pytest.mark.parametrize("boxes", [[], [{}] * 13,
    [{"id": "a", "box": [[0, 0], [9, 6]]}],
    [{"id": "a", "box": [[0, 0], [0, 6]]}],
    [{"id": "a", "box": [[float("nan"), 0], [8, 6]]}],
    [{"id": "a", "box": [[False, 0], [8, 6]]}],
    [{"id": "a", "box": [[0, 0], [8, 6]]}] * 2,
])
def test_bad_boxes_rejected(boxes):
    with pytest.raises(single.AdapterError) as error:
        batch._boxes({"boxes": boxes}, 8, 6)
    assert error.value.status == 400


def test_handler_calls_gpu_once_and_returns_every_id(monkeypatch):
    calls = []
    def request(*args, **kwargs):
        calls.append((args, kwargs))
        return result()
    monkeypatch.setattr(single, "_request_sam3", request)
    context = SimpleNamespace(Response=lambda **kwargs: kwargs)
    prompts = [{"id": name, "box": [[0, 0], [8, 6]]} for name in ("a", "b")]
    response = batch.handler(context, SimpleNamespace(body={"image": png(), "boxes": prompts}))
    assert response["status_code"] == 200
    assert len(calls) == 1 and len(calls[0][1]["boxes"]) == 2
    assert len(json.loads(response["body"])["results"]) == 2


@pytest.mark.parametrize("mismatch", [False, True])
def test_batch_http_payload_and_response_correlation(monkeypatch, mismatch):
    monkeypatch.setenv("SAM3_API_URL", "https://sam3.example")
    monkeypatch.setenv("SAM3_API_TOKEN", "test-token")
    boxes = [{"x1": 0, "y1": 0, "x2": 8, "y2": 6}]
    calls = []

    def open_request(request, timeout):
        payload = json.loads(request.data)
        calls.append(payload)
        assert payload["boxes"] == boxes
        assert "box" not in payload and "points" not in payload
        assert timeout == 65
        reply = {"mode": "instance", "schema_version": "1",
                 "request_id": "wrong" if mismatch else payload["request_id"],
                 "image": {"width": 8, "height": 6}, "instances": []}
        return io.BytesIO(json.dumps(reply).encode())

    monkeypatch.setattr(single.urllib.request, "urlopen", open_request)
    if mismatch:
        with pytest.raises(single.AdapterError) as error:
            single._request_sam3(base64.b64decode(png()), [], None, boxes=boxes)
        assert error.value.status == 502
    else:
        single._request_sam3(base64.b64decode(png()), [], None, boxes=boxes)
    assert len(calls) == 1


def test_timeout_is_504_and_not_retried(monkeypatch):
    monkeypatch.setenv("SAM3_API_URL", "https://sam3.example")
    monkeypatch.setenv("SAM3_API_TOKEN", "test-token")
    calls = []

    def timeout(*args, **kwargs):
        calls.append(1)
        raise TimeoutError()

    monkeypatch.setattr(single.urllib.request, "urlopen", timeout)
    with pytest.raises(single.AdapterError) as error:
        single._request_sam3(base64.b64decode(png()), [], None, boxes=[{}])
    assert error.value.status == 504 and len(calls) == 1
