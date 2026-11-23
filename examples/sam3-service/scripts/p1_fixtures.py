"""Capture real P1 HTTP exchanges, verify masks, or replay them without a GPU.

Capture runs on the GPU host; verify/serve run locally. No credentials are saved.
"""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def request_body(root, case):
    body = dict(case["body"])
    if case.get("image_file"):
        body["image_base64"] = base64.b64encode((root / case["image_file"]).read_bytes()).decode()
    return body


def capture(args):
    root = args.directory
    root.mkdir(parents=True, exist_ok=False)
    (root / "images").mkdir()
    (root / "cases").mkdir()
    container = args.container

    def docker_python(code):
        return subprocess.check_output(["docker", "exec", container, "python", "-c", code], text=True)

    key = docker_python("import os; print(os.environ['SAM3_API_KEY'], end='')")
    metadata = json.loads(docker_python(
        "import json,importlib.metadata as m; from sam3_service.config import get_settings; "
        "s=get_settings(); print(json.dumps({'settings':s.model_dump(mode='json',exclude={'api_key'}),"
        "'packages':{p:m.version(p) for p in ['torch','torchvision','sam3','fastapi','pydantic','pillow','numpy']}}))"
    ))
    metadata["image_id"] = subprocess.check_output(
        ["docker", "inspect", "--format", "{{.Image}}", container], text=True).strip()
    metadata["captured_at_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["transport"] = "HTTP loopback on GPU host; excludes machine A/network/TLS latency"
    metadata["source_sha256"] = {
        str(p.relative_to(args.service_root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((args.service_root / "src/sam3_service").glob("*.py"))
    }
    dump(root / "provenance.json", metadata)
    traffic = "images/traffic-in-vietnam-banner.jpg"
    leaf = "images/bacterial-leaf-spot-780x520.jpeg"
    for name in (traffic, leaf):
        shutil.copyfile(args.service_root / "test-images-seg" / Path(name).name, root / name)
    box = dict(x1=1110, y1=710, x2=1360, y2=900)
    pos = dict(x=1250, y=820, label=1)
    neg = dict(x=1400, y=780, label=0)
    specs = [
        ("instance_box", traffic, "instance", {"box": box}, 200, "valid"),
        ("instance_point", traffic, "instance", {"points": [pos]}, 200, "valid"),
        ("instance_refine", traffic, "instance", {"points": [pos, neg]}, 200, "valid"),
        ("instance_point_removed", traffic, "instance", {"points": [pos]}, 200, "valid"),
        ("concept_car", traffic, "concept", {"text": "car", "threshold": .4, "max_instances": 10}, 200, "valid"),
        ("concept_bus", traffic, "concept", {"text": "bus", "threshold": .4, "max_instances": 10}, 200, "valid"),
        ("instance_leaf", leaf, "instance", {"box": dict(x1=100,y1=100,x2=500,y2=400)}, 200, "valid"),
        ("instance_box_repeat", traffic, "instance", {"box": box}, 200, "valid"),
        ("instance_batch", traffic, "instance", {"boxes": [dict(x1=490,y1=250,x2=620,y2=420), box]}, 200, "valid"),
        ("concept_empty", traffic, "concept", {"text": "qzxv_nonexistent_object_7391", "threshold": .99, "max_instances": 5}, 200, "valid"),
        ("no_prompt", traffic, "instance", {}, 422, "valid"),
        ("bad_box", traffic, "instance", {"box": dict(x1=20,y1=10,x2=5,y2=30)}, 422, "valid"),
        ("threshold_zero", traffic, "concept", {"text": "car", "threshold": 0}, 422, "valid"),
        ("unsupported_mode", traffic, "unknown", {}, 422, "valid"),
        ("invalid_base64", None, "instance", {"image_base64": "!!!", "box": box}, 400, "valid"),
        ("unauthorized", traffic, "instance", {"box": box}, 401, "missing"),
    ]
    manifest = {"format_version": 1, "origin": "real_gpu_http", "cases": []}
    for name, image_file, mode, extra, expected, auth in specs:
        case = {"name": name, "origin": "real_gpu_http", "method": "POST", "path": "/v1/predict",
                "auth": auth, "image_file": image_file,
                "body": {"schema_version": "1", "request_id": "fixture-" + uuid.uuid4().hex, "mode": mode, **extra}}
        headers = {"Content-Type": "application/json"}
        if auth == "valid":
            headers["Authorization"] = "Bearer " + key
        data = json.dumps(request_body(root, case)).encode()
        req = urllib.request.Request(args.url + case["path"], data=data, headers=headers)
        started = time.perf_counter()
        try:
            response = urllib.request.urlopen(req, timeout=150)
        except urllib.error.HTTPError as exc:
            response = exc
        with response:
            raw = response.read()
            case.update(status=response.status, response=json.loads(raw), request_bytes=len(data),
                        response_bytes=len(raw), wall_ms=round((time.perf_counter()-started)*1000, 2),
                        response_headers={k: v for k, v in response.headers.items()
                                          if k.lower() in ("content-type", "x-request-id", "retry-after", "www-authenticate")})
        assert case["status"] == expected, (name, case["status"])
        # Fail closed if a server response unexpectedly contains the credential.
        assert key not in json.dumps(case)
        dump(root / "cases" / (name + ".json"), case)
        manifest["cases"].append("cases/" + name + ".json")
        print(name, case["status"], len(case["response"].get("instances", [])), flush=True)
    for path, filename in (("/v1/openapi.json", "openapi.json"), ("/readyz", "readyz.json")):
        req = urllib.request.Request(args.url + path, headers={"Authorization": "Bearer " + key})
        with urllib.request.urlopen(req, timeout=20) as response:
            dump(root / filename, json.load(response))
    dump(root / "manifest.json", manifest)
    dump(root / "checksums.json", {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*")) if p.is_file()
    })


def verify(args):
    import importlib.util
    import io
    import numpy as np
    from PIL import Image, ImageOps

    root = args.directory
    for name, checksum in json.loads((root / "checksums.json").read_text()).items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == checksum, name
    sdk_path = Path(__file__).resolve().parents[2] / "cvat-sdk/cvat_sdk/masks.py"
    spec = importlib.util.spec_from_file_location("cvat_mask_fixture_check", sdk_path)
    masks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(masks)
    cases = {}
    count = 0
    converted = {}
    png_artifacts = {}
    for file in json.loads((root / "manifest.json").read_text())["cases"]:
        case = json.loads((root / file).read_text())
        cases[case["name"]] = case
        response = case["response"]
        if case["status"] != 200:
            assert "error" in response and response["error"]["code"]
            continue
        with Image.open(root / case["image_file"]) as source:
            size = ImageOps.exif_transpose(source).size
        assert size == (response["image"]["width"], response["image"]["height"])
        points_list = []
        for index, item in enumerate(response["instances"]):
            png = base64.b64decode(item["mask_png_base64"], validate=True)
            png_artifacts[f"{case['name']}-{index}.png"] = png
            with Image.open(io.BytesIO(png)) as im:
                assert im.mode == "L" and im.size == size
                arr = np.array(im)
            assert set(np.unique(arr)).issubset({0, 255})
            bitmap = arr == 255
            assert int(bitmap.sum()) == item["area"] > 0
            ys, xs = bitmap.nonzero()
            assert item["box"] == dict(x1=int(xs.min()),y1=int(ys.min()),x2=int(xs.max()+1),y2=int(ys.max()+1))
            points = masks.encode_mask(bitmap)
            assert np.array_equal(bitmap, masks.decode_mask(points, image_width=size[0], image_height=size[1]))
            points_list.append(points)
            count += 1
        converted[case["name"]] = points_list
    for a, b in (("instance_box", "instance_box_repeat"), ("instance_point", "instance_point_removed")):
        assert cases[a]["response"]["instances"] == cases[b]["response"]["instances"], (a, b)
    assert cases["concept_empty"]["response"]["instances"] == []
    assert [i["prompt_id"] for i in cases["instance_batch"]["response"]["instances"]] == [0, 1]
    assert cases["concept_car"]["response"]["instances"]
    assert cases["concept_bus"]["response"]["instances"]
    geometry = {}
    for name in ("empty", "single_pixel_bottom_right", "full_border", "hole_and_disconnected"):
        bitmap = np.zeros((7, 9), dtype=bool)
        if name == "single_pixel_bottom_right":
            bitmap[-1, -1] = True
        elif name == "full_border":
            bitmap[:] = True
        elif name == "hole_and_disconnected":
            bitmap[1:6, 1:6] = True
            bitmap[2:5, 2:5] = False
            bitmap[6, 8] = True
        points = masks.encode_mask(bitmap)
        assert np.array_equal(bitmap, masks.decode_mask(points, image_width=9, image_height=7))
        geometry[name] = {"origin": "synthetic_geometry_not_model_output", "width": 9, "height": 7,
                          "bitmap_rows": bitmap.astype(int).tolist(), "mask_points": points,
                          "client_should_drop": not bool(bitmap.any())}
    if args.export:
        args.export.mkdir(parents=True, exist_ok=False)
        for name, points in converted.items():
            dump(args.export / (name + ".json"), {"origin": "derived_locally_not_cvat_response", "mask_points": points})
        for name, png in png_artifacts.items():
            (args.export / name).write_bytes(png)
        dump(args.export / "synthetic_geometry.json", geometry)
    print(f"PASS: {len(cases)} captured cases; {count} masks + 4 synthetic geometries; PNG/area/bbox and CVAT RLE round-trip pixel-exact")


def serve(args):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    root = args.directory
    manifest = json.loads((root / "manifest.json").read_text())
    cases = [json.loads((root / p).read_text()) for p in manifest["cases"]]

    def normalize(body):
        return {k: v for k, v in body.items() if k != "request_id"}

    indexed = [(case, normalize(request_body(root, case))) for case in cases]

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status, body, headers=None):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            for key, value in (headers or {}).items():
                if key.lower() != "content-type":
                    self.send_header(key, value)
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ("/healthz", "/readyz"):
                self.send_json(200, {"status": "ready" if self.path == "/readyz" else "ok"})
            else:
                self.send_json(404, {"error": {"code": "fixture_not_found"}})

        def do_POST(self):
            size = int(self.headers.get("Content-Length", "0"))
            if size > 30_000_000:
                self.send_json(413, {"error": {"code": "replay_body_limit"}})
                return
            try:
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise ValueError("object required")
            except ValueError:
                self.send_json(400, {"error": {"code": "invalid_replay_json"}})
                return
            auth = "valid" if self.headers.get("Authorization") == "Bearer offline-fixture-key" else "missing"
            for case, expected in indexed:
                if self.path == case["path"] and auth == case["auth"] and normalize(body) == expected:
                    self.send_json(case["status"], copy.deepcopy(case["response"]), case["response_headers"])
                    return
            self.send_json(409, {"error": {"code": "fixture_not_found", "message": "Only recorded requests are supported."}})

    print(f"OFFLINE REPLAY http://127.0.0.1:{args.port}; token=offline-fixture-key; recorded IDs/timings, no inference", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["capture", "verify", "serve"])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--service-root", type=Path, default=Path.cwd())
    parser.add_argument("--container", default="sam3-service-sam3-1")
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--export", type=Path)
    args = parser.parse_args()
    {"capture": capture, "verify": verify, "serve": serve}[args.action](args)
