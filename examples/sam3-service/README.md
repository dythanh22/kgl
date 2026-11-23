# sam3-service

Dịch vụ suy luận **SAM 3 (ảnh)** cho tích hợp CVAT (phương án A). Chỉ làm một việc:
**nhận ảnh + prompt, chạy inference, trả mask/box/score**. Không import Django/CVAT,
không biết tài khoản CVAT; chọn label, đổi sang CVAT RLE và ghi annotation thuộc
client ở P2/P3.

```
Máy A: CVAT → Nuclio adapter (CPU) ── HTTPS ──→ [reverse proxy TLS] → sam3-service → SAM 3 (GPU)
                                                                  Máy B
```

**Cài mới API trên GPU:** đọc [DEPLOYMENT.md](DEPLOYMENT.md) — Docker hoặc Python
venv, dùng workstation/server nội bộ/cloud; RunPod chỉ là một ví dụ tùy chọn.
**Cài toàn bộ CVAT custom:** đọc [hướng dẫn từ GitHub](https://github.com/dythanh22/cvat-sam3-custom/blob/sam3-custom/deployment/sam3/README.md).
**Mở lại mỗi buổi:** xem [hướng dẫn vận hành](https://github.com/dythanh22/cvat-sam3-custom/blob/sam3-custom/docs-vi/sam3/HUONG_DAN_KHOI_DONG_CVAT_SAM3.md).
README này giữ cấu trúc source, test và contract API; các lệnh triển khai duy trì ở
một nơi là `DEPLOYMENT.md`.

## Cấu trúc

```
src/sam3_service/
  config.py     cấu hình từ biến môi trường SAM3_*, kiểm tra khi khởi động
  schemas.py    request / response / lỗi + quy ước hình học
  images.py     decode ảnh, kiểm kích thước và prompt, encode mask PNG
  inference.py  hàng đợi hữu hạn + MỘT luồng thực thi tuần tự
  model.py      nạp SAM 3, inference instance / concept
  app.py        FastAPI, lifespan, xác thực, giới hạn body, định dạng lỗi
tests/          test_api / test_images / test_inference (không cần GPU), test_model_gpu (cần GPU)
```

## Chạy test

Chạy từ thư mục `examples/sam3-service`. Nhóm test adapter CVAT dùng source Nuclio của repo CVAT gốc và được bỏ qua nếu thiếu thư mục đó. Test API và xử lý ảnh chạy độc lập.

Không cần Torch/GPU (model giả được đưa vào app khi test; production luôn dùng model thật):

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -e ".[test]"
pytest                      # bỏ qua test GPU
```

Test GPU, chạy trên máy B (trong container hoặc venv đã có torch + sam3):

```bash
SAM3_TEST_CHECKPOINT=/srv/sam3/weights/sam3.pt pytest -m gpu -s tests/test_model_gpu.py
# tùy chọn: SAM3_TEST_CHECKPOINT_SHA256=..., SAM3_TEST_IMAGE=anh.jpg SAM3_TEST_TEXT=car
```

`-s` để thấy bảng latency / peak VRAM theo kích thước ảnh (ghi vào baseline).

## Triển khai trên máy B

Làm theo [DEPLOYMENT.md](DEPLOYMENT.md): chuẩn bị checkpoint/API key, chọn Docker
hoặc venv, kiểm CUDA/readiness, công bố HTTPS rồi đăng ký adapter trên máy CVAT.
Checkpoint nằm ngoài Git/image; một worker duy nhất. `/healthz` chỉ báo process
đang phản hồi, `/readyz` mới báo model nạp xong.

## Web demo

Open `/demo` (or the service root `/`) to try the running model before connecting CVAT.
The page is served by the API from the same origin, so it needs no separate frontend server.

1. Enter `SAM3_API_KEY` from the server's `.env`; the page keeps it in tab memory only.
2. Load a JPEG, PNG, or WEBP image (drag and drop also works).
3. In **Instance**, add positive/negative points for one mask, or draw up to 12 boxes and run them together.
4. In **Concept**, enter text, adjust threshold and maximum instances, then run to get masks and scores.
5. Toggle result checkboxes to compare overlays. The page shows boxes, area, confidence, model revision,
   queue time, inference time, and total time.

`/healthz` and `/readyz` remain public for status checks. `/v1/*` requests from the page still require
Bearer authentication. Serve the page over HTTPS before entering the API key. The demo uses the API's
current schema; it does not write annotations to CVAT.

### Mở demo trên máy local khi SAM3 chạy trên Kaggle

Xem [hướng dẫn SAM3 trên Kaggle](../serving-sam3-on-kaggle.md) để cài runtime, nạp checkpoint, thử API và lấy Bearer key từ máy local.

Khởi động Uvicorn trên `127.0.0.1:8013` trong notebook và chờ `/readyz` trả `ready`.
Từ thư mục gốc của repo kgl, chạy với cùng môi trường và `KGL_HOME` đã dùng để kết nối:

```bash
KGL_REQUEST_BYTES=33554432 kgl forward --remote-port 8013 --listen 127.0.0.1:8014
```

Mở [http://127.0.0.1:8014/demo](http://127.0.0.1:8014/demo), nhập `SAM3_API_KEY`
từ file `.env` trên notebook, rồi chọn ảnh và prompt. Dùng `--remote-port` nếu Uvicorn
đang nghe trên cổng khác, hoặc `--listen` để đổi địa chỉ local. Cầu nối phục vụ trang demo
cùng origin và chuyển `/healthz`, `/readyz`, `/v1/predict` qua phiên Jupyter.
Request và response inference được truyền theo chunk. Ảnh lớn và nhiều mask có thể
kéo dài thời gian chờ. Ctrl+C dừng cầu nối local. Dừng Uvicorn riêng trên notebook
khi kết thúc.

## API

Mọi endpoint `/v1/*` cần `Authorization: Bearer <SAM3_API_KEY>` (kiểm tra **trước** khi đọc body).
`/healthz` và `/readyz` không cần xác thực và không trả thông tin model.

| Endpoint | Ý nghĩa |
|---|---|
| `GET /healthz` | Process đang phản hồi |
| `GET /readyz` | 200 khi model đã nạp và chưa gặp lỗi nặng; 503 `loading` / `failed` / `fatal` |
| `POST /v1/predict` | Inference |

Header `X-Request-ID` (nếu hợp lệ) hoặc `request_id` trong body được dùng làm ID truy vết; không có thì server tự sinh. ID luôn có trong response, header `X-Request-ID` và log.

### Request

Trường chung: `schema_version` (= `"1"`), `request_id` (tùy chọn, `[A-Za-z0-9._:-]{1,64}`),
`image_base64` (base64 thuần của file JPEG/PNG/WEBP, không có tiền tố `data:`). Trường lạ bị từ chối (422).

**`mode: "instance"`** — điểm dương/âm và/hoặc một box → tối đa **một** instance. Cần ít nhất một box hoặc một điểm dương. Có thể gửi `boxes` (1–12 box) để chạy nhiều vùng độc lập trong **một request**; không kết hợp `boxes` với `box` hoặc `points`.

```json
{"schema_version": "1", "mode": "instance", "image_base64": "...",
 "points": [{"x": 120.5, "y": 80, "label": 1}, {"x": 300, "y": 40, "label": 0}],
 "box": {"x1": 100, "y1": 60, "x2": 220, "y2": 190}}
```

Batch box dùng `prompt_id` bắt đầu từ 0 để ghép với box tương ứng. Ảnh được mã hóa backbone một lần rồi tái sử dụng cho các box trong batch. Mỗi box vẫn chạy một lượt phân đoạn riêng.

```json
{"schema_version": "1", "mode": "instance", "image_base64": "...",
 "boxes": [{"x1": 100, "y1": 60, "x2": 220, "y2": 190},
           {"x1": 300, "y1": 80, "x2": 420, "y2": 210}]}
```

**`mode: "concept"`** — text → danh sách instance.

```json
{"schema_version": "1", "mode": "concept", "image_base64": "...",
 "text": "car", "threshold": 0.5, "max_instances": 50}
```

`threshold` và `max_instances` chỉ có ở concept (gửi vào instance → 422). Box exemplar cho concept chưa hỗ trợ.

### Quy ước hình học

- Tọa độ tính trên ảnh **sau khi áp dụng EXIF orientation**, đã chuyển RGB (kênh alpha bị bỏ). `image.width/height` trong response là kích thước đó và **mọi mask có đúng kích thước đó**. Client phải gửi tọa độ trên ảnh đã xoay; nếu CVAT gửi frame đã chuẩn hóa orientation thì khớp trực tiếp.
- Pixel `i` chiếm đoạn `[i, i+1)`. Điểm hợp lệ: `0 ≤ x < width`, `0 ≤ y < height`.
- Box: `x1, y1` inclusive; `x2, y2` exclusive; `0 ≤ x1 < x2 ≤ width`, `0 ≤ y1 < y2 ≤ height`.
- Box trong response là **hộp bao chặt của mask** (số nguyên, cùng quy ước). Mask rỗng bị loại khỏi `instances`.

### Chọn candidate (instance)

Đúng một điểm đơn lẻ (không box) → SAM sinh 3 mask, lấy mask có IoU dự đoán cao nhất. Có box, hoặc từ 2 prompt
trở lên → `multimask_output=False`, một mask. `score` là IoU do model dự đoán (`score_type: predicted_iou`).
SAM 3 tự lấp lỗ nhỏ (≤ 256 px ở độ phân giải thấp 256×256) trong mask instance; đây là hành vi upstream.

### Concept

`threshold` là ngưỡng `confidence_threshold` của processor (điểm = sigmoid(logit) × presence), trong
`[SAM3_MIN_THRESHOLD, 1)`. Kết quả sắp theo score giảm dần, cắt còn `max_instances`
(`score_type: detection_confidence`). Ngưỡng thấp làm processor nội suy nhiều mask lên độ phân giải gốc nên tốn
VRAM; vì vậy có sàn `SAM3_MIN_THRESHOLD`.

### Response

```json
{"schema_version": "1", "request_id": "...", "mode": "instance",
 "image": {"width": 640, "height": 480},
 "model": {"name": "sam3-image", "source_sha": "...", "checkpoint_revision": "...",
           "checkpoint_sha256": "...|null", "precision": "fp32", "device": "cuda"},
 "timing": {"queue_ms": 0.4, "inference_ms": 812.3, "total_ms": 861.0},
 "instances": [{"mask_png_base64": "...", "box": {"x1": 100, "y1": 60, "x2": 220, "y2": 190},
                "area": 10231, "score": 0.93, "score_type": "predicted_iou"}]}
```

`mask_png_base64`: PNG 8-bit grayscale, đúng kích thước ảnh, 255 = đối tượng, 0 = nền.
Không có kết quả → `instances: []` (HTTP 200).

Chỉ response của request `boxes` có thêm `batch: {"count": N, "empty_prompt_ids": [...]}`.
Các instance có mask dùng `prompt_id`; mask rỗng không nằm trong `instances` nhưng ID
phải có trong `empty_prompt_ids`. Hai tập ID không trùng/giao nhau và hợp lại phải
bằng chính xác `0..N-1`. Không suy ID thiếu là mask rỗng. Nếu model trả thiếu/trùng/sai
ID, API trả lỗi 500; client không được Accept một phần. Response single/concept không
có trường `batch`.

### Lỗi

Định dạng: `{"error": {"code", "message", "request_id", "details"?}}`. Response lỗi 422 không chứa lại dữ liệu đầu vào.

| HTTP | `code` | Khi nào |
|---|---|---|
| 400 | `invalid_base64`, `invalid_image` | base64/ảnh hỏng, cắt cụt, định dạng ngoài JPEG/PNG/WEBP, ảnh nhiều khung |
| 401 | `unauthorized` | thiếu/sai API key |
| 413 | `payload_too_large`, `image_too_large` | vượt `SAM3_MAX_IMAGE_BYTES` / pixel / cạnh |
| 422 | `invalid_request`, `invalid_prompt` | sai schema; prompt ngoài ảnh, box suy biến, threshold ngoài khoảng |
| 503 | `queue_full` (+`Retry-After`) | hàng đợi đầy |
| 503 | `not_ready` | model đang nạp / đang dừng / sau lỗi nặng |
| 503 | `gpu_out_of_memory` | OOM; dịch vụ tiếp tục phục vụ |
| 504 | `timeout` | quá `SAM3_REQUEST_TIMEOUT_S` (gồm thời gian chờ hàng đợi) |
| 500 | `inference_failed`, `gpu_fatal_error` | lỗi suy luận; lỗi CUDA nặng làm process thoát để restart |

## Thực thi

- Một process, một model, một GPU. Hàng đợi hữu hạn (`SAM3_QUEUE_SIZE`) và **một luồng** làm mọi việc với
  model/processor/predictor, kể cả nạp model. Decode/encode ảnh chạy ở threadpool riêng, không chặn event loop
  và không chiếm luồng GPU.
- **Timeout HTTP không dừng được GPU.** Request đang chạy mà client ngắt hoặc hết hạn vẫn giữ luồng đến hết lượt
  đó; request kế tiếp không chen vào. Request còn nằm trong hàng đợi mà bị hủy/hết hạn thì không chạy.
- Mỗi request tạo state ảnh mới; cuối request luôn đặt lại predictor SAM1-task và ngưỡng của processor, kể cả khi lỗi.
- Nạp checkpoint: `weights_only=True`, ánh xạ key giống `_load_checkpoint` của SAM 3 nhưng **dừng nếu thiếu key**
  (SAM 3 nạp `strict=False` chỉ để in cảnh báo). Có thể xác minh SHA-256 file lúc khởi động.

## Cấu hình

Xem `.env.example` (đủ mô tả từng biến). Bắt buộc: `SAM3_API_KEY` (≥ 24 ký tự) trừ khi `SAM3_AUTH_DISABLED=true`
(chỉ phát triển). Thiếu/ngắn → dịch vụ không khởi động.

## Đã kiểm chứng và còn cần kiểm tra

**Bằng chứng đã ghi đến 08/10/2026:** Docker image đã build thành công; model chạy với checkpoint đã pin
trên Tesla T4, gồm 13 test GPU thật và inference ở bốn kích thước ảnh. API còn được gọi qua HTTPS từ một
GPU RTX A4000; point/box, concept, auth và response đã được kiểm tra. Chi tiết cấu hình, giới hạn phép đo
và fixtures nằm trong [`docs-vi/baseline.md`](https://github.com/dythanh22/cvat-sam3-custom/blob/sam3-custom/docs-vi/baseline.md) và
[`bàn giao P1`](https://github.com/dythanh22/cvat-sam3-custom/blob/sam3-custom/docs-vi/sam3/P1_OFFLINE_HANDOFF.md).

**Trước khi dùng server đích:** build lại image với source hiện tại; kiểm tra NVIDIA Container Toolkit,
GPU/VRAM, readiness, SHA checkpoint, API key, instance và concept trên ảnh đại diện; kiểm DNS/TLS/timeout
từ máy A qua reverse proxy. P3/P4a V2 đã có bằng chứng tích hợp CVAT/Nuclio và UI trong
[báo cáo rollout](https://github.com/dythanh22/cvat-sam3-custom/blob/sam3-custom/docs-vi/sam3/P4A_V2_ROLLOUT.md). Chưa có cam kết p95 hoặc
đánh giá chất lượng trên mọi dataset/môi trường đích. Bằng chứng GPU cũ không thay thế các
kiểm tra trên server đích.
