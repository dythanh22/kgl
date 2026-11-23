# Tự kiểm thử kgl từ máy cá nhân đến GPU Kaggle

**Câu hỏi nghiệm thu:** Từ máy cá nhân, bạn có nhận được câu trả lời của đúng model trên GPU Kaggle, nhận đủ stream, và gửi được yêu cầu tiếp theo sau gián đoạn không?


Các lệnh dùng Bash trên Linux. Chạy từng khối trong terminal A trên máy cá nhân. Bước HTTP cần thêm terminal B. Chọn notebook thử nghiệm cho các ca ngắt WS hoặc restart server.

Lưu số liệu và điều kiện chạy theo ngày trong [thư mục báo cáo](reports/).

## Nghiệm thu demo trên laptop

Chạy lần lượt chat, classifier Transformers và classifier với checkpoint riêng theo [hướng dẫn demo](../examples/demos.md). Trên cả ba luồng, kiểm tra trang hiển thị đúng model và thiết bị. Chat đạt khi trả đủ stream. Classifier đạt khi xử lý hai ảnh liên tiếp, từ chối ảnh lỗi và trả kết quả hợp lệ ở lượt kế tiếp. Với ví dụ hai nhãn, ảnh đỏ và xanh dương phải cho nhãn đứng đầu tương ứng.

Thử gửi thêm một yêu cầu khi worker đang bận và kiểm tra thông báo. Với phiên thử riêng, dừng notebook trong lúc demo đang mở, kiểm tra hướng dẫn kết nối lại rồi mở phiên mới và gửi một yêu cầu thành công. [Kết quả hiện có](reports/2026-11-19-v2.md#demo-trên-laptop-cho-cá-nhân-và-sinh-viên)

## 1. Cài đặt và lập mốc local

Mở terminal tại thư mục dự án chứa đủ các đường dẫn bên dưới. Nếu thiếu, lấy bản source đầy đủ trước khi cài. Chuẩn bị môi trường Python ≥ 3.10 có pip, curl, cùng quyền mở cổng localhost cho test. pip có thể cần Internet để lấy build dependency.

```bash
ls pyproject.toml src/kgl/cli/app.py tests/fake_jupyter.py
python -m pip install -e .
kgl --version
set -o pipefail
RUN=$(mktemp -d "$HOME/kgl-test.XXXXXX")
export KGL_HOME="$RUN/session"       # lần thử độc lập, hoặc dùng lại thư mục session còn hiệu lực
export KGL_WORKDIR=/kaggle/working KGL_CHUNK_KB=256
printf 'Thu muc ket qua: %s\n' "$RUN"
python -V
PYTHONPATH=src python -m unittest discover -s tests -t . -v 2>&1 | tee "$RUN/local-tests.log"
```

**Đạt** khi unittest báo OK. Jupyter Server và ipykernel là dependency tùy chọn của bài test local với kernel thật. Nếu thấy skipped, kiểm tra lý do trước khi kết luận.

Để dùng lại phiên đã lưu của bạn, đặt `KGL_HOME` tới thư mục session tương ứng rồi chạy `kgl exec --timeout 15 'print(2)'`. Nếu nhận được 2, tiếp tục bước 3. Khi URL hết hạn, thực hiện bước 2 và nhập lại qua stdin.

### Thử nhanh bản sửa với session còn hiệu lực

Tại thư mục dự án, dùng lệnh dưới đây sau khi đã kiểm tra session. Bài thử dùng thêm VRAM cho một Qwen 0.5B trên cổng và thư mục riêng của VM, rồi dừng instance do bài thử tạo. Nó giữ session notebook đang chạy. Báo cáo JSON ghi device/dtype, 20 request, độ trễ, kết quả ngắt WS sau chunk có nội dung và kết quả dừng server. Phép thử này cần mạng tới Kaggle và quyền tải hoặc cache model trên VM.

```bash
python -m tests.kaggle_smoke
```

Để nghiệm thu hủy v2, đặt thêm `KGL_CONTROL_KERNEL_ID` bằng ID đầy đủ của kernel phụ đang chạy trong cùng notebook thử. Smoke test yêu cầu xác nhận worker đã nhả lượt sau khi cắt stream và yêu cầu tiếp theo thành công. Nếu không đặt, báo cáo ghi `cancellation_confirmed: null`. Trong một ca riêng, giữ một cell thử nghiệm đang chạy trên kernel notebook và xác minh lệnh hủy inference không ngắt cell đó.

## 2. Lấy URL và chứng minh kết nối

Tạo notebook Python trong tài khoản Kaggle của bạn, chọn accelerator GPU và khởi động session. Bật Internet nếu tải model từ Hugging Face, hoặc gắn sẵn thư mục model có tokenizer và weights. Trong notebook chọn **Run → Kaggle Jupyter Server → Manually Connect → VS Code Compatible URL**. Đây là luồng lấy URL mà Kaggle mô tả cho trình soạn thảo bên ngoài. Khả năng dùng URL đó với bản kgl của bạn được kiểm tra ngay bên dưới. [Tài liệu Kaggle](https://www.kaggle.com/docs/notebooks)

Chạy riêng dòng sau trong Bash và nhấn Enter. Khi thấy lời nhắc, sao chép URL từ Kaggle, quay về terminal, dán bằng Ctrl+Shift+V hoặc chuột phải → Paste rồi nhấn Enter. Tùy chọn `-s` ẩn toàn bộ ký tự khi dán, kể cả dấu sao.

```bash
IFS= read -rsp 'Dan URL (an ky tu), roi nhan Enter: ' KGL_TEST_URL
```

Khi dấu nhắc shell trở lại, kiểm tra giá trị mà không in URL. Nếu kết quả là `0`, sao chép lại URL và chạy lại riêng dòng nhập trên.

```bash
printf 'Da nhan %s ky tu.\n' "${#KGL_TEST_URL}"
```

Khi độ dài lớn hơn 0, tiếp tục trong cùng terminal.

```bash
kgl auth
printf '%s\n' "$KGL_TEST_URL" | kgl connect --proxy-url -
unset KGL_TEST_URL
kgl exec --timeout 30 'print(1 + 1)'
kgl status --json > "$RUN/status.json"
```

**Đạt** khi connect hoàn tất, exec trả đúng `2` và trường `connection` trong JSON là `ok`. `auth` hướng dẫn lấy URL kết nối. Dòng `connected` xuất hiện sau khi probe kernel thành công. Nếu có nhiều kernel, chọn kernel notebook bằng `--kernel-id` khi connect. Với 401/403, lấy URL mới từ phiên còn chạy. Với 404, kiểm tra loại URL. Thử cùng URL trong VS Code để phân biệt lỗi phiên với lỗi xử lý URL của kgl. Nếu VS Code chạy cell được nhưng kgl lỗi, ghi mã HTTP và giai đoạn REST/WS/exec rồi dừng thử model.

## 3. Kiểm tra thực thi, trạng thái kernel và truyền file

```bash
kgl exec --timeout 30 'kgl_test_value = 41'
kgl exec --timeout 30 'print(kgl_test_value + 1)'
kgl exec --timeout 30 <<'PY'
import time
for i in range(3):
    print(i, flush=True)
    time.sleep(1)
PY
kgl exec 'raise ValueError("kgl-test")'
echo "Python exit=$?"     # kỳ vọng 4
kgl exec --shell 'exit 7'
echo "Shell exit=$?"      # kỳ vọng 7
python -c 'import os,sys; open(sys.argv[1],"wb").write(os.urandom(700000))' "$RUN/input.bin"
REMOTE_FILE="$(basename "$RUN")/input.bin"
kgl upload "$RUN/input.bin" "$REMOTE_FILE"
kgl download "$REMOTE_FILE" "$RUN/back.bin"
sha256sum "$RUN/input.bin" "$RUN/back.bin"
cmp "$RUN/input.bin" "$RUN/back.bin" && echo 'File PASS'
```

**Đạt** khi biến tồn tại qua hai lần exec (kết quả 42), thấy 0, 1, 2 hiện dần, hai mã lỗi đúng và file qua nhiều chunk vẫn khớp từng byte. Để kiểm tra riêng lỗi ghi đè biến đã phát hiện, trên notebook thử nghiệm chạy `kgl exec 'info = "kgl-sentinel"'`, rồi `kgl status` và `kgl exec 'print(info == "kgl-sentinel")'`. Ghi FAIL cho tính toàn vẹn namespace nếu kết quả là `False`.

## 4. Tách kiểm tra đường truyền khỏi kiểm tra GPU

```bash
kgl doctor --backend echo -n 3 --report "$RUN/echo.json"
kgl exec --shell 'nvidia-smi'
kgl exec --timeout 90 <<'PY' | tee "$RUN/cuda.txt"
import sys, torch, transformers
print(sys.version, "torch", torch.__version__, "transformers", transformers.__version__)
print("CUDA build", torch.version.cuda, "available", torch.cuda.is_available())
assert torch.cuda.is_available(), "PyTorch khong thay CUDA"
print(torch.cuda.get_device_name(0))
PY
```

Trong `echo.json`, kiểm tra `kernel_exec`, `infer_x3`, `reconnect` và kết quả dừng server. Chạy bước này trước khi mở server model. Mặc định, `doctor` dừng worker do chính nó khởi động sau phép thử. Nếu chỉ `gpu_visible` lỗi, đường truyền vẫn có thể đạt nhưng phép thử GPU còn thiếu. Nếu thiếu torch/transformers hoặc CUDA là False, kiểm tra image, accelerator và phiên bản thư viện trong **notebook** trước khi tiếp tục.

## 5. Chứng minh đúng model và tiến trình GPU

```bash
MODEL=Qwen/Qwen2.5-0.5B-Instruct
kgl serve start --backend transformers --require-gpu --model "$MODEL" --dtype float16 --wait-timeout 900
kgl serve status | tee "$RUN/model-health.json"
kgl infer 'What is 7 plus 5? Answer briefly.' --temperature 0 --max-tokens 32 --timing
kgl infer 'Name one planet.' --temperature 0 --max-tokens 32 --no-stream --json | tee "$RUN/model-reply.json"
kgl exec --shell <<'SH' | tee "$RUN/gpu-process.txt"
set -e
p=$(python -c 'import json; print(json.load(open("/kaggle/working/.kgl/serve.json"))["pid"])')
ps -p "$p" -o pid,args
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_gpu_memory --format=csv
SH
```

Với model gắn sẵn, đặt `MODEL` thành đường dẫn thư mục model trên VM. Trong `model-health.json` kiểm tra `state=ready`, `health.backend=transformers` và `health.model` đúng giá trị đã chọn. Kiểm tra cả hai phản hồi đúng cấu trúc và có nội dung, rồi đối chiếu PID của `serve.pyz` với PID trong bảng NVIDIA và VRAM đã cấp cho tiến trình. Đây là bằng chứng vận hành gián tiếp về GPU. Bản mới còn trả `health.device`, `health.dtype` và `health.instance`. Ghi giá trị thực, ví dụ `cuda:0` và `torch.float16`. Nếu health trả echo hoặc model khác, đánh dấu FAIL, dừng server thử nghiệm và khởi động lại đúng model. Nếu khởi động lỗi trước khi CLI lưu server, xem log bằng `kgl exec --shell 'tail -n 50 /kaggle/working/.kgl/serve.log'`.

### Thử chia trọng số trên hai GPU

Trên notebook thử riêng có hai GPU trống và Accelerate, tải checkpoint vào vùng tạm của VM đủ dung lượng, rồi đặt `MODEL` thành đường dẫn thư mục checkpoint trên VM. Trọng số Qwen2.5-14B-Instruct cần khoảng 29,54 GB. Ghi riêng thời gian tải checkpoint và thời gian từ lúc start đến trạng thái ready.

```bash
kgl serve start --require-gpu --device-map balanced --dtype float16 --model "$MODEL" --wait-timeout 900
kgl serve status
kgl exec --shell 'nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv'
```

Nếu `balanced` báo offload với đúng checkpoint Qwen2.5-14B-Instruct, có thể chia 48 lớp decoder thành hai nhóm 24 lớp bằng bản đồ sau. Chạy khi hai GPU còn đủ bộ nhớ cho trọng số và phần tạo sinh.

```bash
MAP=$(python - <<'PY'
import json

placement = {"model.embed_tokens": 0, "model.rotary_emb": 0, "model.norm": 1, "lm_head": 1}
placement.update({f"model.layers.{i}": int(i >= 24) for i in range(48)})
print(json.dumps(placement))
PY
)
kgl serve start --require-gpu --device-map "$MAP" --dtype float16 --model "$MODEL" \
  --prefill-chunk-size 256 --wait-timeout 900
```

Tùy chọn prefill yêu cầu bản Transformers hỗ trợ `GenerationConfig.prefill_chunk_size`. [Kết quả thử trên hai T4](reports/2026-11-19-v2.md#qwen25-14b-instruct-trên-hai-tesla-t4)

Khi lập bản đồ cho model khác, kiểm tra `tie_word_embeddings` trong cấu hình. Trên Qwen2.5-0.5B-Instruct, đặt `model.embed_tokens` và `lm_head` trên hai GPU khác nhau khiến worker báo `ready` nhưng inference trả HTTP 500 do tensor nằm trên hai thiết bị. Với hai module cùng ở GPU 0 và 24 lớp chia đều qua hai T4, inference đã thành công. Gửi ít nhất một request sau khi worker báo `ready` để xác nhận bản đồ chạy được.

Đối chiếu `health.devices` với VRAM theo PID trên từng GPU. Sau warm-up, thử JSON và SSE với cùng ngân sách token, tăng độ dài prompt từng bước, rồi thử ngắt stream và kiểm tra request kế tiếp. Ghi VRAM từng GPU lúc ready và khi inference, TTFT, độ dài prompt thực từ usage và lỗi đầu tiên nếu có. Dừng instance thử và kiểm tra VRAM đã được trả lại. [Hướng dẫn chia trọng số của Accelerate](https://huggingface.co/docs/accelerate/v1.4.0/en/concept_guides/big_model_inference)

## 6. Gọi API local từ ứng dụng

Tại A chạy proxy và giữ nó ở foreground. Ví dụ dùng loopback và cấu hình khóa mặc định rỗng trong phiên thử này.

```bash
KGL_PROXY_KEY='' kgl serve proxy --listen 127.0.0.1:8001
```

Tại **terminal B trên cùng máy**, kiểm tra `curl -i --max-time 10 http://127.0.0.1:8001/ready` trước khi chạy cả hai request. Readiness trả 200 khi đúng model và instance đã sẵn sàng; mã 503 hoặc timeout cần chẩn đoán tiếp. Terminal B chỉ cần curl.

```bash
curl --fail-with-body --max-time 120 -sS http://127.0.0.1:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Name one ocean."}],"max_tokens":32,"temperature":0}'
curl --fail-with-body --max-time 120 -NsS http://127.0.0.1:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Count from one to ten."}],"stream":true,"max_tokens":64,"temperature":0}'
```

**Đạt** khi JSON có `choices[0].message.content` và model đúng, còn SSE có các delta nội dung rồi kết thúc bằng `data: [DONE]`. Lưu kết quả hoặc ghi lại hiện tượng nếu stream thiếu/lặp. Quay về A, nhấn Ctrl-C **ở proxy** để lấy lại dấu nhắc.

## 7. Đo độ bền và thử gián đoạn thực

1. **Đo sau warm-up.** Sau request đầu ở bước 5, chạy lệnh bên dưới trong A khi server đúng model vẫn hoạt động. `doctor` xác nhận instance đã lưu, backend, model và thiết bị CUDA trước khi gửi request. Nếu server dùng cổng khác 8000, truyền cùng cổng vào `doctor --port`. Server có sẵn được giữ lại cho phép thử kế. `--keep` cũng giữ server nếu `doctor` tự khởi động một instance. Ghi số thành công trên 20, latency p50/p95, TTFT p50 (thời gian đến đoạn văn bản đầu tiên). Báo cáo tổng hợp chỉ đưa các request thành công vào thống kê latency. Nếu bước inference gặp exception, ghi cả lỗi và số liệu còn thiếu.

   ```bash
   kgl doctor --backend transformers --model "$MODEL" -n 20 --keep --report "$RUN/model-20.json"
   kgl serve status
   ```

2. **Ngắt mạng thật.** Chạy `kgl infer 'List the numbers from 1 to 200, one per line.' --temperature 0 --max-tokens 512 --timeout 60`. Khi nội dung đang hiện, ngắt mạng máy local khoảng 30 giây rồi nối lại. Ghi phần đã nhận, thời gian kết thúc và mã thoát bằng `echo $?` ngay sau lệnh. Nếu câu trả lời đã xong trước khi ngắt, ghi “chưa thử được” và lặp lại. Trường hợp stream vẫn liên tục nhờ TCP phục hồi thì chỉ xác nhận gián đoạn mạng ngắn, chưa chứng minh reconnect sau khi WS chết. Với stream thực sự bị cắt, mong đợi lỗi rõ hoặc phản hồi đầy đủ đúng một lần. Đánh FAIL nếu output lặp phần đầu, bị cụt nhưng báo thành công, hoặc treo vượt timeout và thời gian chờ của transport.
3. **Phục hồi cùng phiên.** Chạy `kgl exec --timeout 30 'print(kgl_test_value + 1)'` và `kgl infer 'Name one fruit.' --timeout 60`. Ghi có phải connect lại thủ công hay không. `doctor` tự đóng một channel giữa hai request, nên cần ghi riêng phép thử ngắt mạng thực.
4. **Độ bền một giờ.** Ghi giờ bắt đầu, sử dụng notebook bình thường ít nhất một giờ, rồi kiểm tra lại health và gửi một request. Ghi thời gian thực, số lỗi, lần phải kết nối hoặc nạp lại model. Nếu không thực hiện được, để mục này là “chưa thử”. Để kiểm tra phiên mới, dừng phiên cũ theo bước 8, khởi động phiên mới và lặp bước 2, 4, 5 với URL mới. Ghi thời gian khôi phục. Bản hiện tại chưa có checkpoint/resume.

## 8. Dừng và kết luận

Tắt proxy bằng Ctrl-C nếu nó còn chạy. Trong A chạy `kgl stop`, sau đó **dừng session từ giao diện Kaggle** và kiểm tra giao diện hiển thị trạng thái đã dừng. Nếu máy local đã mất kết nối, vẫn dừng session qua giao diện. Sau khi xóa session local, `kgl status` dự kiến trả mã 3.

| Hạng mục | Điền kết quả và bằng chứng |
|---|---|
| Môi trường | Ngày giờ, Python local/remote, kgl, torch, transformers, GPU, model |
| Local / kết nối / exec / file | Số test, mã lỗi nếu có, kết quả 42, stream, SHA-256, namespace |
| Echo / model / GPU | Số request echo đạt, backend và model trong health, PID và VRAM, câu trả lời thực |
| API / hiệu năng | HTTP JSON và SSE, số đạt trên 20, latency p50/p95, TTFT p50 |
| Gián đoạn / một giờ / phiên mới | Cắt lúc nào, có lặp hoặc mất chữ không, mã thoát, thời gian phục hồi, thời gian quan sát |
| Kết thúc | Session Kaggle đã dừng, vị trí các file bằng chứng |

Ghi **PASS**, **FAIL** hoặc **chưa xác minh** cho từng mục. Chỉ ghi “đã chạy inference trên Kaggle GPU” khi bước 2 và 5 có đủ bằng chứng. Chỉ ghi “đạt độ tin cậy đã thử” khi có 20/20 request bình thường thành công, API trả đủ nội dung, phép thử gián đoạn và một giờ có kết quả đạt. Kết quả một phiên chỉ mô tả GPU, model và điều kiện mạng của lần đo đó.

## llama.cpp GGUF (experimental)

Với GGUF đã gắn vào Kaggle, dùng [quy trình build và nghiệm thu Qwen 35B](../examples/qwen35b-llama/QWEN35B-KAGGLE.md) để kiểm tra build sẵn có, xử lý lỗi CUDA và chạy lại hai request. Các bước dưới đây mô tả luồng GGUF tổng quát.

1. Lấy một commit llama.cpp từ upstream, giữ lại full SHA dùng cho bài thử, rồi chạy lệnh build trên notebook cần kiểm. Lệnh trả JSON với đường dẫn `manifest`, lệnh khởi động, cờ đã dò qua `--help` và phiên bản. Nếu đã có build tương ứng, lệnh kiểm tra binary và trả lại manifest đó. Phiên local ghi đường dẫn manifest để `serve start --preset` dùng trực tiếp.

   ```bash
   kgl model install llama --revision "$LLAMA_CPP_SHA" --jobs 2 --timeout 1800
   ```

2. Chuẩn bị GGUF trên VM. Preset ưu tiên tìm file có tên và SHA-256 khớp trong `/kaggle/input`. Khi tải qua Hugging Face, kgl ghim revision, kiểm SHA-256 và có thể tiếp tục file dở. Cache mặc định là `/tmp/kgl/models`. Với repo cần xác thực, cấp đường dẫn tuyệt đối tới file token 0600 có sẵn trên VM bằng `--hf-token-file`.

   ```bash
   kgl model fetch --preset examples/qwen35b-llama/qwen3.8-35b-a3b.json --quant Q4_K_M --timeout 1800
   ```

3. Đặt `GGUF` và `MANIFEST` theo đường dẫn VM trả về. Bắt đầu với ngữ cảnh 4096, kiểm tra VRAM theo PID, rồi tăng ngữ cảnh theo số đo. Planner ước tính bộ nhớ KV và hồi quy cho GGUF Qwen lai. Giới hạn ngữ cảnh theo tổng VRAM cần kiểm tra lại trên từng card. Dùng `model plan --log` với log llama.cpp trên VM để xem buffer thực tế.

   ```bash
   kgl model plan --model "$GGUF" --ctx 4096
   kgl serve start --backend llama --model "$GGUF" --preset examples/qwen35b-llama/qwen3.8-35b-a3b.json \
     --llama-manifest "$MANIFEST" --ctx 4096 --require-gpu
   kgl status --json
   kgl infer "What is 2 + 2? Reply with one digit." --json
   kgl doctor --backend llama -n 3
   ```

4. Dùng `kgl serve proxy --static examples/chat-demo` để kiểm JSON/SSE qua HTTP local. Trong `infer --json`, `ttft_first_token_s` tính cả reasoning và `ttft_first_content_s` tính đến câu trả lời. Nếu stream kết thúc vì đạt giới hạn token trước khi có câu trả lời, CLI thoát lỗi và chỉ cách tăng `--max-tokens`. Tool calling và vision chưa được nghiệm thu.

Khi kiểm thử hủy giữa stream, dùng kernel phụ (`KGL_CONTROL_KERNEL_ID`) hoặc phiên tạo bằng `kgl connect --new-kernel`. Sau hủy, kiểm tra llama.cpp báo slot rảnh và request mới trả kết quả. Để thử phiên mới, lưu thư mục build dưới dạng tar để giữ quyền chạy của binary, gắn bộ build và GGUF qua dataset, giải nén build vào `/tmp` và chạy lại bước 3.
