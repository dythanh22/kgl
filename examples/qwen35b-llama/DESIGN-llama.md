# Thiết kế backend llama.cpp cho kgl (GGUF trên 2×T4)

> Thiết kế gốc, lưu để đối chiếu. Quy trình hiện tại nằm trong [hướng dẫn Qwen](QWEN35B-KAGGLE.md).

Ngày 20/11/2026. Gateway, CLI và preset đã có bản thử nghiệm. Xem [kết quả kiểm thử](#12-kết-quả-kiểm-thử) và [cách chạy](../../docs/TESTING.md#llamacpp-gguf-experimental). Các nhãn trong thiết kế ban đầu giữ nguyên ý nghĩa: **[đã đọc]** là từ trang model, **[chắc]** là từ mã nguồn tại thời điểm thiết kế và **[cần thử]** là cần bằng chứng trên Kaggle.

## 1. Model mục tiêu và điều đã biết

Nguồn: `empero-ai/Qwen3.8-35B-A3B-Distill-GGUF` **[đã đọc]**.

| Thuộc tính | Giá trị |
|---|---|
| Kiến trúc GGUF | `qwen35moe`, 36B tham số, khoảng 3B hoạt động mỗi token |
| Cấu trúc | Lai: 30 lớp Gated DeltaNet + 10 lớp full attention; 256 expert, chọn 8 mỗi token |
| Yêu cầu runtime | llama.cpp bản mới có hỗ trợ Qwen3.6 / Gated DeltaNet MoE; bản cũ không nạp được kiến trúc |
| Kiểu đầu ra | Model suy luận: mọi câu trả lời mở đầu bằng khối `<think>`; thẻ khuyên `-n 16384` |
| Lấy mẫu khuyến nghị | temperature 0,6; top_p 0,95; top_k 20 |
| Chat template | Nhúng trong GGUF |
| Giấy phép | Apache-2.0 (kế thừa từ base Qwen3.6-35B-A3B) |
| Vision | Có `mmproj` F16 0,899 GB, nhưng thẻ nói phần chưng cất chỉ xử lý văn bản và vision chưa được đánh giá |
| Checksum | Repo có `SHA256SUMS` cho mọi file GGUF |
| Chia shard | Các file liệt kê là file đơn, không thấy shard |

Dung lượng (GB thập phân theo thẻ; GiB là quy đổi của tôi):

| Quant | GB | GiB | Vị trí trên 2×T4 (≈ 30 GB VRAM thô) |
|---|---|---|---|
| IQ4_XS | 19,628 | 18,3 | Vừa, còn nhiều chỗ cho ngữ cảnh |
| **Q4_K_M** | **21,713** | **20,2** | **Mục tiêu chính**, còn khoảng 8–9 GiB cho KV, buffer, ngữ cảnh CUDA |
| Q5_K_M | 25,348 | 23,6 | Sát giới hạn, chỉ thử nghiệm |
| Q6_K / Q8_0 / BF16 | 29,2 / 37,8 / 71,1 | – | Không vừa 2×T4 |

Chưa kiểm chứng và tôi không đánh giá: các tuyên bố về chất lượng hay benchmark của model (nằm ở thẻ chính, tôi chưa đọc). kgl chỉ phục vụ model, không bảo đảm chất lượng.

## 2. Hệ quả thiết kế từ đặc điểm của model

Những điểm này thay đổi một số điều tôi đã nêu ở vòng trước:

1. **Model suy luận làm hỏng giả định "có nội dung ngay".**
   - Mặc định `max_tokens=256` của `kgl infer` sẽ bị cắt trong khối suy nghĩ, trả `finish_reason=length` và nội dung rỗng.
   - `doctor` dùng `max_tokens=32` rồi kiểm `r.text` khác rỗng, nên **sẽ báo FAIL oan** **[chắc]**.
   - Timeout mặc định 600 s không đủ cho 16384 token nếu tốc độ dưới khoảng 27 token/s.
   - TTFT cần tách hai số: đến token đầu tiên (bất kỳ) và đến token `content` đầu tiên.
   - `infer_chat` chỉ đọc `delta.content` **[chắc]**; cần xử lý `reasoning_content`.
2. **Ước tính tốc độ 16 token/s trước đây chỉ đúng cho model dày.** Với MoE 3B hoạt động, trần băng thông cao hơn nhiều, nhưng thực tế còn phụ thuộc định tuyến expert và kernel DeltaNet trên Turing **[cần thử]**. Không đưa con số nào vào tài liệu trước khi đo.
3. **KV cache nhỏ hơn model attention thuần, nhưng không bằng 0.** Chỉ 10 trong 40 lớp giữ KV; 30 lớp DeltaNet giữ trạng thái hồi quy kích thước cố định. Planner phải tính hai loại bộ nhớ riêng, và không được giả định công thức KV của model dày.
4. **Prompt cache giữa các lượt không còn chắc.** Trạng thái hồi quy không cắt bớt tiền tố được như KV. Việc llama.cpp tái dùng tiền tố cho model lai tới mức nào là **[cần thử]**; đừng hứa giảm TTFT cho agent nhiều lượt trước khi đo.
5. **Yêu cầu build mới là rủi ro số một.** Thẻ model cho thấy cách đóng gói mới (`llama serve`, mục tiêu build `llama-app`, script cài `llama.app`) khác `llama-server` tôi giả định trước đó. Tên binary và cờ thay đổi theo phiên bản, nên backend không được hardcode tên mà phải đọc từ manifest build.
6. **Đĩa:** file 21,7 GB lớn hơn mức 20 GB thường được nhắc cho thư mục output của Kaggle **[cần thử: đo `df` thực]**. Không đặt GGUF ở `/kaggle/working`.

## 3. Quyết định kiến trúc

Giữ phương án A đã chọn: **gateway kgl trên VM + `llama-server` là tiến trình con cùng process group**. Lý do thêm từ model này: crash CUDA hoặc lỗi nạp kiến trúc sẽ không kéo sập gateway, nên health và log vẫn trả được lý do thất bại (rất có khả năng xảy ra khi build llama.cpp chưa hỗ trợ `qwen35moe`).

Không chọn: `llama-cpp-python` (wheel CUDA và hỗ trợ kiến trúc mới thường chậm hơn bản gốc), và chạy `llama-server` trần (mất instance identity, cancel, deadline).

## 4. Bố cục mã nguồn (làm trước, không đổi hành vi)

```
src/kgl/
  protocol.py            # chỉ stdlib; header X-Kgl-*, schema health, validate request; dùng chung local và VM
  core/                  # errors, config, state
  transport/             # ws, jupyter, runtime
  remote/snippets/       # *.py thật (nạp bằng importlib.resources), markers, render
  client/                # files, serve_manager, infer, model_fetch
  proxy/                 # handler, security (host/origin/auth), tài nguyên demo từ examples/chat-demo
  cli/                   # app (parser), commands/*
  vm/                    # gateway, state, backends/{base,echo,transformers,image,llama}.py
  presets/*.json         # xem mục 5
```

- `vm/` + `protocol.py` đóng gói thành **zipapp** `serve.pyz` (sha256) thay cho `serve.py`; sửa `inspect_process` đang kiểm `argv[1] == serve.py`.
- CLI và proxy đọc hợp đồng từ `protocol.py`. Smoke test wheel chạy import `kgl.cli` bằng `python -S`.
- `protocol.py` khai báo backend, capability, quyền dùng tham số và quy tắc readiness. Adapter VM khai báo cờ khởi động và các hook chạy model trong `vm/backends/`.
- Pha di chuyển giữ CLI, snippet gửi qua Jupyter và các import tương thích. Xem vị trí từng module trong [kiến trúc](../../docs/ARCHITECTURE.md#module).

## 5. Preset model (cách xử lý "và tương tự")

Một preset khai báo mọi thứ đặc thù của họ model, để thêm model mới không cần sửa mã:

```toml
# presets/qwen3.8-35b-a3b.toml (minh họa)
repo = "empero-ai/Qwen3.8-35B-A3B-Distill-GGUF"
arch = "qwen35moe"
default_quant = "Q4_K_M"
files = { Q4_K_M = "Qwen3.8-35B-A3B-Q4_K_M.gguf", IQ4_XS = "Qwen3.8-35B-A3B-IQ4_XS.gguf" }
checksums = "SHA256SUMS"           # lấy từ repo, kiểm sau khi tải
min_llama_cpp = "<điền sau spike S8>"
reasoning = true
sampling = { temperature = 0.6, top_p = 0.95, top_k = 20 }
max_tokens_default = 4096          # chỉnh theo số đo
request_timeout_s = 1800           # chỉnh theo tốc độ đo được
ctx_default = 32768                # điểm xuất phát, planner có thể hạ
```

- `kgl serve start --preset examples/qwen35b-llama/qwen3.8-35b-a3b.json [--quant IQ4_XS]`.
- Với model chưa có preset: `--hf-repo owner/name --quant Q4_K_M` đọc danh sách file qua API Hugging Face **trên VM**, dùng mặc định thận trọng, không bật capability nào chưa kiểm.
- Preset là dữ liệu, có thể sai; mỗi preset ghi trạng thái xác minh (chưa thử / đã thử trên GPU X, ngày).

## 6. Thành phần cần xây

1. **`kgl model install llama`** (idempotent, chạy trên VM)
   - Ghim commit hoặc tag và ghi manifest: commit, kiến trúc CUDA (Turing = sm_75), cờ build, tên binary, kết quả `--version`.
   - Ưu tiên build từ nguồn ghim, không pipe script cài đặt từ mạng vào shell. Nếu muốn dùng script chính thức, cần kiểm thêm tính toàn vẹn và allowlist mạng **[cần thử]**.
   - Sau khi cài: nạp thử một GGUF nhỏ cùng họ kiến trúc nếu có, rồi mới coi là đạt.
   - Lưu build thành Kaggle dataset để phiên sau gắn vào thay vì build lại.
2. **`kgl model fetch`** (chạy trên VM)
   - Tải có tiếp tục khi đứt, hiển thị tiến độ, kiểm sha256 theo `SHA256SUMS`.
   - Chọn nguồn theo thứ tự: file gắn sẵn trong `/kaggle/input` (mmap, không tốn đĩa) → Hugging Face (cần Internet bật).
   - Không upload GGUF qua kgl (base64 qua WebSocket quá chậm cho 20 GB).
   - HF token: ghi vào file 0600 trên VM, không đưa vào argv hay snippet.
   - Đích: thư mục nằm ngoài `/kaggle/working`, sau khi probe dung lượng thực.
3. **`LlamaBackend`** trong gateway
   - Chạy `llama-server` bind `127.0.0.1`, cổng nội bộ riêng, log vào file có giới hạn kích thước.
   - Giám sát tiến trình con: chết thì `health.status=error` kèm log tail; gateway vẫn sống.
   - Pha trạng thái: `provisioning → downloading → loading → ready`, kèm tiến độ; thay cho một timeout 900 s duy nhất.
   - Chuyển tiếp `/internal/chat/completions` sang upstream, giữ request-id và deadline; `/cancel` đóng kết nối upstream.
   - Tham số khởi động từ preset và planner: offload toàn bộ lớp, chia theo lớp giữa hai GPU, `--parallel 1`, `--jinja`, cờ reasoning; **tên và giá trị cờ kiểm bằng `--help` của build đã ghim**, không hardcode.
4. **Planner (`kgl model plan`)**
   - Đầu vào: kích thước file thực, metadata GGUF (số lớp, KV head, kích thước head, loại lớp DeltaNet), VRAM mỗi GPU từ `nvidia-smi`.
   - Tên khóa metadata theo kiến trúc nên không hardcode; nếu không đọc được, dùng số liệu KV/compute buffer mà llama-server in khi nạp.
   - Đầu ra: ngữ cảnh tối đa khuyến nghị, tỷ lệ chia GPU, cảnh báo OOM trước khi chờ nạp. Thử `--fit`-kiểu tự điều chỉnh nếu build có **[cần thử]**.
5. **Quan sát:** `kgl status` hiển thị VRAM từng GPU đối chiếu theo PID của tiến trình con, `n_ctx`, số lớp offload, tốc độ prompt/sinh từ `timings` của upstream. `infer --timing` in cả hai.

## 7. Thay đổi giao thức và client

- `capabilities` trong `health` (lưu vào `session.serve`): `reasoning`, `tools`, `stop`, `seed`, `top_k`, `usage`. Proxy dùng nó để từ chối hoặc cho qua; Transformers giữ nghiêm, llama cho qua theo allowlist.
- `infer_chat` và proxy giữ nguyên `reasoning_content`, `tool_calls`, `timings`, `usage`; `demo.html` hiển thị phần suy nghĩ tách khỏi câu trả lời, mặc định thu gọn.
- Đầu ra `infer --json` thêm `reasoning`, `finish_reason`, `ttft_first_token_s`, `ttft_first_content_s`.
- Phát hiện cắt cụt: `finish_reason=length` mà `content` rỗng → lỗi rõ ("bị cắt trong lúc suy nghĩ, tăng `--max-tokens`"), mã thoát khác 0.
- Mặc định `max_tokens` và timeout lấy từ preset; timeout cho phép đặt theo `max_tokens / tốc độ đo`.
- `doctor --backend llama`: bước inference chấp nhận khi có `content` **hoặc** `reasoning` không rỗng kèm `finish_reason`; thêm một bước "trả lời đúng một câu hỏi cố định" với `max_tokens` đủ lớn.
- Giới hạn request 512 KiB thành cấu hình; đo trần message WS qua proxy Kaggle trước khi nâng.
- Tool calling: chỉ ghi "hỗ trợ" sau khi có test thật với một agent và model này.
- Vision/mmproj: **ngoài phạm vi** giai đoạn này (thẻ nói chưa đánh giá).

## 8. Spike trên Kaggle (làm trước khi viết backend)

| # | Câu hỏi | Bằng chứng ghi lại |
|---|---|---|
| S1 | Có CUDA toolkit (`nvcc`) không? Build llama.cpp ghim mất bao lâu, binary bao nhiêu MB? | log, thời gian |
| S2 | Internet có bật không? Tốc độ tải 21,7 GB? Dung lượng ghi được ở `/tmp`, `/kaggle/tmp`, `/kaggle/working`? | `df`, thời gian tải |
| S3 | Q4_K_M nạp lên hai GPU: thời gian nạp, VRAM từng GPU, `n_ctx` tối đa thực tế | `nvidia-smi` theo PID, log nạp |
| S4 | Ngắt client giữa stream, upstream có dừng sinh không? | tốc độ GPU về 0, slot rảnh |
| S5 | SSE qua kernel: số message/giây, có chạm giới hạn iopub không? Message 1–4 MiB đi qua WS được không? | thử tải |
| S6 | Prompt cache giữa các lượt trên kiến trúc lai: TTFT lượt 2 so với lượt 1 | số đo |
| S7 | Tốc độ thật: prompt/sinh, ổn định 1 giờ | báo cáo doctor |
| S8 | Build đã ghim có nhận `qwen35moe`? Phiên bản tối thiểu? Kernel DeltaNet chạy trên GPU Turing hay rơi về CPU? | log nạp, tốc độ |
| S9 | Định dạng đầu ra suy luận qua API: `reasoning_content` hay lẫn `<think>` trong `content`? Có tắt được suy nghĩ bằng tham số template không? | JSON/SSE thô |
| S10 | IQ4_XS và Q5_K_M: vừa bao nhiêu ngữ cảnh? Chênh lệch tốc độ? | bảng so sánh |

S8 chặn mọi thứ phía sau; làm nó đầu tiên bằng một GGUF nhỏ cùng kiến trúc nếu có, rồi mới tải 21,7 GB.

## 9. Kế hoạch

| Pha | Nội dung | Nghiệm thu |
|---|---|---|
| 0 | Di chuyển bố cục, `protocol.py`, registry backend, zipapp | Toàn bộ test hiện có qua; CLI import không cần torch; echo qua zipapp chạy ở localhost và Kaggle |
| 1 | Sửa điểm nghẽn chung: fallback interrupt cho INFER khi không có kernel điều khiển, lưu `control_kernel_id` vào session, timeout hash theo kích thước, hàm `gpu_ready` duy nhất, bản ghi `starting`, giữ control runtime khi poll, giới hạn request cấu hình, `Retry-After` cho `/ready` khi bận | Test fake Jupyter: hủy infer không cần kernel điều khiển thì request kế tiếp thành công trong ngân sách; hash file 2 GB không bị timeout giả |
| 2 | Spike S1–S10 (song song với pha 0–1) | Mỗi dòng PASS/FAIL/chưa thử, kèm bằng chứng; quyết định cách build/tải dựa trên số đo |
| 3 | `install`, `model fetch`, `LlamaBackend`, planner, pha trạng thái. Fake llama-server (HTTP nhỏ: `/health` 503→200, SSE có `reasoning_content`, ngắt kết nối, chết giữa chừng) để test không cần GPU | Test contract chung cho echo/Transformers/llama: start hai lần ra cùng instance; đứt giữa stream không lặp prefix; cancel đúng; tiến trình con chết thì health báo lỗi; stop dọn cả hai tiến trình |
| 4 | Capabilities, passthrough có kiểm soát, xử lý reasoning, preset, doctor cho llama, test bằng `openai` SDK | SDK stream và non-stream nhận đủ, `usage` đúng, trường hợp bị cắt trong lúc suy nghĩ báo lỗi rõ |
| 5 | Nghiệm thu Kaggle: 20 request, hủy giữa chừng, nhiều lượt, quan sát 1 giờ, phiên mới tái dùng build và model từ dataset | VRAM từng GPU khớp PID; ghi riêng thời gian build, tải, nạp, tốc độ prompt/sinh; PASS/FAIL/chưa thử từng ca |

## 10. Rủi ro chính

| Rủi ro | Mức | Cách giảm |
|---|---|---|
| Không build được llama.cpp CUDA đủ mới trên Kaggle hoặc build quá lâu | Cao | S1/S8 làm đầu; build một lần rồi đóng gói dataset |
| Kernel Gated DeltaNet chậm hoặc không chạy trên Turing | Cao | S8/S7; nếu chậm, thử IQ4_XS hoặc đổi model họ attention thuần |
| Tải 21,7 GB tốn quota/thời gian mỗi phiên | Trung bình | Dataset riêng tư gắn qua `/kaggle/input` (Apache-2.0 cho phép phân phối lại; kiểm điều khoản Kaggle về dataset) |
| Suy luận dài làm chạm timeout và giữ kernel | Trung bình | Timeout theo preset, interrupt fallback, báo cắt cụt rõ |
| Prompt cache không hiệu quả với kiến trúc lai | Trung bình | Đo S6; không hứa giảm TTFT |
| Lưu lượng SSE lớn qua iopub | Trung bình | Gộp chunk theo thời gian (ví dụ 50–100 ms) trong snippet INFER nếu S5 thấy mất message |
| Điều khoản Kaggle với tải nặng kéo dài | Thấp–trung bình | Giữ phạm vi inference cá nhân tương tác; không né giới hạn phiên |

## 11. Cần bạn chốt

1. Quant mục tiêu: Q4_K_M (khuyến nghị) hay IQ4_XS để có thêm ngữ cảnh.
2. Phiên Kaggle của bạn có Internet không; nếu không, cần chuẩn bị dataset chứa GGUF.
3. Mục đích chính: agent code ngữ cảnh dài hay chat ngắn (quyết định mức đầu tư vào `ctx`, giới hạn request và tool calling).
4. Có chấp nhận dùng chế độ suy nghĩ mặc định (chậm hơn, nhiều token) hay cần tắt/giới hạn suy nghĩ (xác minh được ở S9).

## 12. Kết quả kiểm thử

Mã đã được chia thành các package ở mục 4. Gateway và adapter VM được đóng gói thành `serve.pyz`. Preset dùng JSON để giữ Python 3.10 và thư viện chuẩn. Các import cũ mà demo và test dùng được chuyển tiếp tới package mới.

| Ca thử | Kết quả |
|---|---|
| Mục 4 · Cải tổ bố cục | So sánh 20 màn hình help và nội dung snippet với bản trước khi dời mã. Echo qua zipapp mới trên Kaggle trả đủ JSON/SSE, khởi động lại cùng cấu hình trả cùng instance và dừng thành công. |
| Zipapp, suy luận echo, worker llama giả và hủy giữa SSE | PASS trên Kaggle. Hủy trên kernel thử riêng được xác nhận, request kế tiếp thành công và worker thử đã dừng. |
| SHA-256 file 2 GiB | PASS trên Kaggle với file thưa, đối chiếu với SHA-256 tính độc lập ở local. |
| S1/S2 · Điều kiện VM | Có hai Tesla T4 15.360 MiB, `nvcc` 13.0.88, CMake và Git. GitHub và Hugging Face truy cập được. Tại thời điểm đo, `/tmp` trống khoảng 2,70 TB, `/kaggle/working` trống 20,94 GB, `/kaggle/tmp` không tồn tại. |
| S5 · Kích thước message | PASS cho một execute request chứa khoảng 1 MiB (5,65 s) và 4 MiB (9,37 s) qua WebSocket. Lưu lượng SSE liên tục và giới hạn IOPub còn chờ đo với model đích. |
| Model nhỏ và llama.cpp thật | PASS trên CPU của notebook với binary upstream b12558 (8bf88ba93) và Stories260K GGUF 1.185.376 byte. Fetch xác minh SHA-256, JSON/SSE đều có nội dung và usage, slot báo ngữ cảnh 512. Worker thử đã dừng. |
| Các spike với model đích | Build CUDA, nạp `qwen35moe`, VRAM từng GPU, hủy slot với model thật, prompt cache, tốc độ, định dạng reasoning, các quant và tái sử dụng dataset qua phiên mới còn chờ nghiệm thu. |
