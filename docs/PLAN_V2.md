# Kế hoạch nâng cấp kgl v2

**Trạng thái: đã qua smoke và các ca phục hồi ngắn trên một phiên Kaggle.** Mục tiêu v2 là dùng GPU Kaggle từ terminal cá nhân trong một phiên dài, biết rõ yêu cầu đã chạy đến đâu và lấy lại quyền sử dụng server sau lỗi. Người dùng cần có thể hủy đúng lượt inference, giữ nguyên dữ liệu nếu truyền file bị ngắt và nhận thông báo chính xác khi server bận.

Đợt đầu dùng ngưỡng mở lại channel 20 giây, thu hồi worker qua kernel phụ do người dùng chỉ định và giữ nguyên vị trí metadata trên VM trong lúc kiểm tra khả năng dùng chung thư mục tạm. Chờ xác nhận hủy có ngân sách riêng tối đa 4 giây. Kiểm thử local gồm ca Jupyter báo `aborted`, server đổi instance trên cùng cổng và client ngắt sau chunk đầu. Các phép đo một giờ, chuyển phiên, truyền 100 MB, lưu metadata ngoài Save Version và ma trận GPU vẫn chờ môi trường thực.

**Điểm xuất phát.** Bản hiện tại đã qua 40 test local và 20/20 request ngắn trên một phiên Kaggle GPU. Sau khi WS bị cắt giữa stream, client báo lỗi và request kế tiếp thành công. Các kết quả đó làm mốc hồi quy. Độ bền một giờ, hủy tác vụ thực sự trên VM và phục hồi qua một phiên Kaggle mới vẫn cần bằng chứng. [Kết quả bản hiện tại](PLAN.md) · [Quy trình chạy lại](TESTING.md)

## 1. Kênh rảnh, mất mạng và kernel chết — ưu tiên phát hành

**Đã thấy trong mã.** Channel được dùng lại theo cờ closed. Ping chưa ràng buộc thời hạn nhận pong; trạng thái kernel ngoài request có thể bị bỏ qua theo parent ID. Timeout trước khi ghi byte đầu trong WS có thể bị bắt như lỗi gửi chưa rõ kết quả. Mức độ thường gặp của kết nối rảnh bị đóng trên Kaggle chưa được đo. [WS](../src/kgl/ws.py) · [Jupyter](../src/kgl/jupyter.py) · [Runtime](../src/kgl/runtime.py)

- Mở lại channel trước khi gửi nếu đã rảnh quá ngưỡng. Thử ngưỡng ban đầu 15–20 giây và đo thêm chi phí handshake. Theo dõi hoạt động transport và tiến triển của request bằng hai đồng hồ riêng; bật TCP keepalive để bổ trợ.
- Kiểm tra deadline trước thao tác ghi. Cho phép retry lỗi chắc chắn xảy ra trước gửi. Nếu đã có thể gửi, giữ lỗi “chưa xác định đã thực thi” cho lệnh có tác dụng phụ. Stream đã phát nội dung phải kết thúc rõ khi bị cắt.
- Khi im lặng kéo dài, kiểm tra REST với deadline còn lại. Xử lý kernel mất, dead hoặc restarting ngay cả khi sự kiện thiếu parent ID; phân biệt với lỗi mạng nhất thời và kernel còn sống đang tính toán. Khi REST thành công nhưng WS thất bại, chỉ ra nếu cấu hình HTTP proxy của máy chỉ đang áp dụng cho REST.
- **Nghiệm thu:** fake Jupyter đóng socket rảnh trước request kế tiếp; request đó vẫn chạy đúng một lần. Khi server ngừng đọc/ghi, client thoát trong tối đa ba chu kỳ ping hoặc deadline ngắn hơn của caller. Kernel không xuất stdout nhưng vẫn đáp pong phải hoàn thành bình thường. Mô phỏng kernel chết giữa exec và xác minh CLI thoát hữu hạn.

## 2. Hủy lượt chạy và điều khiển khi kernel bận — ưu tiên phát hành

**Đã thấy trong mã.** Đóng WS giải phóng phía client; worker trên VM có thể tiếp tục sinh token. Probe đi qua hàng đợi kernel nên readiness có thể hết hạn khi inference còn chạy. Lỗi worker bận có thể xuất hiện sau HTTP 200 và chunk đầu. Từng tạo được kernel phụ trên một phiên Kaggle; khả năng dùng nó cho hủy/stop khi kernel chính bận còn cần thử. [CLI](../src/kgl/cli.py) · [Proxy](../src/kgl/serving.py) · [Worker](../src/kgl/remote_server.py)

| Trường hợp | Hành vi v2 cần triển khai | Phép thử quyết định |
|---|---|---|
| Timeout khi chờ lượt local | Bỏ yêu cầu chưa gửi và báo đang chờ kernel | Request đang chạy tiếp tục bình thường |
| Ctrl-C, timeout hoặc client HTTP biến mất khi infer | Gắn request ID, chuyển tín hiệu hủy đến đúng worker qua đường điều khiển vẫn hoạt động, dùng stopping criteria và xác nhận đã nhả lượt. Báo riêng khi chưa xác nhận hủy | Hủy sau chunk đầu, ghi thời gian từ yêu cầu hủy tới worker dừng và tới lúc request mới bắt đầu; kiểm tra tác vụ khác còn nguyên |
| Exec trên kernel dùng chung | Báo timeout riêng với lỗi mạng. Việc interrupt kernel cần lựa chọn rõ vì có thể ngắt cell của người dùng; giữ tương thích mã thoát hiện có cho đến khi tài liệu hóa thay đổi | Cho exec ngủ 60 giây với deadline 5 giây; kiểm tra cả thời điểm CLI trả về và thời điểm kernel thực sự rảnh |
| Status, readiness hoặc stop khi đang sinh | Đọc execution_state qua REST trước probe sâu. Báo busy kèm tuổi và instance của health gần nhất. Quyết định nhận lượt mới trước khi gửi header SSE; tài liệu hóa HTTP status và retry cho client | Chạy infer dài ở A, gọi status và readiness ở B. Phân biệt busy, loading, down, sai instance và state cũ; thử stop đúng server |
| Đường điều khiển độc lập | Thử kernel phụ với phiên thử riêng: gọi health/cancel/stop, đo chi phí tạo và dọn kernel. Nếu không dùng được, ghi giới hạn và đánh giá phương án terminal API được phép trên môi trường đó | Chứng minh lệnh hủy tới worker khi kernel chính đang bận. Nếu mọi lệnh vẫn xếp sau inference, tính năng hủy remote chưa đạt |

## 3. Bảo vệ proxy và tính đúng của CLI — làm song song với nhóm 1

- **Proxy.** Từ chối bind ngoài loopback khi thiếu khóa. Kiểm tra Host theo chế độ bind, Origin cho browser, yêu cầu Content-Type JSON và so sánh bearer key bằng `hmac.compare_digest`. Thiết kế luồng khóa cho client local qua biến môi trường hoặc file 0600; nếu tự sinh khóa, hướng dẫn lấy khóa và tránh ghi vào log. Cập nhật đồng bộ README, SDK, agent và readiness. Test request sai Host, Origin, media type hoặc key bị từ chối trước khi gọi VM; client local hợp lệ vẫn dùng được. [Proxy](../src/kgl/serving.py)
- **Shell và stdout.** Giữ mã 127, ánh xạ signal sang 128 + signal, đặt stdin của shell một lần thành DEVNULL và dùng bộ giải mã UTF-8 tăng dần. Chỉ parse marker cho snippet nội bộ; sinh nonce riêng mỗi lần gọi và báo lỗi nếu thiếu marker kết quả. Render placeholder bằng một lần thay thế. Test process chết do signal, tiếng Việt bị chia ở biên 4096 byte, stdout giống marker, lệnh đọc stdin và tên file có chuỗi `__DATA__`. [Snippet](../src/kgl/remote.py) · [CLI](../src/kgl/cli.py)
- **State và cấu hình.** Chuyển JSON hỏng, schema lạ hoặc biến môi trường sai thành lỗi CLI có hướng khôi phục. Giới hạn chunk theo kích thước sau base64 và overhead của message; kiểm tra timeout, chu kỳ ping hữu hạn và dương ở các chế độ yêu cầu. Test giá trị âm, quá lớn, NaN và session phiên bản cũ. [Config](../src/kgl/config.py) · [State](../src/kgl/state.py)

## 4. Truyền file và vòng đời server — bảo toàn dữ liệu

- Cho phép retry có chọn lọc với STAT, PROBE, DOWNLOAD_CHUNK và SERVE_HEALTH. Khi thiếu marker, báo lỗi giao thức và ngữ cảnh I/O; chỉ báo kernel mất khi có bằng chứng từ transport hoặc REST.
- Upload vào file tạm riêng trên cùng filesystem, ghi theo offset, xác minh kích thước và SHA-256 rồi thay đích nguyên tử. Download cũng dùng tên tạm riêng. Test ngắt giữa chunk, gửi lặp một chunk và hai lượt truyền cùng đích. Đích cũ phải còn nguyên khi upload lỗi. Resume qua lần chạy mới cần định danh nguồn, kiểm phần đã truyền và quy tắc dọn file dở.
- Ghi nhận instance và quyền sở hữu ngay sau spawn. Khi load lỗi/timeout, chỉ dọn tiến trình do lần gọi tạo; giữ log để chẩn đoán. Thử chuyển PID, lock và log ra khỏi vùng Save Version, kiểm tra thư mục tạm dùng được từ hai kernel, và có đường stop cho server cũ trước khi đổi mặc định.
- **Đo trước khi tối ưu băng thông:** truyền hai file 100 MB (dễ nén và ngẫu nhiên), ghi thời gian, hash, kích thước chunk và thông báo IOPub nếu có. Chỉ thử zlib/chunk thích ứng khi xác định được nút thắt. Với weights lớn, hướng dẫn tải ngay trên VM hoặc gắn Kaggle Dataset. [File I/O](../src/kgl/files.py) · [Doctor](../src/kgl/doctor.py)

## 5. Hợp đồng inference và mở rộng GPU

**Độ đúng API.** Kiểm tra tham số trước khi mở stream. Hỗ trợ `max_completion_tokens`, quy định cách xử lý nếu gửi đồng thời `max_tokens`, chuẩn hóa text parts và trả 400 rõ cho tính năng chưa hỗ trợ như tools, stop, seed hoặc n ngoài phạm vi. Kiểm tra giới hạn context sau tokenize. Tránh nhân đôi special tokens khi dùng chat template. Trả đúng finish_reason khi gặp EOS hoặc chạm giới hạn, tính usage bằng token thực và thống nhất event lỗi SSE giữa server VM và proxy. Test cùng input ở JSON và SSE, gồm model có BOS, prompt quá dài và generate lỗi sau chunk đầu. [Model server](../src/kgl/remote_server.py)

**Khả năng GPU.** Đề xuất `--require-gpu` để fail sớm khi tác vụ bắt buộc CUDA. Sau đó mới thử ma trận phiên bản Transformers, tham số dtype, `device_map="auto"` trên 2×T4, lượng tử 4-bit và một model 7–13B phù hợp VRAM. Với model gated, thử lấy token từ secret trên VM và kiểm tra log đã loại bí mật. Ghi phiên bản thư viện, VRAM từng GPU, thời gian tải, TTFT và lỗi thực tế. Các cấu hình này hiện là hạng mục thử nghiệm, chưa có cam kết hỗ trợ.

## Quyết định phát hành

1. Chạy regression local bằng myenv: fake Jupyter để chọn đúng thời điểm lỗi, ipykernel thật để kiểm tra kernel bận, cùng smoke test wheel ngoài repo. Giữ tương thích session cũ và script sử dụng exit code/JSON hiện có; ghi hướng chuyển đổi cho thay đổi về khóa hoặc API.
2. Trên notebook thử riêng, xác nhận năm điều kiện: request sau thời gian rảnh vẫn chạy, mất liên lạc được phát hiện trong ngân sách, hủy giải phóng GPU cho request kế tiếp, upload hỏng vẫn giữ file đích và request HTTP không hợp lệ bị chặn trước VM. Phép thử quyền riêng tư của việc hủy phải bao gồm một cell độc lập đang chạy.
3. Lặp lại 20 request JSON/SSE, quan sát một giờ sử dụng thực, rồi kiểm tra kết nối sang phiên mới sau khi phiên cũ kết thúc. Lưu cấu hình, thời điểm, mã lỗi, PID/instance, độ trễ phục hồi và kết quả request kế tiếp; ghi PASS/FAIL/chưa thử cho từng ca. Chỉ công bố tính năng mở rộng GPU nào đã vượt qua ma trận thử tương ứng.

## Nghiệm thu Kaggle ngày 19/11/2026

Môi trường local `myenv` dùng Python 3.12.16. VM dùng Python 3.13.16, ipykernel 6.17.1, torch 2.11.0+cu130 và Transformers 5.20.2. Notebook có hai Tesla T4 15.360 MiB. Qwen2.5-0.5B-Instruct chạy trên `cuda:0` với `torch.float16`.

| Ca thử | Kết quả quan sát |
|---|---|
| Kết nối và exec | PASS. REST và WebSocket hoạt động, `print(2)` trả đúng 2 |
| Proxy JSON và SSE | PASS. 20/20 request, gồm 10 JSON và 10 SSE, mỗi request tối đa 8 token. p50 0,430 giây và p95 0,434 giây |
| Hợp đồng inference | PASS. JSON và SSE khớp nội dung và usage, cùng báo `finish_reason=length` với 45 token đầu vào, 4 token đầu ra và 49 token tổng. Yêu cầu vượt giới hạn context trả HTTP 400 trước khi mở SSE |
| Cắt WebSocket giữa inference Qwen | PASS. Kernel phụ xác nhận worker đã nhả lượt. Request kế tiếp thành công |
| Ngắt client HTTP, readiness và cell độc lập | PASS với echo. Readiness trả 503 và trạng thái `busy`, lượt POST đồng thời trả 503. Từ khi ngắt client đến khi request kế tiếp hoàn tất là 2,73 giây. Cell chạy ở kernel độc lập hoàn tất bình thường |
| Channel rảnh | PASS. Sau 22 giây, channel được mở lại và lệnh kế tiếp chạy đúng một lần |
| Timeout của exec | PASS. Lệnh ngủ 10 giây với deadline 5 giây trả mã 3 sau 5,10 giây. REST báo kernel vẫn bận và quan sát thấy idle sau 12,06 giây kể từ lúc gọi. Exec tiếp theo thành công |
| Truyền file | PASS. Chủ động dừng upload sau chunk đầu, tải xuống và đối chiếu file đích cũ còn nguyên. Upload lại rồi tải xuống 1.048.607 byte khớp SHA-256 |
| Request HTTP không hợp lệ | PASS. Origin bị từ chối với 403 và Content-Type sai với 415 |

Các server và kernel phụ do bài thử tạo đã được dừng. Còn chờ thử gián đoạn mạng ở mức hệ điều hành, truyền 100 MB, sử dụng một giờ và chuyển sang phiên Kaggle mới.

### Qwen2.5-14B-Instruct trên hai Tesla T4

Checkpoint `cf98f3b` có 29.540.134.000 byte trọng số. Với Accelerate 1.15.0, `balanced` tự offload một phần sang đĩa và kgl từ chối báo ready. Bản đồ tường minh chia 48 lớp decoder thành 24 lớp mỗi GPU, đặt embedding trên GPU 0 và `lm_head` trên GPU 1. Server đạt ready sau 102,92 giây từ lúc bắt đầu serve bằng checkpoint có sẵn.

| Ca thử FP16 | Kết quả |
|---|---|
| Trọng số theo PID | PASS. Cùng worker dùng 14.196/15.360 MiB trên mỗi card. Trọng số nằm hoàn toàn trên CUDA |
| JSON/SSE tuần tự | PASS. 10/10 lượt, 5 JSON và 5 SSE, mỗi lượt sinh 16 token. p50 2,432 giây, p95 2,499 giây theo nearest rank. TTFT p50 0,613 giây |
| Prompt khoảng 2.000 token, prefill mặc định | FAIL. GPU 0 thiếu VRAM cho lượt cấp phát thêm 666 MiB. Proxy báo lỗi SSE; request ngắn tiếp theo thành công |
| Hủy stream và dừng | PASS. Kernel phụ xác nhận hủy sau 1,70 giây. Lượt kế tiếp thành công và PID worker biến mất trên cả hai GPU sau khi dừng |

Đỉnh VRAM toàn card lấy mẫu mỗi 200 ms ở lượt prefill mặc định là 14.703 MiB trên GPU 0 và 14.361 MiB trên GPU 1.

Thử lại cùng bản đồ với `--prefill-chunk-size 256` trên Transformers 5.20.2. Bốn lượt JSON/SSE ngắn đều đạt. Prompt 2.088 token sinh đủ 8 token và kết thúc SSE, TTFT 4,178 giây, tổng thời gian 5,471 giây. Prompt khoảng 4.100 token vẫn OOM khi GPU 0 cần cấp phát thêm 38 MiB. Request ngắn sau OOM, xác nhận hủy sau 1,372 giây và request tiếp theo đều đạt.

Đỉnh VRAM toàn card trong lượt prefill chia đoạn là 14.897/15.360 MiB ở GPU 0 và 14.893/15.360 MiB ở GPU 1, khoảng 97% mỗi card. Các mức toàn card gồm 105 MiB mỗi GPU từ context CUDA mở khi chẩn đoán trên kernel notebook. Sau khi dừng server, hai card trở lại mức 105 MiB và kernel phụ đã được dọn. Checkpoint được giữ trong vùng tạm của VM để có thể dùng lại trong phiên hiện tại. [Cách chạy lại](TESTING.md#thử-chia-trọng-số-trên-hai-gpu) · [Chunked prefill trong Transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py)
