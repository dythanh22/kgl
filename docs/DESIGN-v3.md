# Thiết kế kgl v3 · Chat và API Qwen cho công việc hằng ngày

**Đang triển khai · cập nhật ngày 24/11/2026.** V3 phục vụ người dùng cá nhân có máy local và notebook Kaggle với hai Tesla T4. Mốc phát hành đầu tiên là mở lại hội thoại hôm trước và gửi được yêu cầu từ script Python sau khi đổi VM, với cùng địa chỉ API và cấu hình model.

## Điểm xuất phát

Qwen3.8-35B-A3B Q4_K_M đã trả lời thành công ở cả chế độ streaming và non-streaming với context 4096 trên hai T4. Bản build llama.cpp cho SM75 đã lưu thành dataset riêng tư, version 1. Đã đối chiếu từng file khôi phục và chạy binary trên cùng VM. Ngày 23/11 đã khôi phục build và nạp Qwen thành công trên VM mới. [Quy trình khôi phục và serving](../examples/qwen35b-llama/QWEN35B-KAGGLE.md)

Demo đã hỗ trợ nhiều hội thoại, lưu trong trình duyệt, chọn phạm vi lịch sử gửi đi, hiển thị reasoning và số token. Proxy đã có readiness, danh sách model, Chat Completions, SSE và tùy chọn bearer key. Các mục bên dưới mô tả phần phát triển và nghiệm thu tiếp theo. [Giao diện hiện tại](../examples/chat-demo/demo.html) · [Proxy](../src/kgl/proxy/handler.py) · [Preset Qwen](../examples/qwen35b-llama/qwen3.8-35b-a3b.json)

## Phạm vi

V3 gồm khởi động và phục hồi, ứng dụng chat, API dùng chung cho chat và script, cùng ca nghiệm thu xuyên suốt. Các tác vụ ưu tiên là tóm tắt, soạn hoặc chỉnh sửa nội dung, trích xuất thông tin từ văn bản và tiếp tục hội thoại nhiều lượt.

**Ngoài phạm vi v3:** nhiều người dùng, truy cập từ máy khác ngoài `127.0.0.1`, tích hợp IDE, hỗ trợ coding, autocomplete, FIM và coding agent, RAG và upload tệp, đa phương thức, xếp hàng request.

## Mục tiêu đo được

Dùng ngưỡng M0 dưới đây cho các lượt hiệu chỉnh và nghiệm thu tiếp theo. M0 ghi đường cơ sở của cấu hình hiện tại và chốt tiêu chí trước khi chọn cấu hình ứng viên. [Báo cáo M0](reports/2026-11-25-qwen-m0.md)

| Chỉ số | Điều kiện đo | Ngưỡng |
|---|---|---|
| VM mới → câu trả lời đầu tiên | Bắt đầu lúc gọi connect với notebook và input đã chuẩn bị. Ghi riêng connect, ready.py và chuyển proxy → content đầu | Mục tiêu M4 ≤300 s từ connect tới content đầu và ≤250 s riêng ready.py. Mốc lịch sử ready.py là 204,341 s. |
| Số bước thủ công khi đổi VM | Đếm nhóm thao tác theo checklist M0 và ghi bước phát sinh | ≤6 nhóm thao tác |
| Thời gian tới nội dung đầu tiên | Từ máy local qua SSE, prompt khoảng 2.000 token, cache_n=0. Ghi riêng từng chế độ reasoning đã xác minh. | Trung vị ≤12 s và mỗi ca ≤20 s. Ca thiếu content hoặc kết thúc dở trượt ngưỡng. |
| Tốc độ sinh | Trung vị predicted_per_second của các lượt hoàn tất, khoảng 500 token sinh gồm reasoning khi có | ≥60 token/s |
| Chất lượng | Chấm 16 ca theo rubric cố định và kiểm trạng thái hoàn tất | ≥14/16, tất cả ca bắt buộc đạt, mọi ca trích xuất đạt tiêu chí giữ đúng người và hạn, phản hồi được tính đạt phải hoàn tất. |

## Kiến trúc và vòng đời dữ liệu

Chat chạy cùng origin với proxy trên máy cá nhân. Script và các ứng dụng chat tương thích dùng cùng Base URL `http://127.0.0.1:8001/v1`. Proxy chuyển yêu cầu qua phiên Jupyter tới gateway kgl trên Kaggle; gateway gọi llama.cpp đang giữ trọng số trên hai GPU. Khi VM đổi, kgl kết nối lại và khôi phục worker phía sau địa chỉ local cố định. [Kiến trúc kgl](ARCHITECTURE.md)

Cổng 8001 là cố định. Lịch sử chat gắn với origin của trình duyệt, nên nếu cổng đã bị chiếm, lệnh khởi động phải báo lỗi và dừng thay vì tự chọn cổng khác. Nếu đổi cổng, trình duyệt sẽ coi đó là một origin khác và không thấy lịch sử cũ.

| Nơi lưu | Dữ liệu | Hành vi khi đổi phiên |
|---|---|---|
| Trình duyệt local | Hội thoại, bản nháp và tùy chọn gửi lịch sử | Mở lại bằng cùng origin và hồ sơ trình duyệt. Xuất/nhập JSON để chuyển máy hoặc sao lưu. |
| Máy local | Kết nối kgl trong session store, cấu hình client | Cập nhật phiên qua lời nhắc ẩn. URL Kaggle và bearer key nằm ngoài bản xuất hội thoại và log. |
| Kaggle Input | GGUF và dataset build ghim version | Gắn lại hai input, khôi phục binary, thư viện và manifest vào thư mục chạy trên VM. |
| VM Kaggle | Tiến trình inference, context đang nạp và cache | Khởi động lại worker. Giao diện gửi lịch sử đã chọn cho lượt kế tiếp. |

**Quyền riêng tư.** Nội dung hội thoại được xử lý trên VM Kaggle. Log của proxy và gateway chỉ ghi metadata như thời điểm, mã trạng thái, số token và thời gian xử lý, không ghi nội dung messages. Không gửi dữ liệu mà người dùng không muốn đưa lên hạ tầng Kaggle.

## Ràng buộc Kaggle

| Ràng buộc | Cách xử lý trong v3 |
|---|---|
| Phiên GPU tối đa khoảng 12 giờ | Proxy và GUI hiển thị thời gian còn lại ước tính theo `KGL_SESSION_LIMIT_HOURS`. Cảnh báo khi còn dưới 30 phút. Khi phiên hết giữa lượt sinh, xử lý như mất kết nối ở mục 2. |
| Quota GPU theo tuần, không đọc được qua API | Có lệnh dừng worker và hướng dẫn tắt phiên khi không dùng. Tài liệu nhắc người dùng xem quota trong tài khoản Kaggle. |
| Phiên dừng khi không hoạt động | Ghi hành vi quan sát được vào [TROUBLESHOOTING.md](TROUBLESHOOTING.md) và xử lý như mất phiên. |
| Ảnh Kaggle có thể thay đổi CUDA hoặc driver | Khi khôi phục, kiểm tra binary chạy được và thư viện CUDA nạp được. Nếu hỏng, báo rõ thành phần lỗi và chỉ tới quy trình build lại, đã đo khoảng 2100 giây. Sau khi build lại, lưu thành version dataset mới và cập nhật bản ghim. |

## 1. Khởi động và phục hồi

Cung cấp một lệnh trong môi trường `myenv` để thực hiện quy trình hiện tại. Lệnh kiểm tra phiên, nhận diện worker đang chạy và đối chiếu model, backend, context cùng thiết bị. Nếu worker phù hợp, dùng lại nó. Nếu đã có worker khác, hiển thị cấu hình đang chạy để người vận hành quyết định. Trên VM mới, kiểm tra GGUF và hai T4, khôi phục bản build ghim, xác minh commit rồi khởi động worker với context mặc định 32.768 và tensor split `1,1`. Chỉ báo sẵn sàng sau khi kiểm tra readiness và nhận diện đúng instance.

Khi URL hết hạn, chỉ dẫn kết nối lại. Khi thiếu input hoặc thư viện CUDA, nêu tên thành phần cần xử lý. Lệnh khởi động cho biết địa chỉ chat và API đang phục vụ.

**Các bước thủ công khi đổi VM.** Tài liệu liệt kê đầy đủ và theo thứ tự những việc người dùng phải tự làm: bật notebook với GPU T4 x2, gắn hai input, lấy URL và token rồi kết nối lại kgl qua stdin, chạy lệnh khởi động. Các bước còn lại do lệnh tự thực hiện. Số bước thủ công là một chỉ số nghiệm thu.

**Tiến độ nghiệm thu M1.** Các mục đã đánh dấu được kiểm tra trên VM mới bằng CLI hiện có. [Báo cáo ngày 23/11](reports/2026-11-23-qwen-m1.md)

- [x] Khôi phục build từ dataset ghim version 1 trong 11,35 giây. Binary và manifest xác nhận commit `f4f6f2e`.
- [x] Nạp Q4_K_M từ input. Lệnh `serve start` trả về `ready` sau 153,95 giây, context 4096 và cùng PID trên hai T4.
- [x] Chạy inference qua CLI. Cả hai chế độ streaming và non-streaming trả `Paris` với `finish_reason=stop`. Phép thử số học trả `4`.
- [x] Lặp lại `serve start` với cùng cổng và cấu hình, xác nhận cùng instance và PID.
- [x] Có [lệnh đưa worker về ready](../examples/qwen35b-llama/ready.py). Đã thử hai lần trên worker thật, kiểm tra khôi phục và lỗi ở local, và chạy ca xung đột cấu hình của serve manager.
- [x] Chạy chính lệnh ready trên notebook 373065159 khi chưa có build, rồi chuyển proxy 8001 tới instance mới.
- [x] Chuyển proxy cũ trên 8001 tới instance hiện tại, xác minh `/ready`, gọi Chat Completions và chạy lại lệnh quản lý với cùng PID.
- [x] Mở lại hội thoại mẫu đã lưu trên origin cũ và hỏi tiếp qua GUI sau khi chuyển proxy. Cùng script HTTP trả đúng trước và sau khi đổi VM. Hội thoại mẫu dùng localStorage của hồ sơ Firefox thử riêng. [Chi tiết lượt 3](reports/2026-11-23-qwen-m1.md)
- [x] Lệnh quản lý 8001 từ chối listener khác. Kiểm thử với ứng dụng thật trên cổng loopback riêng xác nhận ứng dụng vẫn nhận kết nối.

## 2. Ứng dụng chat

Mở rộng giao diện hiện có và đọc được hội thoại đã lưu. Ưu tiên bảo toàn nội dung người dùng khi tải lại trang, mất mạng hoặc hủy một lượt sinh.

| Tình huống | Hành vi đề xuất và phép thử |
|---|---|
| Làm việc qua nhiều ngày | Lưu bản nháp theo hội thoại, cho đổi tên và tìm hội thoại. Tải lại trang giữa lúc soạn phải khôi phục đúng bản nháp. Xuất rồi nhập JSON phải giữ đúng nội dung và thứ tự. Bản xuất có `schema_version`; khi nhập, từ chối phiên bản không hỗ trợ và báo rõ lý do. |
| Bộ nhớ trình duyệt gần đầy | localStorage có giới hạn khoảng 5 MB. Khi ghi thất bại, không mất dữ liệu đang có, báo người dùng và gợi ý xuất rồi xóa hội thoại cũ. Thử bằng cách lấp đầy bộ nhớ. |
| Đọc câu trả lời kỹ thuật | Render Markdown an toàn, cho sao chép khối mã và thu gọn reasoning. Thử nội dung có HTML, đoạn mã nhiều dòng và stream bị ngắt giữa khối mã. |
| Dừng hoặc mất kết nối | Giữ phần đã nhận, đánh dấu lượt dở và cho người dùng chọn thử lại. Nút Dừng phản ánh kết quả hủy phía worker. Kiểm tra rằng yêu cầu kế tiếp chạy được và lịch sử không nhân đôi lượt người dùng. |
| Worker đang nạp hoặc bận | Hiển thị trạng thái từ proxy, bảo toàn ô soạn thảo và cho kiểm tra lại. Thử gửi từ GUI khi script đang chiếm worker. |
| Phiên sắp hết | Hiển thị thời gian còn lại và cảnh báo theo mục Ràng buộc Kaggle. |
| Hội thoại dài | Giữ toàn bộ hội thoại trên máy. Cho xem phạm vi lịch sử sẽ gửi và chọn ít lượt hơn hoặc mở chat mới. Áp dụng ngân sách token bên dưới. |
| Proxy bật bearer key | File tĩnh của GUI không cần key, nhưng các route `/ready` và `/v1/*` thì cần. GUI hỏi key một lần và lưu trong `sessionStorage`. Key không nằm trong localStorage, bản xuất hoặc log. Khi nhận 401, GUI hỏi lại key. |

### Tiến độ M3

- [x] M3.1 · Bản nháp theo hội thoại, đổi tên và tìm kiếm.
- [x] M3.2 · Sao lưu và khôi phục JSON, xử lý ID trùng và bộ nhớ đầy.
- [x] M3.3 · Giữ phản hồi dở, Dừng, Thử lại và kiểm tra readiness trước khi gửi tiếp sau khi dừng. Trạng thái hủy phía worker được giữ là chưa xác nhận khi proxy chưa có xác nhận.
- [x] M3.4 · Markdown an toàn, sao chép, ước tính thời gian phiên và nhập lại bearer key.
- [x] M3.5 · Ngân sách theo context thực tế, kiểm tra ở 4096 và đo tới 262.144 trên hai T4. Mặc định trước L8 là 768. L8 dùng 1.024 khi Thinking tắt và 2.048 khi bật trên worker hỗ trợ.

[Báo cáo kiểm thử Firefox và Qwen](reports/2026-11-24-qwen-m3-1-to-4.md)

### Ngân sách token theo context đang chạy

GUI lấy cửa sổ từ `/ready`, ưu tiên `health.n_ctx`. Ngân sách gồm chỉ dẫn hệ thống, các lượt gửi, reasoning và câu trả lời. Worker dùng 32.768 token làm mặc định cho công việc văn bản hằng ngày. Cấu hình 4096 tiếp tục là một mốc kiểm tra.

- **Dành chỗ cho đầu ra.** Với preset Qwen 32.768, GUI bắt đầu với Thinking tắt và dành tối đa 2.048 token để trả lời. Chọn 4.096 cho báo cáo dài hoặc nhập giới hạn riêng. Khi bật Thinking, reasoning và câu trả lời chia sẻ giới hạn này. Worker cũ dùng 768. Caller API bỏ qua giới hạn đầu ra dùng mặc định trong preset. [Số đo cấu hình trước đây](reports/2026-11-25-qwen-l5-l8.md)
- **Kiểm tra trước khi gửi.** GUI hiển thị số message được chọn, ước tính đầu vào, giới hạn đầu ra và context. Khi tổng ước tính vượt cửa sổ, người dùng có thể giảm lịch sử hoặc giới hạn đầu ra. Nếu vẫn muốn gửi, cần xác nhận. Lỗi vượt context từ upstream được hiển thị trong kết quả API hoặc GUI.
- **Đếm token.** GUI dùng ước tính từ số byte UTF-8 và số đơn vị từ, số, dấu câu, rồi cộng dự phòng template và từng message. Sau mỗi lượt, hiển thị usage thực tế của worker.
- **Reasoning trong lịch sử.** Chỉ gửi `role` và `content` của các lượt cũ, không gửi reasoning. Demo hiện đã làm vậy, cần giữ nguyên và có phép thử.
- **Tắt thinking.** Có tùy chọn tắt reasoning cho từng request, dùng cho tác vụ ngắn như trích xuất một thông tin. Đo thời gian tới nội dung đầu tiên ở cả hai chế độ.

Context cao hơn được đo trên hai T4 với cùng GGUF, tensor split và prompt đối chứng. Dùng VRAM đỉnh, độ trễ tới nội dung đầu tiên và khả năng nhắc lại mã đầu tài liệu để chọn cấu hình. [Báo cáo M3.5](reports/2026-11-24-qwen-m3-5.md)

## 3. API dùng chung cho chat và script

| Giao diện | Hợp đồng cần nghiệm thu |
|---|---|
| `GET /ready` | Phân biệt worker sẵn sàng, đang nạp, bận và mất phiên để client chọn chờ hoặc yêu cầu kết nối lại. Trả thêm thời gian phiên còn lại ước tính. |
| `GET /v1/models` | Công bố alias đề xuất `qwen-local` khi Qwen tương ứng đang phục vụ. Proxy ánh xạ alias tới model ID thật của instance và tiếp tục chấp nhận ID hiện tại. |
| `POST /v1/chat/completions` | Nhận danh sách messages, hỗ trợ JSON và SSE. Kiểm tra content, finish reason và usage khi có. Với stream đứt giữa chừng, báo lỗi và để client quyết định gửi lại. |

**Khi worker bận thì từ chối ngay.** Worker có một slot. Khi đang có request chạy, proxy không xếp hàng mà trả ngay `503` với `type: worker_busy` và header `Retry-After`. Client tự quyết định gửi lại ở lượt kế tiếp. Request đang chạy không bị ảnh hưởng. Tài liệu client mô tả rõ hành vi này.

### Triển khai M2 theo từng lượt

Dùng giao diện chat hiện có, curl và một script Python làm client nghiệm thu. Mỗi lượt có một đầu ra và kiểm tra hồi quy riêng trước khi mở lượt tiếp theo. Dùng dữ liệu mẫu tiếng Việt và tiếng Anh, có thể tái hiện trong repo.

| Lượt | Công việc và phạm vi thay đổi | Điều kiện hoàn tất |
|---|---|---|
| M2.1 · Ghi nhận API hiện tại **✓** | Quan sát route, trường JSON và kết quả của giao diện chat đang có. Chạy script đối chứng bằng một prompt tóm tắt hoặc trích xuất văn bản mẫu. Ghi bằng chứng đã loại key và nội dung messages. | Có bảng request và phản hồi thực tế cho `/ready`, `/v1/models` và Chat Completions, có JSON và SSE. [Báo cáo đối chứng JSON/SSE và chat](reports/2026-11-24-qwen-m2-1.md). |
| M2.2 · Alias `qwen-local` **✓** | Chuẩn hóa tên model tại proxy trước khi kiểm tra request. Dùng nhận diện instance của M1 để xác định đúng worker Qwen. | Cả alias và ID hiện tại hoạt động với JSON lẫn SSE. ID sai trả 400. Backend khác không công bố alias Qwen. |
| M2.3 · Phản hồi khi bận **✓** | Hoàn thiện nhánh từ chối POST ở proxy. Test đồng bộ từ chunk đầu tiên của request A. | B nhận `503 worker_busy` và `Retry-After`. A hoàn tất bình thường và C chạy được sau đó. |
| M2.4 · Readiness **✓** | Hoàn thiện trạng thái `/ready` và thời gian phiên còn lại ước tính, dùng lại phép tính CLI. Quy định dữ liệu thiếu và tuổi của health cache. | Kiểm tra ready, loading, busy, mất phiên và thiếu dữ liệu thời gian. Chạy lại test quản lý proxy M1. |
| M2.5 · Ví dụ dùng hằng ngày **✓** | Viết ví dụ curl để kiểm tra API và script Python xử lý một tác vụ văn bản, chẳng hạn tóm tắt ghi chú thành các đầu việc. Ghi cách cấu hình bearer key và xử lý lỗi xác thực, yêu cầu sai, worker bận, mất phiên và SSE đứt giữa chừng. | Chạy được ví dụ từ `myenv` với `qwen-local`, xác minh bearer key khi bật và phản hồi khi key sai. Khi stream bị ngắt, client phân biệt câu trả lời dở và lượt hoàn tất. |
| M2.6 · Nghiệm thu qua hai VM **✓** | Chạy cùng script và cùng một tác vụ chat trước và sau lần đổi VM theo M1. Ghi phiên bản client, cấu hình, alias và instance trước/sau. | Cả hai client tiếp tục nhận câu trả lời tại địa chỉ local đã cấu hình. Báo cáo có kết quả cho JSON, SSE và lỗi stream. |

Kết quả M2.2–M2.4 đã được xác minh trên proxy thử với worker Kaggle thật và bộ hồi quy local. [Báo cáo API và readiness](reports/2026-11-24-qwen-m2-2-to-4.md)

Ví dụ thực hành và kết quả đối chiếu trước và sau khi đổi VM nằm trong [báo cáo M2.5–M2.6](reports/2026-11-24-qwen-m2-5-to-6.md).

Tích hợp VS Code và coding agent được để ngỏ cho một mốc riêng khi có nhu cầu. Nếu phát sinh một ứng dụng nghiệp vụ cụ thể, bổ sung ca tương thích dựa trên request thực tế của ứng dụng đó.

## 4. Đánh giá chất lượng cho công việc hằng ngày

Bộ M0 có 16 ca nghiệp vụ mẫu, tám tiếng Việt và tám tiếng Anh. Các ca bao phủ tóm tắt, soạn hoặc chỉnh sửa, trích xuất, đối chiếu nguồn và dùng lại thông tin từ các lượt trước. Mỗi ca có rubric và một số ca được đánh dấu bắt buộc. Dùng runner và rubric để chạy lại khi đổi model, quant, build, context hoặc cấu hình sinh. [Bộ ca, runner và hướng dẫn chấm](../examples/qwen35b-llama/eval/README.md)

L1–L2 đo giới hạn đầu ra hiện tại trên ba profile. L5 xác minh Thinking để có thể đo chế độ tắt, L6 chọn `max_tokens` và sampling từ các phép thử có đối chứng.

## 5. Tool calling sau mốc phát hành đầu tiên

Các việc như tóm tắt văn bản, soạn nội dung và giải thích lỗi dùng ngay hợp đồng chat. Với client cần model chọn công cụ, thử một công cụ chỉ đọc, chẳng hạn tra cứu công việc theo ID. Client gửi schema, nhận tên tool và đối số, kiểm tra đối số rồi thực thi. Client gửi kết quả kèm đúng tool call ID và nhận câu trả lời cuối từ model.

**Nghiệm thu.** Kiểm tra cả câu hỏi cần gọi tool và câu hỏi có thể trả lời trực tiếp. Đối số sai phải được xử lý trước khi chạy công cụ. Chỉ công bố capability tools trong preset sau khi luồng nhiều lượt đạt trên Qwen thật với client đích. [Hợp đồng giao thức](../src/kgl/protocol.py)

## Thứ tự triển khai

Thứ tự nghiệm thu là M0 → M1 → M2 → M3 → M4. Ca khôi phục bằng CLI của M1 đã được chạy trước để kiểm tra trên VM mới và cung cấp số liệu ban đầu cho M0.

| Mốc | Trạng thái nghiệm thu | Điều kiện còn lại |
|---|---|---|
| M0 · Đo mốc ban đầu | **Đã có đường cơ sở L1–L2 và ngưỡng nghiệm thu.** | Các cấu hình hiện tại còn trượt ca chất lượng. L5–L6 đã xác minh tắt Thinking và chọn giới hạn GUI. M4 đo thời gian từ connect tới content đầu và đếm thao tác thực tế. |
| M1 · Khởi động và phục hồi | **Đạt checklist chức năng.** Đã chạy ready từ VM trống, chuyển proxy, mở lại lịch sử mẫu và gọi API tại cùng origin. | Chi tiết phạm vi bằng chứng ở mục 1. |
| M2 · API cho chat và script | **Đạt M2.1–M2.6.** Cùng chat và script chạy thành công qua hai VM. | Kết quả JSON, SSE, lịch sử và lỗi stream trong báo cáo ở mục 3. |
| M3 · Ứng dụng chat | **Đạt M3.1–M3.5 và hồi quy L8.** | M4 kiểm GUI xuyên suốt qua hai VM. |
| M4 · Ca xuyên suốt và phát hành | **Đã chạy qua hai VM, chưa đạt điều kiện phát hành.** | Khôi phục, bàn giao proxy và chat/API đã qua kiểm. Cả hai VM đạt 12/16 và còn ca bắt buộc trượt. [Báo cáo M4](reports/2026-11-25-qwen-m4.md) |

Số đo đổi VM lịch sử áp dụng đúng phạm vi mỗi báo cáo. Tool calling (mục 5) chờ nghiệm thu riêng sau mốc phát hành đầu tiên.

## Rủi ro

| Rủi ro | Dấu hiệu | Cách xử lý |
|---|---|---|
| Ảnh Kaggle thay đổi làm binary không chạy | Khôi phục báo lỗi thư viện CUDA hoặc driver | Build lại theo quy trình hiện có, lưu thành version dataset mới |
| Client nhận stream không đầy đủ | SSE kết thúc trước sự kiện hoàn tất | Ghi nhận lượt trả lời dở và để người dùng quyết định gửi lại |
| Ngân sách context không đủ cho reasoning cộng lịch sử | `finish_reason: length` khi chưa có nội dung trả lời | Tắt thinking, giảm phạm vi lịch sử, chỉnh `max_tokens` mặc định |
| Hết quota tuần giữa giai đoạn nghiệm thu | Không bật được GPU | Gom các phép thử Kaggle vào ít phiên, dừng worker khi không dùng |

## Điều kiện phát hành

Hoàn thành các mục 1–4 và chạy một ca xuyên suốt trên VM mới. Ca xuyên suốt được viết thành checklist hoặc script có thứ tự cố định để lần sau chạy lại được. Lưu bằng chứng vào `docs/reports/` theo một mẫu cố định, gồm commit build, cấu hình GPU/context, các chỉ số so với ngưỡng, số bước thủ công, kết quả bộ đánh giá chất lượng, kết quả giữ lịch sử, hủy giữa stream, phục hồi sau mất phiên, phản hồi khi worker bận, và kết quả chat và script với cùng cấu hình qua hai VM. Dùng các kiểm thử proxy, gateway và transport hiện có cho hồi quy local, rồi xác nhận những hành vi phụ thuộc Kaggle bằng phiên thật. Mục 5 được nghiệm thu riêng khi có nhu cầu dùng agent. [Quy trình kiểm thử](TESTING.md)
