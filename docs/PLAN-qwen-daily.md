# Kế hoạch nâng cấp Qwen cho chat và API hằng ngày

Ngày rà soát 25/11/2026. Mục tiêu phát hành là hoàn thành một tác vụ thật bằng tiếng Việt hoặc tiếng Anh, từ chat và từ script, rồi tiếp tục công việc sau khi đổi VM tại cùng địa chỉ local. Kế hoạch này bổ sung cho [DESIGN-v3](DESIGN-v3.md). Các mục bên dưới là công việc cần thực hiện trừ nơi ghi rõ đã kiểm tra.

## Căn cứ và các kết quả phải giữ được

| Thành phần | Bằng chứng và ý nghĩa đối với nghiệm thu |
|---|---|
| Khôi phục và đổi VM | Dataset build version 1, commit llama.cpp ghim, kiểm tra manifest, SM75 và chạy binary trước khi đổi tên thư mục tạm. M2.6 ghi nhận ready.py đưa VM mới về ready sau 204,341 giây. Đo riêng tới **nội dung trả lời đầu tiên** cho M0. [M1](reports/2026-11-23-qwen-m1.md) · [Phép đo M2.6](reports/2026-11-24-qwen-m2-5-to-6.md) |
| Proxy và API | Giữ origin 127.0.0.1:8001, alias và ID chính tắc. Giữ kiểm tra instance, quyền sở hữu listener, bearer key, trạng thái busy với Retry-After và khả năng gọi lại sau hủy. Giữ phân biệt JSON, SSE hoàn tất và câu trả lời dở. [M2.2–4](reports/2026-11-24-qwen-m2-2-to-4.md) · [M2.5–6](reports/2026-11-24-qwen-m2-5-to-6.md) |
| GUI | Bảo toàn lịch sử cũ, nháp, tìm kiếm, xuất/nhập, Markdown an toàn, Dừng/Thử lại. Lịch sử gửi lên chỉ gồm role/content của lượt được chọn. Đọc context từ worker và báo khi vượt ngân sách. [M3.1–4](reports/2026-11-24-qwen-m3-1-to-4.md) |
| Context và hiệu năng | M3.5 đã đo 4K–262K. Ở 32.768, prompt 25.459 token cho nội dung đầu sau 17,23 s, tốc độ sinh 60,47 token/s và còn ít nhất 4.375 MiB trên mỗi GPU. Prompt đối chứng 2.005 token đạt 2,61 s. Phép đo prompt dài đi từ trong VM, dùng văn bản tổng hợp và một ca nhớ mã. Giữ 32K làm mốc triển khai và 4K làm ca hồi quy. [M3.5](reports/2026-11-24-qwen-m3-5.md) |
| Test và tiến độ | Chạy lại ba module test_qwen_ready, test_qwen_proxy, test_llama bằng myenv. 17 ca được phát hiện, 16 đạt, một ca OpenAI SDK tùy chọn bị bỏ qua. M1–M3 đã có báo cáo chức năng. M0 còn thiếu chất lượng và M4 còn mở. |

**Kiểm tra trực tiếp trên notebook mới.** Phiên cũ hết thời hạn chờ khi gọi thử. Kết nối bằng URL mới qua lời nhắc ẩn, với session store kiểm tra riêng. Notebook báo Python 3.13.16 và hai Tesla T4, mỗi GPU 15.360 MiB, đang dùng 0 MiB khi kiểm tra. Đường dẫn Kaggle Input của ready.py có GGUF v3, dung lượng 21.713.462.944 byte, kiến trúc qwen35moe. Chưa có manifest ở thư mục build đích trong /tmp. Trong chính GGUF, tokenizer.chat_template dài 7.764 ký tự và có nhánh kiểm tra enable_thinking bằng false. Đây là cơ sở để thử tính năng trên binary ghim. Lượt rà soát chỉ đọc metadata, chưa hash toàn file, nạp worker, sinh câu trả lời hoặc đo hiệu năng mới.

**Nguồn sampling.** Tác giả Empero xác định đây là bản Distill, khuyên dùng temperature 0.6, top_p 0.95, top_k 20 và lưu ý nguy cơ lặp khi sinh dài bằng greedy. Họ đề xuất ngân sách 16.384 cho model nguồn. Đây là dữ kiện để chọn đối chứng, còn giới hạn GUI cần xét context, thời gian chờ và tác vụ của Q4_K_M trên T4. Đã đọc model card hiện hành, chưa xác minh được README tại revision GGUF đã ghim. [Model card Distill](https://huggingface.co/empero-ai/Qwen3.8-35B-A3B-Distill#best-practices) · [Model card GGUF](https://huggingface.co/empero-ai/Qwen3.8-35B-A3B-Distill-GGUF)

## Thứ tự triển khai và điều kiện chuyển bước

### 1. Đóng phần còn thiếu của M0 trước khi đổi mặc định — vấn đề 1 và 13

- Tạo examples/qwen35b-llama/eval/ gồm bộ 16 ca có ID, messages, ngôn ngữ, dữ kiện bắt buộc, lỗi khiến ca trượt và cách chấm. Phân bổ 4 tóm tắt, 3 soạn/chỉnh sửa, 3 trích xuất, 3 đối chiếu nguồn và 3 hội thoại nhiều lượt, có cả hai ngôn ngữ. Dùng mẫu nghiệp vụ đã khử dữ liệu riêng. Đưa vào ngày hạn bị thiếu, hai số liệu mâu thuẫn, sửa hạn ở lượt sau, bài dài và yêu cầu giữ nguyên tên/số. Đánh dấu trước các ca bắt buộc.
- Runner Python dùng cùng API qwen-local ở local, chạy tuần tự vì worker có một slot. So khớp trường và dữ kiện tự động khi có đáp án xác định. Với bài viết, người chấm đối chiếu ý bắt buộc, độ trung thành với nguồn, ngôn ngữ và định dạng. Lưu điểm từng ca và lý do trượt. Phân biệt lỗi transport, lỗi cắt và lỗi chất lượng để một lượt chạy lại không xóa kết quả ban đầu.
- Lấy mốc hiện trạng riêng cho GUI 768, script 512/temperature 0 và API bỏ các trường để nhận preset. Trên nhóm ca đại diện, thay từng yếu tố sampling, thinking và giới hạn đầu ra. So sánh 768, 2.048, 4.096 trên cùng prompt và context trước khi chạy cấu hình được chọn trên cả bộ. Lặp ít nhất ba lần các ca có kết quả dao động. Ghi số lần và khoảng biến thiên.
- Đo từ máy local bằng đồng hồ đơn điệu. Với SSE ghi thời gian tới sự kiện đầu, reasoning đầu, **content không rỗng đầu**, kết thúc, finish_reason, usage và [DONE]. Với JSON ghi thời gian nhận toàn bộ. Ghi cache_n khi có, số byte request, instance, quant/hash, build, context, split, tham số thực gửi, VRAM đỉnh mỗi GPU và nguồn số liệu tốc độ. Trường thiếu ghi rõ. So riêng prompt khoảng 2.000 token không cache và đầu ra khoảng 500 token, tách lần đầu trên VM mới khỏi worker nóng. Ghi thời gian connect → restore → load → ready → content và đếm thao tác thủ công.
- Trước giờ chạy ứng viên, lưu bảng ngưỡng trong DESIGN-v3 và báo cáo M0. Đề xuất ban đầu cho 16 ca là ≥14/16, toàn bộ ca bắt buộc đạt, không bịa người phụ trách hoặc hạn trong ca trích xuất, mọi ca được tính đạt có content hoàn tất. Với hiệu năng, lấy trung vị và độ dao động từ lần đo nền để điền số cụ thể cho TTFT, token/s, thời gian đổi VM và số bước. Nếu chất lượng nền thấp, ghi rõ ca lỗi và thử cấu hình thay thế trước khi xét phát hành. M4 dùng đúng bộ ca và ngưỡng đã ghi.

### 2. Đồng bộ preset, đường khôi phục và nhận diện worker — vấn đề 6–10

- Chọn preset Qwen làm nguồn cho alias, context vận hành 32.768, mapping quant–filename–SHA, đường dẫn input có version, revision model, commit build và dataset build ghim version. Giữ revision model và commit llama.cpp là hai định danh riêng. Ghi verification theo từng cấu hình thực đo, bảo toàn bằng chứng 4.096. Công bố thay đổi mặc định cho người chạy trực tiếp bằng --preset. --ctx tường minh tiếp tục ưu tiên.
- Cập nhật client/model_fetch.py để kiểm kiểu các trường mới, client/llama.py để truyền metadata cần thiết. ready.py đọc một lần và truyền tham số cụ thể cho restore_worker.py chạy trong notebook. Sử dụng lại cơ chế kiểm SHA ở vm/llama_tools.py cho đường --model của Kaggle Input. Đo riêng chi phí đọc 21,7 GB. Xác minh ít nhất lần đầu của input/version trên VM, chỉ dùng lại kết quả khi nhận diện được đúng artifact bất biến. Hash sai phải dừng trước lúc nạp và chỉ ra quant/version. Kiểm ca thiếu input, thiếu GPU, build sai, khôi phục dở rồi chạy lại.
- Truyền alias cùng cấu hình instance qua serve manager, session state và health tới proxy/handler.py. Kiểm tra alias gắn đúng model/quant đã chọn, nhất là khi caller dùng --model đồng thời với --preset. Giữ ID chính tắc, tương thích session/preset cũ và lộ trình dùng lại qwen-local khi nâng cấp. Thử quant thứ hai, model khác trùng basename, worker mới dùng lại cổng, session cũ thiếu trường, worker bận và khởi động lại cùng cấu hình. Alias chỉ được công bố cho worker đúng.
- check_answer.py dùng sys.executable -m kgl như ready.py. Chạy thử từ myenv khi PATH không có kgl. Ghi rõ proxy.py quản lý cổng trên Linux, cần ss, /proc và pidfd. Giữ kiểm tra listener lạ, chủ sở hữu, key, KGL_HOME và PID. Đưa hướng dẫn proxy foreground thủ công vào tài liệu cho môi trường thiếu điều kiện, ghi rõ nơi đã thử.

### 3. Xác minh rồi mở hợp đồng inference — vấn đề 2, 4 và 5

- Thử trực tiếp trên llama.cpp ghim trước khi đổi allowlist. Cùng template, GGUF, prompt ngắn và sampling, gửi cặp enable_thinking true/false. Kiểm tra nội dung nhìn thấy, reasoning_content, usage, TTFT và dấu think lọt vào content, qua cả JSON/SSE. Lặp với tác vụ trích xuất và soạn dài. Phản hồi HTTP 200 hoặc GUI ẩn reasoning chưa đủ để kết luận tắt được. Nếu build bỏ qua tùy chọn, ghi tổ hợp chưa hỗ trợ, kiểm cách render template và tách thử nghiệm đổi build thành lượt riêng.
- Chỉ cho phép chat_template_kwargs chứa enable_thinking kiểu boolean khi capability tương ứng đã xác minh. Xét presence_penalty và repeat_penalty riêng, kiểm miền giá trị của đúng bản server, số hữu hạn, giá trị trung tính và tác động tới ca hay lặp. Chỉ bật trường nào đã qua kiểm tra. Giữ nguyên hành vi khi bỏ trường. Bỏ temperature 0 trong ví dụ hằng ngày sau đối chứng, giữ greedy ở ca kiểm tra có mục đích ghi rõ. Không mặc định áp penalty cho mọi tác vụ.
- Thử response_format với json_object trước. Đánh giá riêng json_schema nếu build và template hỗ trợ. action_items.py hiện in văn bản từng dòng. Giữ chế độ đó, bổ sung lựa chọn xuất JSON máy đọc được với mảng đầu việc, người, việc, hạn có thể null. Kiểm dữ kiện so với ghi chú. JSON hợp lệ cú pháp vẫn có thể sai người hoặc hạn. Với đầu ra dành cho máy, gom đủ stream và kiểm JSON, schema, stop và [DONE] trước khi xuất kết quả thành công. Kiểm ca không có đầu việc, thiếu hạn, length và mất kết nối.
- Sửa protocol.py, proxy/handler.py và vm/backends/llama.py theo cùng hợp đồng. Proxy và gateway đều áp defaults/validation. Thử qua proxy lẫn gateway trực tiếp, gồm request thiếu trường, kiểu sai, khóa lạ, capability tắt, override của caller và xung đột hai trường max token. Xác nhận request thực đến llama.cpp, lỗi 400 rõ ràng, JSON/SSE cùng ngữ nghĩa, backend khác tiếp tục chạy. API.md ghi ví dụ và bảng capability theo tổ hợp đã thử.

### 4. Hiệu chỉnh trải nghiệm GUI — vấn đề 2–4

Trong examples/chat-demo/demo.html, đọc health.capabilities từ /ready để hiện Thinking khi instance hỗ trợ, đặt tùy chọn theo từng lượt. Phân biệt điều khiển sinh reasoning với thu gọn phần hiển thị. 768 hiện là mặc định **có thể sửa** khi biết context. Nếu chưa có context và người dùng để trống, payload có thể bỏ max_tokens. Giữ lựa chọn tường minh của người dùng. Sau M0 mới chọn giá trị cho từng chế độ, xét ngân sách đầu vào tiếng Việt, template, lịch sử và context thực; thử cả 4K và 32K.

Khi length, giữ phần dở và cho người dùng chỉnh giới hạn rồi Thử lại một lần theo ý họ. Tự gửi lại có thể tốn thêm GPU và nhân đôi lượt. Kiểm reasoning-only, có một phần content, stop rỗng, [DONE] thiếu, busy sau Dừng, chuyển worker khác capability, nhập lịch sử cũ và payload thực. Ca đạt khi lịch sử không nhân đôi, reasoning không gửi lại, phần dở không được đánh dấu hoàn tất, GUI và script cùng báo đúng nguyên nhân. Kiểm Firefox ở cùng origin và chạy hồi quy giao diện ảnh dùng chung demo.

### 5. Chỉnh tài liệu và thử tối ưu có điều kiện — vấn đề 7, 10–12

- README đã có link Qwen và quy ước ngôn ngữ. Sửa Requirements theo backend, thêm lộ trình connect → ready.py → proxy.py → chat/API ngay trong Quick start. Giữ quick start và CLI tiếng Anh, vận hành và báo cáo tiếng Việt, thêm liên kết qua lại từ chat-demo/README.md.
- QWEN35B-KAGGLE.md giữ thao tác hằng ngày, đổi VM, xử lý lỗi và dừng phiên. BUILD.md trong cùng thư mục chứa thao tác build/debug/khôi phục thủ công và phát hành version dataset mới. Dẫn lại script/preset cho giá trị đang dùng. Chuyển kết quả lịch sử sang docs/reports/, giữ số liệu, điều kiện đo và liên kết cũ có chỉ dẫn. API.md ghi rõ giới hạn thời gian phía client, token, byte, chế độ văn bản/JSON và ví dụ xử lý lỗi. Chạy lại lệnh từ thư mục nêu trong từng hướng dẫn.
- Chỉ mở nhánh thử Flash Attention và KV cache q8_0 khi M0 cho thấy thiếu VRAM hoặc tốc độ. Kiểm flag/syntax và hỗ trợ SM75 bằng chính binary ghim, kiểm cấu hình thực sự có hiệu lực trong log. Đo mặc định → thay từng biến → phối hợp nếu cần trên cùng quant, split và bộ M0, dùng cách lấy mẫu M3.5. Ghi prompt dài, TTFT, tốc độ, đỉnh từng GPU, lỗi/OOM và điểm chất lượng. Dừng sớm khi không có lợi ích đo được hoặc chất lượng giảm. Giữ cấu hình nền để phục hồi và chỉ mở rộng dải context khi có nhu cầu.

### 6. Nghiệm thu M4 và quyết định phát hành — vấn đề 13

1. Trên VM A, lưu cấu hình đã chọn, tạo hội thoại có dữ kiện ở lượt đầu, bản nháp và bản xuất. Gọi cùng tác vụ từ script JSON và GUI SSE trên origin cố định. Chạy bộ M0 và các phép đo đã khóa ngưỡng.
2. Trên VM B mới, ghi các thao tác người dùng, xác nhận đúng hai input/version, kết nối qua stdin, chạy ready.py, proxy.py và đo tới content đầu. Chạy lại ready/proxy để kiểm dùng lại đúng instance và PID trong cùng VM. Dùng hồ sơ trình duyệt cũ, mở lại lịch sử, hỏi tiếp đúng dữ kiện và chạy lại script với cùng alias.
3. Kiểm key sai/đúng, worker bận trả 503 + Retry-After trong khi lượt đầu hoàn tất, hủy giữa stream rồi gửi lượt mới, mất phiên/stream dở và phục hồi, wrong_instance, context sát giới hạn, payload quá giới hạn byte (mặc định 512 KiB) và timeout của client. So sánh đường local xuyên Jupyter với số đo trong VM, nhất là tài liệu tiếng Việt dài. Giữ cả kết quả thử thất bại.
4. Báo cáo trong docs/reports/ ghi commit kgl/build, revision+hash GGUF, dataset version, GPU/context, chế độ thinking/sampling, bảng ngưỡng–thực đo–đạt/trượt, ID ca chất lượng, số bước, thời gian từng chặng, kết quả GUI/API qua hai VM, ca bỏ qua và lý do. Chỉ lưu metadata và kết quả chấm trên mẫu đã duyệt, loại thông tin phiên, key và nội dung riêng. Cập nhật DESIGN-v3 khi đủ bằng chứng.
5. Nếu thiếu phép đo bắt buộc hoặc có hồi quy về dữ kiện, lịch sử, nhận diện instance hay kết thúc stream, giữ M4 mở. Khôi phục preset và build/version đã ghim trước đó, chạy lại ready, alias, JSON/SSE và mở hội thoại cũ. Đưa phương án chưa đạt về thử nghiệm với ca lỗi cụ thể. Tool calling giữ mốc nghiệm thu riêng sau v3.

**Đầu ra theo thứ tự bàn giao:** bộ ca và báo cáo nền M0 → preset/khôi phục nhất quán → bảng capability có bằng chứng trên worker → GUI và script dùng chung hợp đồng → hướng dẫn đã chạy thử → báo cáo M4 qua hai VM. Sau khi đổi model, quant, build hoặc context vận hành, chạy lại các phép thử capability và chất lượng tương ứng trước khi công bố cấu hình mới.

## Lịch triển khai theo lượt

Mỗi lượt bắt đầu từ cấu hình đã qua kiểm tra ở lượt trước. Kết thúc lượt, lưu phạm vi diff, lệnh và kết quả kiểm thử, cấu hình thực đo, ca trượt và quyết định tiếp tục. Mỗi lỗi đã tái hiện cần một ca hồi quy ở tầng phát sinh lỗi. Lỗi phụ thuộc Kaggle có thể thêm ca local mô phỏng hợp đồng và giữ phép thử GPU trong checklist phát hành. Mỗi lượt nên có thể rà soát và quay lui riêng. Nếu một capability chưa đạt, giữ trạng thái chưa công bố và ghi phép thử còn thiếu. Các lượt độc lập có thể tiếp tục, còn M4 chỉ đóng khi đạt điều kiện phát hành.

| Lượt | Phụ thuộc | Phạm vi và sản phẩm bàn giao | Điều kiện đóng và ca hồi quy |
|---|---|---|---|
| L0 · Môi trường | Phiên Kaggle đang mở | Khởi động bằng ready.py và proxy.py. Ghi instance, context, GPU, origin, alias và một lượt chat SSE. | /ready đúng instance và hai T4, /v1/models có qwen-local, content có dữ kiện dự kiến, stop và [DONE]. |
| L1 · Bộ ca M0 | L0 để chạy đối chứng | Tạo 16 ca và rubric cố định, runner tuần tự, định dạng báo cáo và ID ca. | Chạy thử bộ chấm với đầu ra đúng/sai, thiếu dữ kiện và stream dở. Kiểm cả lỗi mạng và lỗi chất lượng. Giữ dữ liệu mẫu có thể chia sẻ. |
| L2 · Đo nền và ngưỡng | L1 | Đo ba đường GUI, API và script. Đo lượt nóng/lạnh, prompt 2.000 token, đầu ra khoảng 500 token, cửa sổ 4K/32K và số bước khi đổi VM dựa trên phép đo có thể tái lập. | Chốt bảng mục tiêu chất lượng và số cụ thể cho TTFT, tốc độ, đổi VM và số thao tác. Có ID ca và cấu hình để chạy lại. Đưa số đo chỉ có từ trong VM vào bảng riêng với điều kiện đo. |
| L3 · Preset và khôi phục | L2 | Đưa đường dẫn input có version, mapping quant, checksum, build/dataset và context vào preset. Dùng chung cơ chế hash và sửa check_answer.py. | Hai lượt ready liên tiếp dùng lại đúng worker. Test hash sai, thiếu input/GPU, binary lỗi, khôi phục dở, cấu hình xung đột, override context và kgl vắng trong PATH. Ghi riêng thời gian hash. |
| L4 · Alias và instance | L3 | Truyền alias gắn với artifact qua cấu hình launch, session, health và proxy. Ghi giới hạn Linux trong hướng dẫn proxy. | JSON/SSE nhận alias và ID chính tắc. Test session cũ, quant thứ hai, basename trùng, worker khác, cổng nội bộ dùng lại và listener ngoài quyền quản lý. Kiểm bảo toàn proxy cũ khi đích chưa sẵn sàng. |
| L5 · Thinking | L4 | Chạy đối chứng trực tiếp trên binary ghim và qua proxy. Nếu đạt, thêm capability boolean và tài liệu API. | Kiểm true/false, mặc định, trường lạ, kiểu sai, model không hỗ trợ, tách reasoning khỏi content và ghi tác động chất lượng/độ trễ của cả hai chế độ. |
| L6 · Sampling và giới hạn token | L2, L5 để so theo chế độ | Đo sampling preset so với greedy, đánh giá riêng presence_penalty và repeat_penalty, chọn ngân sách đầu ra. Cập nhật ví dụ API/script theo kết quả. | Chất lượng đạt ngưỡng M0, kiểm override, giá trị sai, trường không được hỗ trợ và ca hay lặp/cắt. Ghi phép đo cho các giá trị được chọn. |
| L7 · Đầu ra JSON | L4, phối hợp L5–L6 | Thử json_object và json_schema riêng; thêm chế độ đầu ra có cấu trúc vào action_items.py khi đạt. | JSON và SSE cho kết quả hợp lệ khi hoàn tất. Thiếu hạn ánh xạ null, không có đầu việc cho danh sách rỗng. Schema sai, length, đứt SSE và sai dữ kiện phải được phát hiện khi chấm nghiệm thu. |
| L8 · GUI và lịch sử | L5–L6 | Thêm Thinking theo capability và điều chỉnh ngân sách theo M0. Giữ phần dở và thao tác Thử lại minh bạch. | Firefox kiểm cả 4K/32K, reasoning-only, phần content bị cắt, stream thiếu [DONE], hủy/bận, lịch sử cũ, đổi capability và chế độ ảnh. Kiểm payload và số lượt để phát hiện gửi reasoning hoặc nhân đôi lịch sử. |
| L9 · Tài liệu vận hành | L3–L8 | Tách quick start, BUILD.md, API.md và báo cáo. Cập nhật README theo backend và liên kết từ chat-demo. | Chạy lệnh từ đúng thư mục và env có ghi trong tài liệu. Kiểm link và đường khôi phục hoặc dừng phiên. Tài liệu nêu đúng các capability đã thử. |
| L10 · Tối ưu GPU nếu cần | L2, L6 | Chỉ khi có mục tiêu đo được, thử Flash Attention và q8_0 từng biến. | Xác nhận hỗ trợ SM75 và cấu hình có hiệu lực. Ghi lợi ích, chi phí và hồi quy chất lượng. Nếu bỏ qua, ghi số đo làm căn cứ trong báo cáo. |
| L11 · M4 và phát hành | L1–L9 và quyết định L10 | Chạy checklist xuyên suốt qua hai VM, kèm báo cáo, cấu hình ghim và đường quay lui. | Tất cả ca bắt buộc đạt, ngưỡng M0 đạt, GUI và script tiếp tục tại origin cũ. Nếu trượt, mở lại đúng lượt liên quan, thêm ca tái hiện và chạy lại phần nghiệm thu bị ảnh hưởng. |

### Tiến độ L11

Đã nghiệm thu qua hai VM, gồm khởi động lạnh, bàn giao port 8001, mở lại hội thoại và bản nháp, chạy lại API, Dừng giữa stream, worker bận, timeout và phục hồi. VM B đạt 172,26 giây cho ready.py và 261,58 giây từ connect tới nội dung đầu. Giữ M4 mở vì hai VM đều đạt 12/16 và còn ca bắt buộc trượt. [Báo cáo M4](reports/2026-11-25-qwen-m4.md)

### Kết quả L9–L10

**Hoàn tất hướng dẫn vận hành và quyết định GPU.** Đã tách quick start, build/khôi phục, hợp đồng API và số liệu lịch sử. Đã chạy lại các lệnh thao tác trên phiên hiện tại và ca hồi quy local cho khôi phục/dừng. Ba memo qua proxy đạt 2,48 giây tới content đầu và 69,09 token/s ở trung vị. VRAM trống thấp nhất đo tức thời là 3.925 MiB nên giữ cấu hình GPU hiện tại. Ca JSON rút gọn “trước thứ Sáu” thành “thứ Sáu” cần giữ trong nghiệm thu chất lượng. [Báo cáo L9–L10](reports/2026-11-25-qwen-l9-l10.md)

### Kết quả L5–L8

**Đã triển khai và kiểm trên worker Kaggle.** API và GUI hỗ trợ Thinking theo capability. GUI mặc định tắt Thinking và dùng 1.024 token, bật gợi ý 2.048. Script có JSON hợp lệ sau khi phản hồi hoàn tất và kiểm schema. Bộ M0 qua cấu hình GUI mới hoàn tất cả 16 lượt, đạt 13/16 theo rubric. Ba prompt khoảng 2.000 token không cache có content đầu trong 2,16–2,24 giây qua local proxy. M4 vẫn cần ngưỡng chất lượng cùng nghiệm thu xuyên suốt. [Báo cáo L5–L8](reports/2026-11-25-qwen-l5-l8.md)

### Kết quả L3–L4

**Đã triển khai.** Preset giữ thông tin artifact, build, context và alias. Trình kiểm SHA trên Kaggle đo 56,72 giây lần đầu và 0,01 giây khi dùng lại kết quả trên mount chỉ đọc. Bàn giao giữa hai session store có ca thành công, thử lại và bảo vệ proxy cũ khi còn phục vụ. Cần chạy luồng khởi động và bàn giao trọn vẹn bằng mã mới trên hai VM ở M4. [Báo cáo L3–L4](reports/2026-11-25-qwen-l3-l4.md)

### Kết quả L1–L2

**Hoàn tất đường cơ sở.** Đã tạo 16 ca cân bằng Việt–Anh và runner, chạy 48 yêu cầu chất lượng qua proxy live. Ba profile đều đạt 13/16 và cùng trượt E1, C1. S4 cho thấy hai lỗi cần theo dõi khi đổi giới hạn đầu ra gồm phản hồi bị cắt và diễn giải sai trạng thái phê duyệt. L5–L8 dùng lại các ca này khi thay cấu hình. Chất lượng và thời gian đổi VM xuyên suốt vẫn là điều kiện nghiệm thu phát hành. [Báo cáo và ngưỡng M0](reports/2026-11-25-qwen-m0.md)

### Kết quả L0

**Đạt.** Qwen Q4_K_M đã chạy với context thực 32.768 trên hai T4. Proxy local trả HTTP 200 ở /ready với đúng instance và hai thiết bị cuda, /v1/models công bố qwen-local. Firefox headless gửi câu hỏi tiếng Việt qua giao diện thật, hiển thị đúng người phụ trách và hạn, nhận finish_reason=stop và [DONE]. Worker và proxy tiếp tục phục vụ tại http://127.0.0.1:8001/. Đây là phép thử khởi động. L2 sẽ đo chất lượng và hiệu năng theo bộ M0.

### Phát hiện khi chuyển proxy ở L0

Phiên Kaggle cũ đã mất kết nối nhưng proxy cũ vẫn giữ cổng 8001. Proxy thuộc KGL_HOME cũ nên proxy.py của phiên mới từ chối chuyển tự động theo quy tắc kiểm tra chủ sở hữu. Lần này đã xác minh worker đích sẵn sàng, đối chiếu listener với đúng proxy cũ, kiểm trạng thái connection_lost, dùng pidfd để dừng đúng tiến trình rồi khởi động proxy mới trên cùng origin. L4 bổ sung ca hai session store và hướng dẫn bàn giao cổng có kiểm tra. Ca đạt khi proxy lạ hoặc đang phục vụ vẫn được bảo vệ, worker đích phải sẵn sàng trước khi chuyển và người dùng giữ được origin 8001.

### Phạm vi mỗi lượt sửa lỗi

Trước khi sửa, ghi ca tái hiện, đường gọi và kết quả mong đợi. Sau khi sửa, chạy ca đó qua điểm vào bình thường, thêm ca biên gần nhất và chạy nhóm kiểm thử liên quan. Với lỗi xuất hiện ở cả proxy và gateway, kiểm cùng payload qua hai đường. Với lỗi khôi phục, kiểm thêm lần chạy lại và trường hợp lần trước dở dang. Khi lỗi có thể ảnh hưởng dữ liệu chat, dùng lịch sử được tạo bởi phiên bản cũ để kiểm khả năng mở, tiếp tục và sao lưu. Sau khi đổi model, build, quant hoặc context, dùng đúng tiêu chí trong M0 trên cấu hình mới rồi chạy lại phần M4 chịu ảnh hưởng.
