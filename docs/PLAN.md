# Kế hoạch hoàn thiện kết nối và server Kaggle

Lộ trình tiếp theo nằm ở [Kế hoạch nâng cấp v2](PLAN_V2.md), gồm các vấn đề sau nghiệm thu, thứ tự sửa và điều kiện phát hành.

**Quyết định:** giữ kiến trúc CLI → Jupyter REST/WebSocket → kernel → server HTTP trên loopback của VM. Một phiên Kaggle thật đã chạy được toàn bộ đường này. Mốc kế tiếp là bảo đảm client biết khi nào yêu cầu đã hoàn tất, khi nào kết quả chưa xác định và làm thế nào phục vụ lại sau lỗi. Model nhỏ hiện tại đủ dùng làm tải thử. [Kiến trúc](ARCHITECTURE.md) · [Đề bài](BRIEF.md)

## Mốc đã kiểm chứng ngày 19/11/2026

| Khả năng | Bằng chứng | Phạm vi kết luận |
|---|---|---|
| Attach vào kernel và chạy Python | Báo cáo `echo.json` ghi REST thành công, exec trả 2, 3/3 request echo thành công | Một notebook và URL còn hiệu lực. Bước reconnect trong doctor chủ động đóng channel giữa hai request |
| Server thực dùng GPU | `model-health.json` ghi Transformers và Qwen2.5-0.5B-Instruct. `gpu-process.txt` ghi PID 139 chạy đúng script và chiếm 1.146 MiB VRAM. `cuda.txt` ghi Tesla T4 | Đã đối chiếu PID trong lần đo. Bản đầu tiên chưa xuất device/dtype thực tế của model |
| Inference qua proxy | Thử trực tiếp 20 request tuần tự, 10 JSON và 10 SSE đều nhận đủ phản hồi. Trung vị 0,61 s, p95 0,91 s, trung vị tới chunk nội dung đầu tiên 0,40 s | Prompt ngắn, tối đa 32 token, một phiên. Dùng số đo để đối chiếu hồi quy trong cùng điều kiện |
| File và kiểm thử local | Upload/download 700.000 byte khớp từng byte. Suite localhost với Jupyter giả qua 27/27 | Đã kiểm tra đường truyền file bình thường. Ghi đè file khi upload gián đoạn còn cần kiểm thử |

## Kết quả trên bản đã sửa

- 36/36 test qua trong myenv, gồm TLS, fake Jupyter, Jupyter Server + ipykernel thật và các ca hồi quy lỗi giữa request.
- Đã triển khai sáu mục trong bảng bên dưới. Bản mới được thử ở một thư mục và cổng VM riêng trên cùng session Kaggle. Start hai lần trả cùng instance. Remote health ghi nhận Qwen trên `cuda:0`, `torch.float16`; local readiness trả 200.
- 20/20 request HTTP thành công (10 JSON, 10 SSE, mỗi request tối đa 8 token). Trung vị 0,440 s, p95 0,442 s. Phép đo ngắn này không đủ để so tốc độ giữa hai phiên bản dùng prompt và số token khác nhau.
- Đã chủ động ngắt socket phía client sau chunk có nội dung trong SSE. Request đang chạy báo mất kết nối, request mới trả đủ stream. Stop đúng instance thử trả thành công. Ca start lặp từng làm kernel chờ khóa trên VM; bản sửa giải phóng khóa trong cùng snippet và có test trên ipykernel sống lâu.
- Còn mở: sử dụng liên tục một giờ, cấp phiên GPU mới sau khi phiên cũ hết hạn, và lỗi GPU/driver kéo dài trên VM thật. Test worker exception ở local dùng mô hình giả. Thử cắt socket khác với thử gián đoạn mạng ở mức hệ điều hành.

## Nguồn và ranh giới của giải pháp

Kaggle cung cấp URL kết nối Jupyter Server từ giao diện notebook. Người dùng bật GPU, mở phiên và lấy **VS Code Compatible URL** như bước 2 trong hướng dẫn. kgl gắn vào phiên đang chạy; việc tạo kernel trên server khác với cấp phát VM. Kaggle CLI chính thức có luồng batch push/status/output để xử lý job khi nhu cầu cho phép chờ kết quả. Colab CLI dùng API cấp phát riêng của Colab. Remote Bridge sử dụng phiên trình duyệt và Playwright; luồng đã thử của kgl dùng URL lấy trực tiếp từ giao diện. [Kaggle Notebooks](https://www.kaggle.com/docs/notebooks) · [Kaggle CLI](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md) · [Colab client](https://github.com/googlecolab/google-colab-cli/blob/main/src/colab_cli/client.py) · [Remote Bridge](https://github.com/JoseGarciaMayen/remote-bridge)

Phạm vi vận hành là inference cá nhân, một request hoạt động trên mỗi runtime của proxy, trong thời gian phiên cho phép. Uptime/quota thực tế cần theo dõi trên Kaggle. Dừng phiên từ giao diện sau khi dùng. [Kaggle Notebooks](https://www.kaggle.com/docs/notebooks) · [Điều khoản Kaggle](https://www.kaggle.com/terms)

## Thứ tự triển khai

Bảng giữ lại vấn đề gốc và tiêu chí đối chiếu cho bản đã sửa. Mỗi hàng ghép phần sửa với một phép thử hồi quy có thể viết trước khi sửa.

| Thứ tự | Vấn đề của bản ban đầu | Hành vi cần duy trì | Nghiệm thu |
|---|---|---|---|
| 1 · P0 · Quyết định gửi lại | Trường hợp `sendall` lỗi được đánh dấu `sent=False`. Client có thể chạy lại lệnh đã tới kernel. [ws.py](../src/kgl/ws.py) · [jupyter.py](../src/kgl/jupyter.py) · [runtime.py](../src/kgl/runtime.py) | Phân biệt lỗi mở kết nối trước khi gửi với lỗi sau khi bắt đầu ghi. Với lệnh thay đổi dữ liệu và kết quả gửi chưa rõ, trả lỗi “chưa xác định đã thực thi”, đóng channel và để caller quyết định | Trong fixture local, làm hỏng kết nối ngay trước gửi rồi ngay sau khi server nhận lệnh tăng bộ đếm. Trường hợp đầu chạy đúng một lần; trường hợp sau báo chưa xác định, bộ đếm không vượt 1 |
| 2 · P0 · Toàn vẹn stream | Snippet INFER coi EOF là thành công. Proxy phát lại dấu kết thúc khi đã mở SSE. `infer_chat` cho phép retry kể cả sau khi đã phát nội dung. [remote.py](../src/kgl/remote.py) · [serving.py](../src/kgl/serving.py) | Dùng quyết định gửi lại ở mục 1. Theo dõi chunk đã chuyển tới client và dấu kết thúc thực của upstream. Trước khi gửi HTTP headers có thể trả lỗi HTTP; sau đó kết thúc stream lỗi và ghi nguyên nhân. CLI trả mã lỗi khi thiếu dấu hoàn tất | Với backend echo xác định, cắt trước chunk đầu, sau chunk đầu, và ngay trước dấu kết thúc. Đếm số lần chạy ở backend, so chuỗi chunk ở client. Ca bình thường có đúng một `[DONE]`; ca đứt giữa chừng không lặp prefix, không có `[DONE]` giả và request kế tiếp thành công |
| 3 · P0 · Vòng đời server | `serve start` dùng lại PID còn sống, `wait_ready` chỉ nhìn trạng thái. `serve stop` gửi tín hiệu theo PID đã lưu. PID có thể đã được tái sử dụng. CPU fallback vẫn báo ready. [remote.py](../src/kgl/remote.py) · [remote_server.py](../src/kgl/remote_server.py) · [serving.py](../src/kgl/serving.py) | Lưu định danh lần khởi động, PID, thời điểm tạo process, cổng và cấu hình mong đợi. Đối chiếu danh tính trước khi dùng lại/dừng, kể cả với PID file phiên bản cũ. Health trả backend, model, device và dtype thực tế. Ghi session sau khi probe thành công; vẫn truy xuất được log khi load lỗi | Start cùng cấu hình hai lần cho cùng instance. Start khác model/cổng trả xung đột rõ. PID file cũ không làm dừng process khác. Với ca GPU bắt buộc, echo/CPU không thể PASS. Stop lặp lại an toàn và giải phóng cổng |
| 4 · P0 · Timeout và phục hồi | Worker gọi `generate` ở thread riêng; exception có thể để streamer chờ và giữ khóa. `Runtime.call` không đặt deadline; chờ lock có thể vượt timeout của lệnh. Bản tin của request khác có thể reset bộ đếm idle. [remote_server.py](../src/kgl/remote_server.py) · [runtime.py](../src/kgl/runtime.py) · [jupyter.py](../src/kgl/jupyter.py) | Truyền exception và tín hiệu kết thúc từ worker; đặt deadline xuyên suốt chờ lượt, probe, WS và đọc stream. Chỉ tính tiến triển của request hiện tại. Khi worker thực sự kẹt, đánh dấu server lỗi và cho phép restart có kiểm soát theo mục 3 | Giả lập lỗi generate, kernel bận và worker không trả lời. Ghi deadline cấu hình và thời gian thực. Client kết thúc trong ngân sách, lượt chờ thứ hai có kết quả xác định. Sau lỗi có thể phục vụ request mới hoặc restart đúng instance rồi phục vụ lại |
| 5 · P1 · Chẩn đoán đúng tầng | Local `/health` trả OK khi chỉ có proxy sống. `doctor` suy diễn tên model từ tham số; phép thử reconnect xảy ra giữa hai request, phép stop có thể PASS dù kết quả chứa lỗi. [serving.py](../src/kgl/serving.py) · [doctor.py](../src/kgl/doctor.py) | Giữ probe liveness nhẹ. Bổ sung readiness có deadline, kiểm tra REST → kernel execute → remote health → identity/device. Doctor ghi trạng thái quan sát, lỗi từng tầng, thời gian và kết quả stop thực. Khi tài nguyên đang bận, trả trạng thái bận có lý do | Dừng model trong fixture trong khi proxy sống: liveness vẫn 200, readiness báo backend down. Thử token hết hạn, kernel mất, server loading/sai model và stop lỗi; mỗi ca có bước FAIL phù hợp và báo cáo đã che token |
| 6 · P1 · Tương thích kernel sống lâu | Các snippet nội bộ ghi biến vào namespace notebook. Mock hiện chạy subprocess mới mỗi lần execute. WebSocket lưu danh sách fragment bên trong từng lần recv. [remote.py](../src/kgl/remote.py) · [ws.py](../src/kgl/ws.py) · [fake_jupyter.py](../tests/fake_jupyter.py) | Cô lập namespace của snippet quản trị, giữ semantics của exec người dùng. Giữ fragment của cùng message qua timeout. Bổ sung một bài tích hợp Jupyter Server + ipykernel thật trên localhost | Đặt biến sentinel, gọi status/upload/infer và kiểm tra sentinel còn nguyên. Hai lệnh exec liên tiếp đọc được biến của nhau. Test frame phân mảnh có timeout và ping xen giữa trả lại đủ payload |

Có thể làm mục 3 song song với mục 1–2. Hoàn thiện mục 4 sau khi xác định được đúng instance cần khôi phục. Dùng báo cáo từ mục 5 để quyết định vòng nghiệm thu. Nếu quan sát thấy thời gian chờ lượt cao, đo riêng thời gian xếp hàng, mở channel, tới chunk đầu và sinh nội dung trước khi thay transport hoặc tăng concurrency.

## Nghiệm thu bản đã sửa

1. **Local với lỗi có thể tái hiện.** Chạy 27 test nền cùng các regression ở bảng trên. Dùng fake server để chủ động chọn đúng thời điểm lỗi và đếm số lần thực thi. Dùng Jupyter thật để kiểm tra namespace, reconnect và timeout của kernel sống lâu. Không cần GPU cho các lỗi transport.
2. **Kaggle với session mới.** Theo các bước 1–6 trong [hướng dẫn tự test](TESTING.md). Lưu version/commit, loại GPU, backend/model thực, PID, kết quả exec, hash file và JSON/SSE. Tách báo cáo echo và model.
3. **Độ bền.** Chạy ít nhất 20 yêu cầu tuần tự và quan sát một giờ sử dụng thực. Yêu cầu 20/20 phản hồi đúng cấu trúc, đủ kết thúc; ghi cả lỗi và thời gian của từng lần. Trong một ca riêng, ngắt kết nối giữa stream, ghi client đã thấy chunk nào, dấu kết thúc, thời gian báo lỗi và kết quả request kế tiếp. Nếu TCP tự hồi phục, đánh dấu ca WS chết là chưa thử.
4. **Khôi phục.** Thử mất channel cùng kernel, server chết trong phiên và phiên mới với URL mới. Ghi thao tác nào tự phục hồi, thao tác nào cần connect/start. Đối chiếu PID và GPU sau restart. Kết thúc bằng stop server rồi dừng session Kaggle trên giao diện. Ghi PASS/FAIL/chưa thử cho từng ca. [Các bước 7–8](TESTING.md#7-đo-độ-bền-và-thử-gián-đoạn-thực)

Mốc để chuyển sang tối ưu serving là các lỗi P0 có regression đạt, vòng Kaggle trên có bằng chứng về toàn vẹn stream, thời hạn phản hồi và khả năng phục hồi. Checkpoint/resume, backend khác, MCP và đánh giá chất lượng model thuộc giai đoạn sau.

## Hoàn thiện trải nghiệm CLI sau nghiệm thu runtime

**Quyết định ngày 19/11/2026.** Nên học cách Colab CLI dẫn người dùng qua một công việc hoàn chỉnh. Với kgl, công việc đầu tiên nên là mở notebook GPU, gắn CLI vào phiên đang chạy, xác nhận kernel, nhận phản hồi đầu tiên và biết cách dừng sử dụng GPU. Giữ argparse và kiến trúc Jupyter hiện tại. Hai vướng mắc đã gặp trong lần dùng thật là dán URL nhưng không thấy ký tự và tưởng proxy foreground bị treo. Đây là hai tình huống cần đưa vào phép thử trải nghiệm đầu tiên.

### Những điểm đối chiếu trực tiếp

| Thiết kế ở Colab CLI | Hiện trạng và quyết định cho kgl |
|---|---|
| README đi từ quick start đến danh mục lệnh và ví dụ theo công việc. [README Colab CLI](https://github.com/googlecolab/google-colab-cli#quick-start) | README kgl đang đưa cả truyền file và doctor vào đường đi tới inference. Ví dụ download còn phụ thuộc một file chưa được tạo. Tách đường tối thiểu khỏi bài kiểm thử và các ví dụ tùy chọn. |
| CLI tổ chức nhóm lệnh, sắp xếp help và tránh banner làm bẩn output của một số lệnh. Phần execution kiểm tra TTY. [cli.py](https://github.com/googlecolab/google-colab-cli/blob/main/src/colab_cli/cli.py) · [execution.py](https://github.com/googlecolab/google-colab-cli/blob/main/src/colab_cli/commands/execution.py) | kgl đã có stdin, exec file, exit code, status/infer JSON. Chuẩn hóa các đường này và thêm hướng dẫn ngay ở điểm lỗi. Chưa cần đổi framework hoặc tách mọi lệnh thành một module. |
| State store dùng khóa cho đọc/ghi từ nhiều tiến trình. [state.py](https://github.com/googlecolab/google-colab-cli/blob/main/src/colab_cli/state.py) | kgl lưu một session với file tạm tên cố định rồi replace. Hai terminal có thể tranh file tạm hoặc lệnh cũ ghi đè session vừa connect. Ưu tiên giữ đúng session trước khi cân nhắc hỗ trợ nhiều profile. [State hiện tại](../src/kgl/state.py) |
| Dự án có hướng dẫn riêng cho agent về xác thực, chọn phiên và chạy lệnh headless. Cấp phát và thu hồi VM dùng client riêng của Colab. [Hướng dẫn operator](https://github.com/googlecolab/google-colab-cli/blob/main/skills/colab-operator/SKILL.md) · [Colab client](https://github.com/googlecolab/google-colab-cli/blob/main/src/colab_cli/client.py) | Viết một ví dụ script/agent chạy được với URL người dùng đã cấp. Kaggle vẫn cần người dùng mở và dừng phiên trên giao diện. Hoãn tự cấp GPU, tự gia hạn phiên, SSH và auto-update. |

### Các bản thay đổi đề xuất, theo thứ tự phụ thuộc

| Mốc | Phạm vi triển khai | Tiêu chí nghiệm thu để chuyển bước |
|---|---|---|
| **0 · Đóng nghiệm thu vận hành** | Dùng quy trình ở mục trên để quan sát một giờ sử dụng thực và chuyển từ phiên hết hạn sang notebook mới. Lưu phiên bản mã, thời gian, mã lỗi, thao tác connect/start lại và kết quả request tiếp theo. | Ca đã biết vẫn qua trên myenv: suite local, 20/20 request, ngắt WS giữa stream và phục hồi. Phiên cũ báo mất kết nối trong deadline; URL mới cho phép connect, start, infer thành công. Nếu còn lỗi, ưu tiên sửa ở runtime. |
| **1 · Rút ngắn lần dùng đầu** | Một PR cho README, help và CLI. Đưa chỉ dẫn lấy VS Code Compatible URL ngay tại bước connect. Khi thiếu URL ở terminal tương tác, đề xuất nhập ẩn với lời nhắc rõ và xác nhận đã nhận; giữ nguyên môi trường, cờ URL và stdin cho script. Quick start chỉ gồm connect → exec → serve start → infer → stop. Proxy có ví dụ riêng ghi rõ terminal A/B, readiness và cách dừng session Kaggle. | Người chưa biết dự án làm theo README từ checkout sạch, nhận được kết quả và dừng đúng phiên mà không cần đọc tài liệu test. Test CLI xác nhận URL không lộ, stdin không mở prompt, thiếu input headless thoát có hướng dẫn. Ghi thời gian thao tác riêng với thời gian tải model. |
| **2 · Giữ đúng phiên khi nhiều terminal hoạt động** | Một PR cho state và caller trong CLI. Dùng file tạm riêng, replace nguyên tử và kiểm tra thế hệ/định danh session khi cập nhật hoặc xóa. Khóa ngắn quanh thao tác local, tránh giữ khóa trong lúc chờ mạng. Đọc được session.json cũ và giữ quyền truy cập hiện hành. | Test bằng hai process: A bắt đầu lệnh trên phiên cũ, B connect phiên mới, A hoàn tất muộn. Phiên B vẫn nguyên vẹn, không lỗi file tạm và không lẫn metadata server. Thử cả khi process ghi bị ngắt. |
| **3 · Hoàn thiện đường chạy cho script và agent** | Dựa trên các output JSON có sẵn. Ghi rõ stdout chứa kết quả, stderr chứa tiến trình/lỗi, mã thoát và cách xử lý session mất. Giữ schema đang có; khi cần thêm trường, bổ sung tương thích. Viết một ví dụ dùng status, /ready, infer và xử lý stream gián đoạn. | Chạy ví dụ trong subprocess không TTY; parse được toàn bộ stdout ở chế độ JSON. Test thành công, URL hết hạn, model chưa sẵn sàng và đứt stream. Agent phân biệt được lệnh có thể thử lại với lệnh có kết quả thực thi chưa xác định. |
| **4 · Kiểm tra bản đóng gói** | Tách test offline/localhost và smoke Kaggle opt-in. Build wheel từ checkout sạch, kiểm tra entry point và script server được đóng gói. Chạy test bản Python tối thiểu đã khai báo và bản dùng ở myenv; ghi phạm vi hệ điều hành thực sự đã thử. | Cài wheel vào đích thử cô lập, chạy từ ngoài repo: help/version, connect fake Jupyter, serve echo và infer thành công. Test mặc định chạy khi không có tài khoản Kaggle. Chỉ đánh dấu bản sẵn sàng phân phối khi cả quy trình cài mới và quy trình vận hành đạt. |

**Thứ tự mở rộng sau các mốc trên.** Nếu số đo cho thấy người dùng mất nhiều thời gian chuyển nhiều file, thử nghiệm sync có dry-run và kiểm SHA-256 trước. Nếu tác vụ chính là chạy notebook, đánh giá exec .ipynb với output từng cell và lỗi theo cell. Chọn dựa trên công việc thực tế. Đo riêng thời gian xếp hàng, mở channel, TTFT và thời gian sinh trước khi quyết định tối ưu transport hoặc tăng concurrency.

### Trạng thái triển khai trên máy Linux, 19/11/2026

| Mốc | Kết quả |
|---|---|
| 0 | Smoke Kaggle qua 20/20 request trên CUDA, p50 0,433 s, p95 0,449 s. Đã phục hồi sau ngắt WS và dừng instance thử. Phép đo một giờ và chuyển sang notebook mới sau hết hạn còn mở. |
| 1 | Connect hỗ trợ nhập URL ẩn ở TTY, cờ và stdin cho script. README có luồng dùng lần đầu và ví dụ hai terminal. Đã kiểm tra prompt và headless tự động; chưa đo thời gian thao tác của người dùng mới. |
| 2 | Save/clear ở CLI đối chiếu phiên bản state dưới khóa local. File ghi tạm riêng được thay nguyên tử. Test process thứ hai thay session và process ghi bị ngắt đều đạt. |
| 3 | [Ví dụ agent](../examples/agent_infer.py) kiểm tra status, readiness và chỉ in JSON sau khi nhận dấu kết thúc SSE. Có test subprocess cho thành công, session mất, readiness 503 và stream bị cắt. |
| 4 | 40 test local qua với myenv Python 3.12.16. Wheel build từ bản sao source sạch, cài vào thư mục tạm và chạy CLI ngoài repo qua fake Jupyter. Đã cấu hình CI Ubuntu cho 3.10 và 3.12. Python 3.10 chưa được chạy trên máy này. |

Từ gốc repo, chạy suite với `PYTHONPATH=src python -m unittest discover -s tests -t . -q` và kiểm tra bản phân phối với `python -m tests.wheel_smoke`. Bài Kaggle opt-in dùng session của người chạy theo [hướng dẫn nghiệm thu](TESTING.md).
