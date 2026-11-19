# Project: Kaggle GPU CLI & Model Serving (đề bài gốc)

> Bản sao nguyên văn đề bài do người dùng cung cấp, lưu để đối chiếu với `PLAN.md`.

Bạn là Senior Software Architect và AI Infrastructure Engineer, có kinh nghiệm về Python, Jupyter Protocol, WebSocket, GPU computing, model inference và CLI development.

## Mục tiêu

Nghiên cứu, thiết kế và triển khai một công cụ mã nguồn mở tương tự [google-colab-cli](https://github.com/googlecolab/google-colab-cli), nhưng sử dụng **Kaggle làm GPU backend**.

Công cụ cho phép lập trình viên hoặc AI coding agents kết nối và tận dụng GPU Kaggle từ máy cá nhân để thực thi Python, phát triển, chạy inference và serving các mô hình AI mà không phải thao tác trực tiếp trên Kaggle Notebook.

Ưu tiên trải nghiệm CLI đơn giản, ổn định, dễ sử dụng cho cá nhân có tài nguyên hạn chế.

## Yêu cầu cốt lõi

1. **Kaggle Runtime Connection**
   - Khởi tạo hoặc kết nối tới Kaggle GPU runtime.
   - Nghiên cứu Kaggle API chính thức, Jupyter Kernel Protocol, REST/WebSocket và các giải pháp cộng đồng.
   - Hỗ trợ gửi code, nhận output, quản lý session và phát hiện mất kết nối.
   - Không giả định Kaggle cung cấp API tương tác chính thức nếu chưa xác minh.

2. **Remote Execution & File Management**
   - Chạy Python script, notebook hoặc lệnh trực tiếp từ CLI.
   - Upload/download file, đồng bộ project và quản lý dependencies.
   - Stream logs, theo dõi GPU/RAM và xử lý lỗi runtime.

3. **GPU Model Serving**
   - Load và chạy inference với các mô hình AI trên GPU Kaggle.
   - Thiết kế giao diện inference có thể được gọi từ ứng dụng local, CLI hoặc AI agents.
   - Nghiên cứu khả năng xây dựng API tương thích OpenAI API.
   - Đánh giá kỹ giới hạn networking, inbound connections, session timeout và chính sách Kaggle.
   - Nếu không thể public HTTP server ổn định, đề xuất cơ chế request/response qua kết nối được hỗ trợ, chẳng hạn Jupyter messaging, thay vì mặc định dùng public tunnel.
   - Phân biệt rõ interactive inference với production serving 24/7.

4. **Session & Resource Management**
   - Theo dõi thời gian phiên, GPU availability và quota nếu có nguồn dữ liệu đáng tin cậy.
   - Phát hiện session bị ngắt và hỗ trợ recovery.
   - Hỗ trợ checkpoint và resume cho các tác vụ dài.
   - Tuân thủ giới hạn tài nguyên và điều khoản sử dụng của Kaggle; không sử dụng kỹ thuật vượt quota hay né giới hạn phiên.

5. **Developer Experience**
   - CLI đơn giản, nhất quán và dễ mở rộng.
   - Có thể tích hợp Claude Code, Codex và các AI agents khác qua CLI hoặc MCP.
   - Thiết kế theo hướng modular, có khả năng bổ sung Colab backend sau này.

## Quy trình thực hiện

**Giai đoạn 1 — Research & Feasibility**

Đọc và phân tích mã nguồn thực tế của:

- https://github.com/googlecolab/google-colab-cli
- https://github.com/Kaggle/kaggle-api
- https://github.com/JoseGarciaMayen/remote-bridge

Kiểm tra tài liệu Kaggle hiện hành, khả năng networking, điều khoản sử dụng và tính khả thi của từng phương án kết nối.

Phân loại rõ: API chính thức, giải pháp không chính thức, đã kiểm chứng và giả thuyết cần thử nghiệm. Không suy đoán endpoint hay tính năng.

**Giai đoạn 2 — Architecture & Technical Design**

Đề xuất kiến trúc tổng thể, các module chính, giao thức kết nối, lifecycle của GPU runtime và luồng inference request/response.

So sánh những phương án triển khai, phân tích trade-off và lựa chọn thiết kế khả thi nhất cho mục tiêu sử dụng cá nhân.

**Giai đoạn 3 — Proof of Concept**

Ưu tiên kiểm chứng những rủi ro kỹ thuật lớn nhất:

1. Kết nối được với Kaggle GPU runtime.
2. Gửi code Python và nhận kết quả từ local.
3. Load một model nhỏ trên GPU.
4. Gửi inference request từ local và nhận response.
5. Kiểm tra độ ổn định, latency, reconnect và các giới hạn thực tế.

Chỉ chuyển sang thiết kế tính năng phức tạp khi các khả năng nền tảng đã được kiểm chứng.

**Giai đoạn 4 — MVP Implementation**

Xây CLI có các chức năng tối thiểu:

```bash
kgl auth
kgl connect
kgl status
kgl exec
kgl upload
kgl download
kgl serve
kgl infer
kgl stop
```

Đây là giao diện CLI mong muốn, không phải các lệnh hiện có.

Ưu tiên code đơn giản, có typing, logging, error handling, cấu hình rõ ràng và automated tests.

**Giai đoạn 5 — Validation & Documentation**

Kiểm thử end-to-end trên Kaggle runtime thực tế nếu có quyền truy cập. Viết tài liệu cài đặt, sử dụng, kiến trúc, troubleshooting và giới hạn đã xác minh.

## Nguyên tắc

- Ưu tiên giải pháp hoạt động thực tế hơn kiến trúc phức tạp.
- Không tự giả định Kaggle có khả năng giống Colab.
- Không nhầm lẫn khả năng chạy batch notebook với khả năng phục vụ inference tương tác.
- Không xây dựng các chức năng phụ trước khi xác minh kết nối GPU và inference.
- Tái sử dụng thư viện mã nguồn mở phù hợp thay vì tự triển khai lại toàn bộ.
- Không hardcode credentials hoặc phụ thuộc endpoint nội bộ mà không nêu rõ rủi ro.
- Nếu không có quyền truy cập Kaggle để kiểm thử, phải nêu rõ phần chưa được xác minh, cung cấp PoC và hướng dẫn kiểm thử; không tuyên bố đã hoạt động.
- Khi gặp giới hạn kỹ thuật hoặc chính sách, đề xuất phương án thay thế hợp lệ.

## Kết quả mong muốn

Trước tiên, hãy cung cấp:

1. Đánh giá tính khả thi thực tế, đặc biệt đối với **GPU model serving trên Kaggle**.
2. Kiến trúc đề xuất và lựa chọn giao thức kết nối.
3. Các rủi ro hoặc trở ngại quan trọng có thể khiến dự án thất bại.
4. Phạm vi MVP nhỏ nhất chứng minh được mục tiêu.
5. Kế hoạch triển khai theo thứ tự ưu tiên, có tiêu chí nghiệm thu cụ thể.

Sau đó, nếu có môi trường coding và quyền truy cập repository, hãy bắt đầu triển khai PoC theo kế hoạch. Không dừng ở mô tả lý thuyết khi có thể kiểm chứng bằng code.

**Tiêu chí thành công quan trọng nhất:** Từ máy cá nhân, gửi một inference request đến model đang chạy trên GPU Kaggle và nhận được kết quả một cách đáng tin cậy, trong giới hạn kỹ thuật và chính sách được phép.
