# Đánh giá Qwen cho công việc hằng ngày

Chạy từ gốc repo bằng Python của `myenv` khi [worker và proxy Qwen](../QWEN35B-KAGGLE.md) đã sẵn sàng. Chạy từng profile tuần tự vì worker có một slot. Runner đọc `KGL_PROXY_KEY` nếu proxy yêu cầu xác thực.

```bash
python examples/qwen35b-llama/eval/run.py gui
python examples/qwen35b-llama/eval/run.py api
python examples/qwen35b-llama/eval/run.py script
python examples/qwen35b-llama/eval/run.py perf
python examples/qwen35b-llama/eval/run.py perf --short-output
python -m unittest discover -s examples/qwen35b-llama/eval -v
```

- Profile mặc định dùng lại tham số M0. Để đo cấu hình chat mới, chạy `run.py gui --thinking off --max-tokens 2048`. Khi cần chọn ca dài hoặc so cùng prompt với Thinking bật, thêm `--case S4` và đổi sang `--thinking on --max-tokens 2048`.
- `gui` gửi SSE, yêu cầu usage và giới hạn 768 token như chat khi biết context. `api` gửi JSON qua alias và dùng defaults của preset. `script` tái hiện giới hạn 512 token và greedy của action_items.py, dùng cùng messages để đối chiếu công bằng giữa ba profile.
- `perf` gửi ba prompt dài qua SSE để đo từ máy local. Dùng usage để kiểm độ dài thực, timings để biết số token cache và tốc độ sinh trong worker. Tính thời gian tới content từ lần gọi HTTP đến delta content đầu tiên có ký tự hữu ích. Dùng `--short-output` để đo memo 100–130 từ với giới hạn 1.024 token và ba prefix khác. Với JSON chỉ có thời gian nhận toàn bộ.
- Dùng `--case E1 --case M1` để chạy lại ca cụ thể. Runner in JSONL gồm số đo và câu trả lời cho dữ liệu mẫu. Chọn thư mục ngoài repo khi lưu kết quả thô. Runner trả mã 1 nếu có ca gặp lỗi vận chuyển, phản hồi thiếu/cắt hoặc trượt kiểm tra tự động.
- `needs_review` nghĩa là qua kiểm tra từ khóa. Đọc câu trả lời, đối chiếu rubric trong cases.json và ghi lý do khi trượt. Với ca tổng hợp hoặc soạn thảo, kiểm cả định dạng, ngôn ngữ và dữ kiện phát sinh. M2 và W2 cần đối chiếu chữ hoa/thường hoặc ký tự nguyên gốc theo rubric. Mỗi lượt hội thoại nhiều lượt dùng lịch sử đã ghim để các profile nhận cùng dữ kiện.
- Tiêu chí phát hành áp dụng cho cấu hình được chọn sau hiệu chỉnh. Giữ lại kết quả từng lần chạy và các ca trượt khi thử cấu hình mới. [Thiết kế v3](../../../docs/DESIGN-v3.md) · [Kế hoạch theo lượt](../../../docs/PLAN-qwen-daily.md)
