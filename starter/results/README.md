# Kết quả và bảng audit

Đây là kết quả lượt Gemini Embedding 2 trên Python Standard Library 3.12.10. B được chốt theo dev trước khi chạy test Gemini. Năm câu test đã từng dùng với Ollama, nên kết quả là lượt chạy lại trên cùng bộ câu hỏi. Xem [báo cáo cá nhân](../report/individual/2A202603713_LeGiaHung.md).

| Tệp | Dùng để kiểm tra |
|---|---|
| [index.json](index.json), [pipeline.log](pipeline.log) | Số chunk, cấu hình index, token, thời gian và các lần thử lại HTTP 429 |
| [selection.json](selection.json) | Cấu hình được chọn, thời điểm chốt, checksum corpus và benchmark |
| [Dev: chỉ số](dev_metrics.json) · [trace](dev_traces.json) | 15 câu, hạng, text, điểm tương đồng, metadata và nhãn evidence |
| [Test: chỉ số](test_metrics.json) · [trace](test_traces.json) | 5 câu, cùng định dạng trace |
| [Coherence: mẫu](coherence_samples.json) · [nhãn](coherence_audit.csv) | 10 vị trí nguồn × 3 cấu hình |
| [Metadata](metadata_audit.csv) | 5 câu test × 3 hit × 3 cấu hình |
| [Grounding](grounding_audit.csv) | 5 câu test × 3 cấu hình |

1. **Gold evidence:** đọc đáp án và trích đoạn của 20 câu trong [bộ benchmark](../benchmark/questions.json); dùng ID nguồn và mục để tìm lại trong [manifest](../data/manifest.json). Lệnh `validate` kiểm tra trích đoạn có tồn tại trong corpus; người đọc kiểm tra trích đoạn có đủ để hỗ trợ đáp án.
2. **Coherence:** nối nhãn với mẫu bằng `strategy`, `position` và `chunk_id`. Lấy 10 đơn vị đầu theo thứ tự `(document_id, page)`, chọn vị trí giữa văn bản (dịch sang ký tự không trắng kế tiếp nếu cần), rồi lấy chunk bao phủ có offset bắt đầu thấp nhất. Chấm 0 nếu ý chính khó hiểu, 1 nếu hiểu một phần, 2 nếu giữ trọn ý tại vị trí mẫu.
3. **Metadata:** nối bằng `strategy`, `question_id` và `rank`. Đối chiếu URL, mục, anchor, phiên bản, phạm vi và offset với manifest, corpus. Ở cột kiểm tra scope, ô trống nghĩa là câu không yêu cầu lọc.
4. **Grounding:** đọc cả top-3 của cùng câu test cho A, B và C. Chấm 0 nếu thiếu bằng chứng, 1 nếu có một phần nhưng thiếu điều kiện hoặc ngữ cảnh, 2 nếu đủ để trả lời và truy về nguồn. Đây là điểm cho bằng chứng truy xuất.

Hit@1 và Evidence Hit@3 được tính từ trace. Điểm coherence, metadata và grounding được tổng hợp riêng từ các bảng CSV.
