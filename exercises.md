# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> *Khi temperature = 0.0, phản hồi mang tính xác định cao, nhất quán và tập trung vào các sự thật phổ biến nhất. Khi tăng temperature lên (0.5 đến 1.0), câu trả lời trở nên đa dạng, tự nhiên và phong phú hơn về mặt từ vựng. Ở mức 1.5, phản hồi bắt đầu có dấu hiệu lan man, hỗn loạn hoặc xuất hiện các từ ngữ bất hợp lý do mô hình chọn các token có xác suất xuất hiện rất thấp.*

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> *Tôi sẽ đặt temperature = 0.0 đến 0.2. Đối với chatbot hỗ trợ khách hàng, ưu tiên hàng đầu là tính chính xác, nhất quán và độ tin cậy của thông tin (tránh việc mô hình "sáng tác" ra các chính sách, giá cả hoặc quy định sai sự thật).*

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> *Ước tính chi phí: GPT-4o đắt hơn GPT-4o-mini khoảng 16.67 lần cho output ($10/1M tokens so với $0.60/1M tokens). Với workload $10.000 \times 3 \times 350 = 10.500.000$ tokens/ngày, dùng GPT-4o tốn khoảng $105/ngày, trong khi GPT-4o-mini chỉ tốn khoảng $6.3/ngày.Trường hợp nên dùng GPT-4o: Khi cần xử lý logic phức tạp, phân tích tài chính/hợp đồng pháp lý, hoặc giải quyết các khiếu nại nhạy cảm đòi hỏi khả năng suy luận cao.Trường hợp nên dùng mini: Khi xử lý các tác vụ phân loại ý định khách hàng (intent classification), trả lời các câu hỏi FAQ thường gặp, hoặc tóm tắt các đoạn hội thoại ngắn.*

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> *Sự khác nhau: Phản hồi của "giáo viên tiểu học" ngắn gọn, dùng từ ngữ đơn giản và ẩn dụ hình ảnh (như cuốn sổ nhật ký truyền tay nhau). Trong khi đó, phản hồi của "chuyên gia tài chính" dài hơn, dùng thuật ngữ chuyên ngành (mạng lưới ngang hàng P2P, cơ chế đồng thuận, mã hóa bất đối xứng, sổ cái phân tán). Ảnh hưởng: System prompt định hình toàn bộ tư duy, giọng văn (tone/style) và giới hạn không gian tri thức mà AI được phép truy xuất, giúp hướng lái phản hồi đúng đối tượng mục tiêu mà không cần sửa câu hỏi chính.*

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> *Đánh giá & Chênh lệch: Một đoạn văn tiếng Việt ~100 từ thường ngốn khoảng 150–200 token qua tiktoken, cao hơn khoảng 15% – 50% so với công thức ước lượng số từ / 0.75 (~133 token).Nguyên nhân: Bảng mã hóa (tokenizer) của OpenAI được tối ưu hóa cho tiếng Anh. Tiếng Việt có nhiều thanh điệu và từ ghép, khiến các thuật toán như BPE (Byte Pair Encoding) thường bị tách nhỏ một từ tiếng Việt thành nhiều sub-token hoặc byte lẻ, dẫn đến tốn nhiều token hơn tiếng Anh trên cùng một lượng thông tin.*

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> *Streaming quan trọng nhất trong các ứng dụng giao tiếp thời gian thực như Chatbot CLI/Web, trợ lý ảo, hoặc các tác vụ sinh văn bản dài (như viết bài luận, tạo code). Việc hiển thị từng token giúp giảm thời gian chờ cảm nhận (perceived latency) xuống gần như bằng 0, tạo cảm giác phản hồi tức thì cho người dùng. Ngược lại, non-streaming phù hợp hơn khi xử lý ngầm (background jobs), gọi API hệ thống để lấy dữ liệu dạng JSON/Structured Outputs, hoặc khi cần xử lý toàn bộ văn bản đầu ra trước khi lưu vào cơ sở dữ liệu.*

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> *Lợi thế: Exponential backoff tự động dãn cách khoảng thời gian giữa các lần thử lại, giúp hệ thống backend/API có thêm thời gian để tự phục hồi và giảm tải áp lực dồn dập.Nếu dùng delay cố định: Hàng nghìn client thử lại đồng loạt sau mỗi $1s$ sẽ tạo ra hiệu ứng Thundering Herd Problem (hoặc Retry Storm). Điều này làm cho máy chủ API vừa bắt đầu gỡ tải lại tiếp tục bị đánh sập ngay lập tức, dẫn đến nghẽn mạng kéo dài và hệ thống hoàn toàn không thể phục hồi.*

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> *Chỉ định "trả lời ngắn gọn dưới 3 câu" giúp tiết kiệm tối đa số lượng output token (giảm chi phí API) và tối ưu trải nghiệm đọc trên giao diện CLI. Yêu cầu "sử dụng tiếng Việt" đảm bảo trợ lý luôn phản hồi bằng ngôn ngữ người dùng dễ hiểu nhất mà không bị tự động chuyển sang tiếng Anh.*

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> *Hạn chế lớn nhất: Trợ lý chỉ lưu lịch sử 3 lượt gần nhất (sliding window) và sẽ mất toàn bộ bộ nhớ ngữ cảnh ngay khi người dùng đóng/kết thúc phiên chat (không có bộ nhớ dài hạn/persistent storage). Đề xuất cải thiện: Triển khai lưu trữ lịch sử vào CSDL (như SQLite hoặc PostgreSQL) kết hợp với kỹ thuật tóm tắt ngữ cảnh (Context Summarization). Mô tả triển khai: Sau mỗi 5 lượt trò chuyện, gọi một LLM phụ để tóm tắt các điểm chính của cuộc hội thoại thành 1-2 câu ngắn, sau đó lưu bản tóm tắt này cùng với thông tin người dùng vào CSDL. Khi mở lại phiên mới, hệ thống tải bản tóm tắt đó vào System Prompt làm bộ nhớ dài hạn cho AI.*

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
