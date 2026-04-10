# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** [Nguyễn Tuấn Kiệt]
**Nhóm:** [C401-A3]
**Ngày:** [10/4/2026]

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> High cosine similarity = 2 vector embedding gần cùng hướng/vector có góc nhỏ so với nhau. Có nghĩa là 2 câu có ngữ nghĩa giống nhau (nhiều từ liên quan, trùng lặp v.)

**Ví dụ HIGH similarity:**
- Sentence A: “Làm sao để đặt lại mật khẩu?” 
- Sentence B: “Cách đổi mật khẩu tài khoản là gì?”
- Tại sao tương đồng: Hai câu cùng ý định (intent) là đổi/đặt lại mật khẩu. Khác từ (“đặt lại”, “đổi”, “tài khoản”) nhưng cùng ngữ nghĩa. Embedding học được sự tương đương này → vector cùng hướng → cosine cao.

**Ví dụ LOW similarity:**
- Sentence A: “Làm sao để đặt lại mật khẩu?”
- Sentence B: “Hôm nay thời tiết có mưa không?”
- Tại sao khác: Domain khác nhau, không có chung intent, ngữ cảnh, từ vựng

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Euclidean bị ảnh hưởng độ dài vector: câu dài/ngắn sẽ có vector norm khác nhau, khoảng cách thay đổi dù nghĩa gần. Cosine similarity chuẩn hóa vector và chỉ đo góc.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:* 1st chunk=500, following effective chunks = 50.
> `#chunks = ceil((10000 - 500) / 450 + 1) = 23`
> *Đáp án:* 23

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
``` 
chunks = ceil((10000 − 500) / 400) + 1
= ceil(9500 / 400) + 1
= ceil(23.75) + 1
= 24 + 1 = 25 chunks
```
- Giữ ngữ cảnh ở ranh giới chunk (câu/ý không bị cắt cụt), tăng recall khi retrieval (query khớp nội dung nằm gần biên), giảm mất thông tin liên kết giữa các đoạn


---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:**  FAQ về nhà trường cho học sinh (thông tin synthetic)

**Tại sao nhóm chọn domain này?**
> Dữ liệu đơn giản, dễ tạo synthetic, không phụ thuộc nguồn ngoài.
> Cấu trúc FAQ rõ ràng (question–answer) → dễ chunk, dễ đánh giá retrieval.
> Bao phủ nhiều intent phổ biến (học phí, lịch học, quy định, hỗ trợ) → test tốt semantic search.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | dang_ky_hoc_phan.md | Phòng Đào tạo | 2,599 | type: hoc_vu, urgency: high |
| 2 | diem_va_xep_loai.md | Sổ tay sinh viên | 2,642 | type: hoc_vu, urgency: medium |
| 3 | ho_tro_sinh_vien.md | Phòng CT&CTSV | 2,825 | type: chinh_sach, urgency: low |
| 4 | hoc_phi_hoc_bong.md | Phòng Kế hoạch Tài chính | 2,783 | type: tai_chinh, urgency: medium |
| 5 | ky_luat_chuyen_can.md | Sổ tay sinh viên | 2,908 | type: ky_luat, urgency: low |
| 6 | tot_nghiep.md | Quy định tốt nghiệp | 3,012 | type: tot_nghiep, urgency: high |



### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| type | string | hoc_vu, tai_chinh | Cho phép sinh viên lọc câu hỏi theo mảng kiến thức (ví dụ: chỉ tìm trong mảng Học phí). |
| urgency | string | high, medium | Giúp hệ thống ưu tiên hiển thị các quy định có tính thời hạn hoặc mức độ quan trọng cao. |


---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| `dang_ky_hoc_phan.md` | FixedSizeChunker (`fixed_size`) | 13 | 193 | No |
| `dang_ky_hoc_phan.md`| SentenceChunker (`by_sentences`) |12  |156 | Yes |
| `dang_ky_hoc_phan.md`| RecursiveChunker (`recursive`) | 31 | 59 | Yes |
| `diem_va_xep_loai.md` | FixedSizeChunker (`fixed_size`) | 14 | 189 | No |
| `diem_va_xep_loai.md`| SentenceChunker (`by_sentences`) | 11 |179 | Yes |
| `diem_va_xep_loai.md`| RecursiveChunker (`recursive`) | 29 | 67 | Yes |



### Strategy Của Tôi

**Loại:**  RecursiveChunker

**Mô tả cách hoạt động:**
> Text split theo separator list ưu tiên giảm dần: paragraph → sentence → clause → token limit. Nếu chunk > max size, tiếp tục split lại bằng rule tiếp theo.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Document có self-contained knowledge: Các câu thường trả lời 1 câu hỏi hoàn chỉnh
> Recursive chunker sẽ cho 1 chunk 1 câu, phù hợp


### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
|`tot_nghiep.md` | baseline (paragraph/sentence markdown parsing) | 7 | 323 | Kept full context of surrounding sentences |
|`tot_nghiep.md` | **của tôi** | 29 | 76 | Kept full context of the target sentence |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Nguyễn Văn Bách | Recursive | 9/10 | Giữ trọn vẹn ngữ cảnh Điều/Khoản | Cấu trúc đệ quy phức tạp |
| Nguyễn Duy Hưng| Semantic | 4 / 10 | Giữ trọn vẹn ngữ cảnh của đoạn văn, gom chung nhóm ý tưởng rất tốt để làm context cho LLM. | Tốn nhiều bước tính toán (chạy nhúng từng câu), nếu xác định ngưỡng threshold sai thì có thể thu nhầm cả cụm đoạn không liên quan. |
| Nguyễn Đức Duy | SentenceChunker (3 câu) | 8 | Giữ nguyên ý trọn vẹn, phù hợp văn bản quy chế | Chunk dài hơn, avg ~300 chars |
| Trần Trọng Giang | MarkdownHeader | 8/10 | Tối ưu tuyệt đối cho cấu trúc Markdown | Chỉ hiệu quả với file có Header rõ ràng | 
| Nguyễn Xuân Hoàng | FixedSize (Size 200, Overlap 50) | 4/10 | Triển khai nhanh, dễ tính toán số lượng | Hay cắt ngang từ, phân mảnh câu |
| Nguyễn Tuấn Kiệt | Recursive | 8/10 | Lấy được điều khoản đúng, giữ được context | Score match khá thấp dù top 3 đúng, 1 chunk = 1 câu có khả năng mất info dài |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> Recursive - Balance tốt giữa chunk size và context retrieval

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> regex `(?<=[.!?])\s+` để tách câu dựa trên dấu kết thúc (. ! ?) kèm khoảng trắng phía sau. Tránh mất dấu câu và không phụ thuộc spacing cụ thể. 
> Edge cases chưa xử lý: viết tắt (e.g., Dr.), newline (\n), hoặc câu không có dấu kết thúc → cần strip và filter empty.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Chia văn bản theo thứ tự separator từ lớn → nhỏ (\n\n → \n → space → ký tự).
> Với mỗi phần sau khi split, nếu vẫn > chunk_size thì tiếp tục đệ quy với separator nhỏ hơn.
> Base case: khi len(text) <= chunk_size → trả về `[text]`; hoặc hết separator → trả về `[text]` (dù còn dài).

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Mỗi document được embed thành vector và lưu cùng id, content, metadata (thêm doc_id tại chromadb check `valid_metadata`). Với Chroma: lưu trực tiếp vào collection; in-memory: lưu dict + embedding. 
> Khi search: embed query, tính similarity bằng dot product (hoặc cosine), sort giảm dần theo score, trả top_k.

**`search_with_filter` + `delete_document`** — approach:
> Filter trước rồi mới search để thu hẹp candidate set (Chroma dùng where, in-memory lọc list trước khi tính similarity).
> Delete theo metadata["doc_id"]: Chroma gọi delete(where=...), return DeleteResult, trả true nếu số trong result > 0.
> in-memory xây lại list trừ các record match, đồng thời trả True nếu có phần tử bị xóa.

### KnowledgeBaseAgent

**`answer`** — approach:
> Prompt Structure: `Question + Context + Answer`
> Prompt đặt instruction ngắn gọn để ép model chỉ trả lời dựa trên context, tránh hallucination

### Test Results

`================================================= 42 passed in 0.75s ==================================================
`
**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|

| The cat sat on the mat. | The cat is on the mat. | High | 0.8666 | Yes | 
| I love programming Python. | Python is a type of snake. | Low | 0.5119 | No |
| The weather is sunny today. | "It is a bright, sunny day." | High | 0.7823 | Yes |
| Apples are my favorite fruit. | I really enjoy eating apples. | High | 0.7169 | Yes |
| Democracy is a form of government. | Pizza is a popular Italian food. | Low | 0.1103 | Yes |

(code for this is in file `similarity_test.py`, test on OpenAI Embedder)

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Kết quả bất ngờ nhất là Cặp 2 (Python), vì mức độ tương đồng vẫn ở mức trung bình (~0.51) dù ngữ cảnh hoàn toàn khác nhau. Điều này cho thấy embeddings không chỉ biểu diễn nghĩa của từng từ riêng lẻ mà còn bị ảnh hưởng bởi sự xuất hiện của các từ khóa chung (như "Python"). Trùng lặp từ vựng vẫn có thể làm nhiễu kết quả đánh giá ngữ cảnh thực tế.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Sinh viên cần bao nhiêu tín chỉ để tốt nghiệp? | Dao động từ 120 đến 150 tín chỉ tùy theo ngành đào tạo. |
| 2 | GPA bao nhiêu thì được làm luận văn tốt nghiệp? | Sinh viên có GPA tích lũy từ 2.8 trở lên. |
| 3 | Những đối tượng nào được miễn giảm học phí? | Sinh viên thuộc hộ nghèo, cận nghèo, con thương binh liệt sĩ, khuyết tật. |
| 4 | Khi nào sinh viên bị cảnh cáo học vụ? | Khi GPA học kỳ thấp dưới 1.0 hoặc GPA tích lũy dưới 1.2 (năm 1). |
| 5 | Chuẩn đầu ra ngoại ngữ để tốt nghiệp là gì? | Chứng chỉ B1 quốc tế hoặc tương đương theo khung 6 bậc Việt Nam. |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Sinh viên cần bao nhiêu tín chỉ để tốt nghiệp?  | Sinh viên được xét tốt nghiệp khi đáp ứng đầy đủ các điều kiện sau: tích lũy đủ số tín chỉ... |0.176 | Yes | tích lũy đủ số tín chỉ theo chương trình đào tạo (thường từ 120 đến 150 tín chỉ) |
| 2 | GPA bao nhiêu thì được làm luận văn tốt nghiệp?  | Sinh viên có GPA tích lũy từ 2.8 trở lên được đăng ký làm luận văn... | 0.388 | Yes | Sinh viên có GPA tích lũy từ 2.8 trở lên được đăng ký làm luận văn tốt nghiệp |
| 3 | Những đối tượng nào được miễn giảm học phí?  | ## Miễn giảm học phí... | 0.627 | No (header) | sinh viên thuộc hộ nghèo, hộ cận nghèo, sinh viên là người dân tộc thiểu số... (chunk top 2) |
| 4 | Khi nào sinh viên bị cảnh cáo học vụ? | Dịch vụ hỗ trợ sinh viên... | 0.019 | No |# Dịch vụ hỗ trợ sinh viên... (relevant chunk is no.3) |
| 5 | Chuẩn đầu ra ngoại ngữ để tốt nghiệp là gì? | ## Chuẩn đầu ra ngoại ngữ | 0.125 | No (header) |Sinh viên được xét tốt nghiệp khi ... (irrelevant) |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 4 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> tạo synthetic data, parse PDF thành markdown, text processing cleanup techniques

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> cách viết file thử nghiệm nhanh (`similarity_test.py), cách chạy main để thử function, kĩ năng sửa code do AI viết

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> chú trọng hơn vào việc chọn dữ liệu dễ tiêu chuẩn hoá (chuyển đổi thành .md), loại bỏ các loại formatting làm nhiễu embedding

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 8 / 10 |
| Chunking strategy | Nhóm | 12 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 5 / 5 |
| **Tổng** | | ** 85 / 90 ** |
