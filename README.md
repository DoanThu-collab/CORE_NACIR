# NACIR++ — Plug-and-Play Edition

Bản refactor này **giữ nguyên 100% logic toán học / xử lý gốc** của NACIR++
(Concept Memory Board, Orthogonal Projection / Vector Surgery, Attention
Masking, ITM Re-ranking, lịch trình động alpha/beta/ortho/penalty/itm-weight,
công thức Hits@K / Recall@K / BRI) — chỉ **tách các phần đặc thù PlugIR /
VisDial / BLIP ra khỏi core**, để bất kỳ phương pháp tìm-kiếm-ảnh-tương-tác-
đa-vòng nào khác cũng cắm được vào mà không phải sửa một dòng công thức nào.

## Vì sao bản gốc chưa "plug-and-play thực sự"?

Bản gốc (`main.py`) hardcode:
- Đường dẫn dataset PlugIR/VisDial
- Model BLIP cụ thể (`BlipForRetrieval`)
- 11 vòng hội thoại cố định
- Lịch trình động α/β/ortho/penalty/itm-weight viết thẳng trong vòng lặp
- `Corpus`/`Queries` Dataset chỉ hiểu đúng format VisDial

=> Muốn dùng NACIR++ cho một phương pháp khác (backbone khác, dataset khác,
số vòng hội thoại khác, bộ trích belief khác...) thì phải **sửa trực tiếp
vào code lõi** — không phải plug-and-play.

## Cái gì đã đổi, cái gì KHÔNG đổi

| | Trạng thái |
|---|---|
| `core/concept_memory.py` (Step 2) | **Không đổi 1 dòng công thức** |
| `core/orthogonal_projection.py` (Step 3) | **Không đổi 1 dòng công thức** |
| `core/attention_masking.py` (Step 4) | **Không đổi 1 dòng công thức** |
| `core/reranker.py` (Step 5) | Công thức combine điểm giữ nguyên; chỉ thay lời gọi BLIP cứng bằng interface `ImageScorer` |
| Lịch trình động (dynamic scheduling) | Công thức giữ nguyên (`config.py::default_dynamic_schedule`), chỉ chuyển từ "viết thẳng trong vòng lặp" sang "hàm pluggable" |
| Hits@K / Recall@K / BRI | Công thức giữ nguyên 1:1 (`metrics.py`) |
| PlugIR / VisDial / BLIP | Chuyển thành **adapter** trong `adapters/` — không còn nằm trong core |

## Hợp đồng Input/Output thống nhất (`schema.py` + `interfaces.py`)

Bất kỳ phương pháp nào muốn cắm vào NACIR++ chỉ cần cung cấp:

**INPUT**
1. `corpus_vectors: Tensor[N, D]` — embedding ảnh của corpus (từ backbone bất kỳ)
2. Với mỗi turn hội thoại, một `DialogTurn`:
   - `query_text` (bắt buộc, dùng cho rerank) và/hoặc `query_vector` (nếu bạn đã tự encode)
   - `question`/`answer` (nếu muốn NACIR++ tự trích belief) **hoặc** `beliefs` (nếu bạn đã có sẵn positive/negative concepts)
3. (tuỳ chọn) một trong 3 "chân cắm" chuẩn hoá:
   - `TextEncoder.encode(texts) -> Tensor[B,D]`
   - `BeliefSource.get_beliefs(session_id, turn_index, question, answer) -> BeliefBundle`
   - `ImageScorer.score(query_text, image_refs) -> Tensor[K]` (chỉ cần nếu muốn re-rank)

**OUTPUT** (`SessionOutput` / `TurnOutput`)
- `query_vector` — vector đã "phẫu thuật" (sau Concept Memory + Orthogonal Projection)
- `scores`, `ranked_indices`, `top_k_indices` — kết quả retrieval sau khi masking (và rerank nếu có)
- `target_rank` — thứ hạng ảnh đích (nếu bạn cấp `target_index`, dùng để tính metric)

## Cấu trúc thư mục

```
nacir_plusplus/
  config.py            # NACIRPlusPlusConfig + lịch trình động (pluggable)
  schema.py             # DialogTurn / RetrievalSession / TurnOutput / SessionOutput / BeliefBundle
  interfaces.py         # TextEncoder / BeliefSource / ImageScorer (Protocol)
  pipeline.py           # NACIRPlusPlusPipeline — bộ điều phối chính (Plug-and-Play)
  metrics.py            # Hits@K / Recall@K / BRI (method-agnostic)
  core/                 # KHÔNG ĐỔI so với bản gốc
    concept_memory.py
    orthogonal_projection.py
    attention_masking.py
    reranker.py          # (tổng quát hoá scorer, công thức combine giữ nguyên)
  adapters/              # Các "lớp cắm" cụ thể — nơi chứa mọi thứ đặc thù dataset/backbone
    blip_backbone.py      # BLIP TextEncoder + ImageScorer (để tái lập kết quả gốc)
    belief_sources.py     # PrecomputedBeliefSource (JSON) + RuleBasedBeliefSource (regex)
    visdial_corpus.py     # Corpus/Queries Dataset cho VisDial/PlugIR

examples/
  run_plugir_visdial.py  # Tái lập chính xác kết quả gốc (BRI = 0.6861) bằng kiến trúc mới
  run_generic_demo.py    # Chứng minh cắm được vào MỘT PHƯƠNG PHÁP HOÀN TOÀN KHÁC
```

## Cách cắm một phương pháp mới (3 bước)

```python
from nacir_plusplus import NACIRPlusPlusPipeline, NACIRPlusPlusConfig, OPTIMAL_CONFIG
from nacir_plusplus.schema import DialogTurn, RetrievalSession

# 1) Bạn có: corpus_vectors [N, D] từ backbone của bạn, và text_encoder/belief_source
#    (tự viết hoặc dùng RuleBasedBeliefSource/PrecomputedBeliefSource có sẵn)
pipeline = NACIRPlusPlusPipeline(
    config=OPTIMAL_CONFIG,           # hoặc NACIRPlusPlusConfig(...) tự chỉnh
    corpus_vectors=my_corpus_vectors,
    text_encoder=my_text_encoder,     # implement .encode(list[str]) -> Tensor
    belief_source=my_belief_source,   # implement .get_beliefs(...) -> BeliefBundle
    image_scorer=None,                # bỏ qua nếu phương pháp không cần rerank
)

# 2) Đóng gói hội thoại của bạn thành RetrievalSession
session = RetrievalSession(
    session_id="query-123",
    target_index=42,  # index ảnh đúng trong corpus (để tính metric); None nếu chỉ suy luận
    turns=[
        DialogTurn(turn_index=0, query_text="a red backpack"),
        DialogTurn(turn_index=1, query_text="a red backpack", question="Is it black?", answer="No, not black"),
        ...
    ],
)

# 3) Chạy — core logic NACIR++ (Concept Memory / Orthogonal Projection / Masking)
#    áp dụng y hệt bản gốc, bất kể backbone/dataset của bạn là gì.
result = pipeline.run_session(session)
for turn in result.turns:
    print(turn.turn_index, turn.target_rank, turn.top_k_indices)
```

Xem `examples/run_generic_demo.py` để thấy ví dụ đầy đủ chạy được ngay (không
cần BLIP, không cần GPU, không cần dataset VisDial).

## Chạy lại kết quả gốc

```bash
CUDA_VISIBLE_DEVICES=0 python examples/run_plugir_visdial.py
```

Dùng đúng `OPTIMAL_CONFIG`, đúng lịch trình động, đúng công thức metric như
`main.py` gốc — chỉ khác là toàn bộ phần PlugIR/BLIP giờ nằm trong
`adapters/`, không còn trộn lẫn với core logic NACIR++.
