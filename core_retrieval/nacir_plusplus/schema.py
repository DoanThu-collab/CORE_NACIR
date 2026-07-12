"""
NACIR++ Plug-and-Play — Unified I/O Schema
============================================
Đây là "hợp đồng" (contract) chung mà BẤT KỲ phương pháp tìm kiếm ảnh tương tác
đa vòng nào (PlugIR, ChatIR, hay phương pháp tự viết) cũng phải tuân theo để
cắm được vào NACIR++.

Nguyên tắc: NACIR++ không quan tâm phương pháp nền (base method) sinh ra câu
query bằng cách nào (LLM rewrite, template, rule-based...). Nó chỉ cần:
    1. Một vector query cho mỗi turn (đã được base method mã hoá) — INPUT
    2. (tuỳ chọn) beliefs positive/negative cho turn đó — INPUT
    3. Corpus vectors để tính điểm — INPUT (context, không đổi theo turn)
    và trả về:
    4. Vector query đã "phẫu thuật" (updated) — OUTPUT
    5. Điểm số / ranking đã điều chỉnh — OUTPUT

=> Bất kỳ backbone/method nào (BLIP, CLIP, SigLIP, hay retrieval model riêng
   của bạn) chỉ cần đóng gói dữ liệu đúng theo các dataclass dưới đây.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


# ============================================================
# Beliefs (Step 1 output — do phương pháp bên ngoài / extractor cung cấp)
# ============================================================

@dataclass
class Belief:
    """Một khái niệm (concept) được trích xuất từ câu trả lời hội thoại."""
    attribute: str
    confidence: float = 0.7


@dataclass
class BeliefBundle:
    """Gói belief cho MỘT turn của MỘT session."""
    positive_beliefs: List[Belief] = field(default_factory=list)
    negative_beliefs: List[Belief] = field(default_factory=list)

    @staticmethod
    def empty() -> "BeliefBundle":
        return BeliefBundle()

    @staticmethod
    def from_raw(raw: Dict[str, List[Dict[str, Any]]]) -> "BeliefBundle":
        """Dựng từ dict thô kiểu {"positive_beliefs":[{"attribute":..,"confidence":..}], ...}."""
        pos = [
            Belief(attribute=b.get("attribute", ""), confidence=b.get("confidence", 0.7))
            for b in raw.get("positive_beliefs", [])
            if b.get("attribute")
        ]
        neg = [
            Belief(attribute=b.get("attribute", ""), confidence=b.get("confidence", 0.7))
            for b in raw.get("negative_beliefs", [])
            if b.get("attribute")
        ]
        return BeliefBundle(positive_beliefs=pos, negative_beliefs=neg)


# ============================================================
# INPUT — một turn hội thoại / một phiên hội thoại
# ============================================================

@dataclass
class DialogTurn:
    """
    Một vòng hội thoại (turn) đã được BASE METHOD xử lý sẵn.

    query_text:
        Câu query mà base method (PlugIR/ChatIR/...) muốn dùng để retrieve
        ở turn này (có thể là cả lịch sử hội thoại ghép lại, hoặc câu do LLM
        viết lại — NACIR++ không quan tâm cách sinh ra, chỉ cần chuỗi text).
    query_vector:
        (tuỳ chọn) nếu base method đã tự encode sẵn thì truyền thẳng vector,
        Pipeline sẽ bỏ qua bước gọi text_encoder.
    question / answer:
        Dùng để trích beliefs nếu bạn dùng BeliefSource dạng "online extractor".
    beliefs:
        (tuỳ chọn) nếu base method / pipeline khác đã có sẵn beliefs
        (positive/negative concept) thì truyền thẳng vào đây, Pipeline sẽ bỏ
        qua BeliefSource.
    """
    turn_index: int
    query_text: str = ""
    query_vector: Optional[torch.Tensor] = None
    question: str = ""
    answer: str = ""
    beliefs: Optional[BeliefBundle] = None


@dataclass
class RetrievalSession:
    """Một phiên hội thoại tìm kiếm ảnh tương tác đầy đủ (nhiều turn)."""
    session_id: Any
    turns: List[DialogTurn]
    target_index: Optional[int] = None  # index ảnh đích trong corpus (để tính metric); None nếu chỉ inference


# ============================================================
# OUTPUT — kết quả mỗi turn / mỗi session
# ============================================================

@dataclass
class TurnOutput:
    turn_index: int
    query_vector: torch.Tensor            # [D] — query vector đã update (Step 2+3)
    scores: torch.Tensor                  # [N] — điểm số cuối cùng (sau Step 4 + rerank nếu có)
    ranked_indices: torch.Tensor          # [N] — index corpus xếp hạng giảm dần theo scores
    top_k_indices: List[int]              # top-K sau cùng (có thể đã qua re-rank)
    target_rank: Optional[int] = None     # thứ hạng của ảnh đích (None nếu không có target)
    memory_snapshot: Optional[List[Dict]] = None  # (debug) trạng thái Concept Memory Board


@dataclass
class SessionOutput:
    session_id: Any
    turns: List[TurnOutput]

    def target_ranks(self) -> List[Optional[int]]:
        return [t.target_rank for t in self.turns]
