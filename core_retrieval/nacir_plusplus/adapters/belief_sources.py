"""
NACIR++ Plug-and-Play — Belief Source Adapters
==================================================
Bọc lại 2 nguồn belief gốc (utils/negative_detector.py + core/semantic_parser.py)
thành `BeliefSource` chuẩn hoá, KHÔNG đổi logic bên trong:

    1. PrecomputedBeliefSource — load JSON beliefs tính sẵn (Format A / B như
       semantic_parser.load_precomputed_beliefs gốc).
    2. RuleBasedBeliefSource   — dùng utils/negative_detector.py (3-stage cascade,
       hiện Stage 1 rule-based) để tự suy ra positive/negative NGAY TỪ (question,
       answer) — hữu ích khi phương pháp khác không có sẵn file beliefs.

Bất kỳ phương pháp nào khác muốn cấp beliefs theo cách riêng (LLM, NLI, bộ
NLU nội bộ...) chỉ cần viết một class có hàm `get_beliefs(...)` — không cần
kế thừa gì cả (Protocol là structural typing).
"""

import json
import re
from typing import Any, Dict, List, Optional

from ..schema import Belief, BeliefBundle

# ============================================================
# 1. Precomputed (giữ nguyên logic load_precomputed_beliefs gốc)
# ============================================================


def load_precomputed_beliefs(path: str) -> Dict[int, Dict[int, Dict]]:
    """Giữ nguyên 1:1 logic gốc core/semantic_parser.py::load_precomputed_beliefs."""
    with open(path) as f:
        raw = json.load(f)

    result: Dict[int, Dict[int, Dict]] = {}

    if isinstance(raw, list):
        for dialog in raw:
            dialog_id = int(dialog["dialog_id"])
            result[dialog_id] = {}
            for turn in dialog.get("turns", []):
                turn_idx = int(turn["turn"])
                result[dialog_id][turn_idx] = {
                    "positive_beliefs": turn.get("positives", turn.get("positive_beliefs", [])),
                    "negative_beliefs": turn.get("negatives", turn.get("negative_beliefs", [])),
                }
    elif isinstance(raw, dict):
        for did_str, turns in raw.items():
            dialog_id = int(did_str)
            result[dialog_id] = {}
            for tidx_str, beliefs in turns.items():
                turn_idx = int(tidx_str)
                result[dialog_id][turn_idx] = {
                    "positive_beliefs": beliefs.get("positive_beliefs", beliefs.get("positives", [])),
                    "negative_beliefs": beliefs.get("negative_beliefs", beliefs.get("negatives", [])),
                }
    else:
        raise ValueError(f"Unknown beliefs format in {path}")

    return result


class PrecomputedBeliefSource:
    """BeliefSource tra cứu từ dict {session_id: {turn_index: beliefs}} tính sẵn.

    `session_id` phải map trực tiếp sang key của dict (thường là index int
    của dialog trong dataset gốc). Với turn=0 (chưa có Q&A trước đó), NACIR++
    gốc luôn dùng beliefs của turn TRƯỚC ĐÓ (t-1) — hành vi này được giữ
    nguyên qua tham số `turn_offset`.
    """

    def __init__(self, beliefs: Dict[int, Dict[int, Dict]], turn_offset: int = -1):
        self.beliefs = beliefs
        self.turn_offset = turn_offset

    @classmethod
    def from_json(cls, path: str, turn_offset: int = -1) -> "PrecomputedBeliefSource":
        return cls(load_precomputed_beliefs(path), turn_offset=turn_offset)

    def get_beliefs(self, session_id: Any, turn_index: int, question: str, answer: str) -> BeliefBundle:
        lookup_turn = turn_index + self.turn_offset
        raw = self.beliefs.get(session_id, {}).get(lookup_turn, {})
        return BeliefBundle.from_raw(raw)


# ============================================================
# 2. Rule-based online (giữ nguyên các pattern gốc utils/negative_detector.py)
# ============================================================

NEGATIVE_PATTERNS = [
    r"\bno\b", r"\bnot\b", r"\bnope\b", r"\bnah\b",
    r"\bdon'?t\b", r"\bdoesn'?t\b", r"\bdidn'?t\b", r"\bcan'?t\b", r"\bcannot\b",
    r"\bwon'?t\b", r"\bisn'?t\b", r"\baren'?t\b", r"\bwasn'?t\b", r"\bweren'?t\b",
    r"\bhasn'?t\b", r"\bhaven'?t\b", r"\bcouldn'?t\b", r"\bwouldn'?t\b", r"\bshouldn'?t\b",
    r"\bwithout\b", r"\bnone\b", r"\bnever\b", r"\bneither\b", r"\bnobody\b",
    r"\bnothing\b", r"\bnowhere\b",
    r"i don'?t think", r"i don'?t see", r"i don'?t believe", r"not really",
    r"not that i", r"can'?t see", r"can'?t tell", r"hard to tell",
    r"doesn'?t look like", r"doesn'?t appear", r"doesn'?t seem", r"not sure",
    r"i wouldn'?t say",
]
_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in NEGATIVE_PATTERNS]


def is_negative_rule_based(answer: str) -> bool:
    answer = answer.lower().strip()
    return any(p.search(answer) for p in _COMPILED_PATTERNS)


def _extract_attribute(question: str, answer: str) -> str:
    """Trích thô 'attribute' từ câu hỏi (fallback đơn giản khi không có
    DualExtractor NLP đầy đủ). Lấy phần danh từ cuối câu hỏi kiểu
    "Is there a backpack?" -> "backpack"."""
    q = question.lower().strip().rstrip("?")
    for prefix in ("is there a ", "is there an ", "is there ", "are there ",
                   "does it have a ", "does it have ", "do you see a ", "do you see "):
        if q.startswith(prefix):
            return q[len(prefix):].strip()
    return q or answer.strip()


class RuleBasedBeliefSource:
    """
    BeliefSource online dùng Stage-1 rule-based negation detection (y hệt
    utils/negative_detector.py gốc) để tự gán positive/negative TỪ (question,
    answer) mà không cần file beliefs tính sẵn hay LLM. Phù hợp cho việc
    cắm NACIR++ vào một phương pháp/dataset hoàn toàn mới, miễn dữ liệu là
    dạng hội thoại Q&A tiếng Anh.

    Với các phương pháp có bộ trích xuất concept tốt hơn (LLM, NLI...), hãy
    thay bằng class riêng của bạn — chỉ cần cùng chữ ký `get_beliefs`.
    """

    def __init__(self, default_confidence: float = 0.7):
        self.default_confidence = default_confidence

    def get_beliefs(self, session_id: Any, turn_index: int, question: str, answer: str) -> BeliefBundle:
        if not answer:
            return BeliefBundle.empty()

        attribute = _extract_attribute(question, answer)
        if not attribute:
            return BeliefBundle.empty()

        if is_negative_rule_based(answer):
            return BeliefBundle(negative_beliefs=[Belief(attribute=attribute, confidence=self.default_confidence)])
        return BeliefBundle(positive_beliefs=[Belief(attribute=attribute, confidence=self.default_confidence)])
