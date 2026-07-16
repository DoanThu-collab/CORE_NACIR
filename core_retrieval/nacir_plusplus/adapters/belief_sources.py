"""NACIR++ plug-and-play belief source adapters.

This module wraps the two original belief sources from `utils/negative_detector.py`
and `core/semantic_parser.py` into a standardized `BeliefSource` interface
without changing the internal logic.

     1. PrecomputedBeliefSource loads precomputed JSON beliefs (Format A/B,
         matching the original `semantic_parser.load_precomputed_beliefs`).
     2. RuleBasedBeliefSource uses `utils/negative_detector.py` (the original
         three-stage cascade, currently Stage 1 rule-based) to infer
         positive/negative beliefs directly from `(question, answer)` pairs. This
         is useful when another method does not provide a beliefs file.

Any other method that wants to provide beliefs in its own way (LLM, NLI,
internal NLU, etc.) only needs to implement a class with `get_beliefs(...)`.
No inheritance is required because the protocol relies on structural typing.
"""

import json
import re
from typing import Any, Dict, List, Optional

from ..schema import Belief, BeliefBundle

# ============================================================
# 1. Precomputed beliefs (preserves the original load_precomputed_beliefs logic)
# ============================================================


def load_precomputed_beliefs(path: str) -> Dict[int, Dict[int, Dict]]:
    """Preserve the original 1:1 logic from `core/semantic_parser.py::load_precomputed_beliefs`."""
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
    """Lookup belief bundles from a precomputed `{session_id: {turn_index: beliefs}}` dict.

    `session_id` must map directly to a dictionary key, usually the integer
    index of the dialog in the original dataset. For turn 0, the original
    NACIR++ behavior is to use the beliefs from the previous turn (`t-1`); that
    behavior is preserved through the `turn_offset` parameter.
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
# 2. Rule-based online (preserves the original `utils/negative_detector.py` patterns)
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
    """Heuristically extract an attribute from the question.

    This is a simple fallback when a full NLP DualExtractor is not available.
    It extracts the noun phrase at the end of questions such as
    "Is there a backpack?" -> "backpack".
    """
    q = question.lower().strip().rstrip("?")
    for prefix in ("is there a ", "is there an ", "is there ", "are there ",
                   "does it have a ", "does it have ", "do you see a ", "do you see "):
        if q.startswith(prefix):
            return q[len(prefix):].strip()
    return q or answer.strip()


class RuleBasedBeliefSource:
    """
    Online belief source that uses Stage 1 rule-based negation detection
    (matching the original `utils/negative_detector.py`) to assign positive or
    negative beliefs directly from `(question, answer)` pairs without a
    precomputed beliefs file or an LLM.

    This is suitable for plugging NACIR++ into a new method or dataset, as
    long as the data is conversational English Q&A.

    If your method has a better concept extractor (LLM, NLI, etc.), replace
    this with your own class that exposes the same `get_beliefs` signature.
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
