"""Typed data contracts shared by training, inference, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Mapping, Optional

import torch


@dataclass(frozen=True)
class Belief:
    attribute: str
    confidence: float = 0.7
    fact_type: str = "unknown"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.attribute, str):
            raise ValueError("Belief.attribute must be a string")
        attribute = self.attribute.strip()
        if not attribute:
            raise ValueError("Belief.attribute must be non-empty")
        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not math.isfinite(float(self.confidence))
            or not 0.0 <= float(self.confidence) <= 1.0
        ):
            raise ValueError("Belief.confidence must be finite and in [0, 1]")
        if not isinstance(self.fact_type, str) or not self.fact_type.strip():
            raise ValueError("Belief.fact_type must be a non-empty string")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("Belief.metadata must be a mapping")
        object.__setattr__(self, "attribute", attribute)
        object.__setattr__(self, "confidence", float(self.confidence))


@dataclass(frozen=True)
class BeliefBundle:
    positive: List[Belief] = field(default_factory=list)
    negative: List[Belief] = field(default_factory=list)
    source_turn: Optional[int] = None
    question: str = ""
    answer: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.positive, list)
            or not isinstance(self.negative, list)
            or any(not isinstance(item, Belief) for item in self.positive + self.negative)
        ):
            raise ValueError("BeliefBundle polarities must be lists of Belief")
        if (
            self.source_turn is not None
            and (
                isinstance(self.source_turn, bool)
                or not isinstance(self.source_turn, int)
                or self.source_turn < 0
            )
        ):
            raise ValueError("BeliefBundle.source_turn must be a non-negative integer")
        if not isinstance(self.question, str) or not isinstance(self.answer, str):
            raise ValueError("BeliefBundle question/answer must be strings")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("BeliefBundle.metadata must be a mapping")

    @classmethod
    def empty(cls) -> "BeliefBundle":
        return cls()

    @property
    def total(self) -> int:
        return len(self.positive) + len(self.negative)

    @property
    def negative_fraction(self) -> float:
        return len(self.negative) / self.total if self.total else 0.0


@dataclass(frozen=True)
class DialogTurn:
    turn_index: int
    query_text: str
    query_vector: Optional[torch.Tensor] = None
    beliefs: Optional[BeliefBundle] = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.turn_index, bool)
            or not isinstance(self.turn_index, int)
            or self.turn_index < 0
        ):
            raise ValueError("turn_index must be a non-negative integer")
        if not isinstance(self.query_text, str):
            raise ValueError("query_text must be a string")
        if self.query_vector is not None and not isinstance(self.query_vector, torch.Tensor):
            raise ValueError("query_vector must be a tensor")
        if self.beliefs is not None and not isinstance(self.beliefs, BeliefBundle):
            raise ValueError("beliefs must be a BeliefBundle")
        if self.query_vector is None and not self.query_text.strip():
            raise ValueError("A turn needs query_text or query_vector")
        if self.query_vector is not None and self.query_vector.ndim != 1:
            raise ValueError("query_vector must have shape [D]")
        if self.query_vector is not None and (
            not torch.isfinite(self.query_vector).all() or float(self.query_vector.norm()) <= 1e-8
        ):
            raise ValueError("query_vector must be finite and non-zero")


@dataclass(frozen=True)
class RetrievalSession:
    session_id: Any
    turns: List[DialogTurn]
    target_index: Optional[int] = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.turns, list)
            or any(not isinstance(turn, DialogTurn) for turn in self.turns)
        ):
            raise ValueError("RetrievalSession.turns must be a list of DialogTurn")
        if (
            self.target_index is not None
            and (
                isinstance(self.target_index, bool)
                or not isinstance(self.target_index, int)
            )
        ):
            raise ValueError("target_index must be a non-negative integer")
        indices = [turn.turn_index for turn in self.turns]
        if not indices:
            raise ValueError("RetrievalSession must contain at least one turn")
        if sorted(indices) != list(range(len(indices))):
            raise ValueError("turn_index values must be contiguous and start at zero")
        if self.target_index is not None and self.target_index < 0:
            raise ValueError("target_index must be non-negative")


@dataclass
class TurnTrace:
    turn_index: int
    accepted: bool
    decision_mode: str
    features: Dict[str, float] = field(default_factory=dict)
    reject_rank: Optional[int] = None
    accept_rank: Optional[int] = None
    final_rank: Optional[int] = None
    top_k_indices: List[int] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def oracle_delta_rank(self) -> Optional[int]:
        if self.reject_rank is None or self.accept_rank is None:
            return None
        return self.reject_rank - self.accept_rank


@dataclass
class SessionOutput:
    session_id: Any
    turns: List[TurnTrace]

    def target_ranks(self) -> List[Optional[int]]:
        return [turn.final_rank for turn in self.turns]

