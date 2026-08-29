"""Conversation-scoped concept memory with explicit polarity overrides."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from ..config import MemoryConfig
from ..schema import Belief, BeliefBundle


def canonicalize_concept(text: str) -> str:
    return " ".join(text.lower().strip().split())


@dataclass
class MemoryEntry:
    name: str
    polarity: str
    vector: torch.Tensor
    confidence: float
    turn_created: int
    turn_updated: int
    override_count: int = 0


class ConceptMemory:
    def __init__(self, config: MemoryConfig, encoder) -> None:
        config.validate()
        self.config = copy.deepcopy(config)
        self.encoder = encoder
        self.entries: Dict[str, MemoryEntry] = {}
        self.current_turn = 0
        self.override_log: List[dict] = []

    def clone(self) -> "ConceptMemory":
        cloned = ConceptMemory(self.config, self.encoder)
        cloned.entries = copy.deepcopy(self.entries)
        cloned.current_turn = self.current_turn
        cloned.override_log = copy.deepcopy(self.override_log)
        return cloned

    @property
    def negative_fraction(self) -> float:
        if not self.entries:
            return 0.0
        return sum(entry.polarity == "negative" for entry in self.entries.values()) / len(self.entries)

    def _match(self, name: str, vector: torch.Tensor) -> Optional[str]:
        key = canonicalize_concept(name)
        if key in self.entries:
            return key
        if not self.config.semantic_merge:
            return None
        if not self.entries:
            return None
        keys = list(self.entries)
        existing = torch.stack([self.entries[item].vector for item in keys])
        similarities = existing @ vector
        best_value, best_index = similarities.max(dim=0)
        return (
            keys[int(best_index)]
            if float(best_value) >= self.config.semantic_merge_threshold
            else None
        )

    def has_polarity_conflict(self, bundle: BeliefBundle) -> bool:
        for polarity, beliefs in (("positive", bundle.positive), ("negative", bundle.negative)):
            for belief in beliefs:
                key = canonicalize_concept(belief.attribute)
                entry = self.entries.get(key)
                if entry is not None and entry.polarity != polarity:
                    return True
        return False

    def add_bundle(self, bundle: BeliefBundle, turn: int) -> Dict[str, int]:
        self.current_turn = turn

        # Canonical NACIR processes negative beliefs only.
        pairs = [("negative", belief) for belief in bundle.negative]

        stats = {"added": 0, "updated": 0, "overridden": 0, "evicted": 0}
        if not pairs:
            return stats

        vectors = self.encoder.encode([belief.attribute for _, belief in pairs])
        if vectors.ndim != 2 or vectors.shape[0] != len(pairs) or vectors.shape[1] < 1:
            raise ValueError("Text encoder returned an invalid concept embedding batch")
        if not torch.isfinite(vectors).all():
            raise ValueError("Text encoder returned non-finite concept embeddings")
        if self.entries and vectors.shape[1] != next(iter(self.entries.values())).vector.numel():
            raise ValueError("Concept embedding dimension changed within a session")
        vectors = F.normalize(vectors.float(), dim=-1)

        for (polarity, belief), vector in zip(pairs, vectors):
            matched = self._match(belief.attribute, vector)
            if matched is None:
                key = canonicalize_concept(belief.attribute)
                self.entries[key] = MemoryEntry(
                    name=belief.attribute,
                    polarity=polarity,
                    vector=vector,
                    confidence=belief.confidence,
                    turn_created=turn,
                    turn_updated=turn,
                )
                stats["added"] += 1
                continue

            entry = self.entries[matched]
            if entry.polarity == polarity:
                entry.confidence = max(entry.confidence, belief.confidence)
                entry.vector = vector
                entry.turn_updated = turn
                stats["updated"] += 1
            else:
                old_polarity = entry.polarity
                entry.polarity = polarity
                entry.confidence = min(1.0, max(entry.confidence, belief.confidence) + self.config.override_boost)
                entry.vector = vector
                entry.turn_updated = turn
                entry.override_count += 1
                self.override_log.append(
                    {
                        "concept": entry.name,
                        "old_polarity": old_polarity,
                        "new_polarity": polarity,
                        "turn": turn,
                    }
                )
                stats["overridden"] += 1

        if len(self.entries) > self.config.max_concepts:
            ordered = sorted(self.entries, key=lambda key: self.entries[key].turn_updated)
            for key in ordered[: len(self.entries) - self.config.max_concepts]:
                del self.entries[key]
                stats["evicted"] += 1
        return stats

    def vectors(self, polarity: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        selected = [entry for entry in self.entries.values() if entry.polarity == polarity]
        if not selected:
            return None, None
        vectors = torch.stack([entry.vector for entry in selected])
        weights = []
        for entry in selected:
            age = max(0, self.current_turn - entry.turn_updated)
            weights.append(entry.confidence / (1.0 + self.config.recency_decay * age))
        return vectors, torch.tensor(weights, dtype=vectors.dtype, device=vectors.device)

    def synthesize(
        self,
        query: torch.Tensor,
        *,
        positive: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        negative: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """q^- = norm(q_bar - lambda * norm(sum_j w_j v_j))."""
        negative = negative if negative is not None else self.vectors("negative")
        updated = query.clone().float()

        vectors, confidences = negative
        if vectors is not None and confidences is not None and vectors.numel() > 0:
            aggregate = (confidences[:, None] * vectors).sum(dim=0)
            if float(aggregate.norm()) > 0:
                updated = updated - self.config.negative_weight * F.normalize(aggregate, dim=0)

        return F.normalize(updated, dim=0)

    def diagnostics(self) -> dict:
        return {
            "size": len(self.entries),
            "positive": sum(entry.polarity == "positive" for entry in self.entries.values()),
            "negative": sum(entry.polarity == "negative" for entry in self.entries.values()),
            "overrides": len(self.override_log),
        }
