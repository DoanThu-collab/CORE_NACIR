"""Configuration schemas for the canonical NACIR method."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class MemoryConfig:
    positive_weight: float = 0.55
    negative_weight: float = 0.275
    recency_decay: float = 0.10
    override_boost: float = 0.15
    max_concepts: int = 50
    semantic_merge: bool = False
    semantic_merge_threshold: float = 0.85

    def validate(self) -> None:
        if self.negative_weight < 0:
            raise ValueError("negative_weight cannot be negative")
        if self.recency_decay < 0:
            raise ValueError("recency_decay must be non-negative")
        if self.override_boost < 0:
            raise ValueError("override_boost must be non-negative")
        if self.max_concepts < 1:
            raise ValueError("max_concepts must be at least 1")


@dataclasses.dataclass
class NACIRMinusConfig:
    memory: MemoryConfig = dataclasses.field(default_factory=MemoryConfig)
    top_k: int = 1000

    def validate(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")
        self.memory.validate()
