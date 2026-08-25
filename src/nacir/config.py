"""Configuration for the paper's F1 dual-route trust-fusion method.

This module intentionally exposes only the components evaluated in the paper:
signed memory, the H1 anchor, asymmetric proposal/constraint routing, and fusion.
"""

from __future__ import annotations

import math
import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


def _build_dataclass(cls: type[T], values: dict[str, Any]) -> T:
    if not is_dataclass(cls):
        raise TypeError("cls must be a dataclass type")
    kwargs: dict[str, Any] = {}
    for item in fields(cls):
        if item.name not in values:
            continue
        current = getattr(cls(), item.name)
        raw_value = values[item.name]
        if is_dataclass(current) and isinstance(raw_value, dict):
            kwargs[item.name] = _build_dataclass(type(current), raw_value)
        else:
            kwargs[item.name] = raw_value
    return cls(**kwargs)


def _finite(name: str, *values: float) -> None:
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError(f"{name} values must be finite")


@dataclass
class MemoryConfig:
    positive_weight: float = 0.55
    negative_weight: float = 0.275
    recency_decay: float = 0.1
    semantic_merge: bool = False
    semantic_merge_threshold: float = 0.92
    override_boost: float = 0.15
    retain_history: bool = True
    max_concepts: int = 50

    def validate(self) -> None:
        _finite(
            "memory",
            self.positive_weight,
            self.negative_weight,
            self.recency_decay,
            self.semantic_merge_threshold,
            self.override_boost,
        )
        if self.positive_weight < 0 or self.negative_weight < 0:
            raise ValueError("memory weights must be non-negative")
        if self.recency_decay < 0:
            raise ValueError("recency_decay must be non-negative")
        if not 0 <= self.semantic_merge_threshold <= 1:
            raise ValueError("semantic_merge_threshold must be in [0, 1]")
        if not 0 <= self.override_boost <= 1:
            raise ValueError("override_boost must be in [0, 1]")
        if self.max_concepts < 1:
            raise ValueError("max_concepts must be positive")
        if not self.retain_history:
            raise ValueError("the paper method requires persistent memory")


@dataclass
class ProjectionConfig:
    enabled: bool = True
    strength: float = 0.2

    def validate(self) -> None:
        _finite("projection", self.strength)
        if not self.enabled or not 0 <= self.strength <= 1:
            raise ValueError("F1 requires enabled projection with strength in [0, 1]")


@dataclass
class MaskingConfig:
    enabled: bool = True
    threshold: float = 0.25
    max_penalty: float = 0.18
    temperature: float = 0.1

    def validate(self) -> None:
        _finite("masking", self.threshold, self.max_penalty, self.temperature)
        if not self.enabled:
            raise ValueError("F1 requires enabled negative masking")
        if not -1 <= self.threshold <= 1:
            raise ValueError("masking threshold must be in [-1, 1]")
        if not 0 <= self.max_penalty <= 1 or self.temperature <= 0:
            raise ValueError("masking penalty or temperature is invalid")


@dataclass
class AsymmetricConstraintConfig:
    mode: str = "dual_route_trust"
    candidate_k: int = 500
    negative_strength: float = 0.275
    posterior_temperature: float = 0.05
    min_spread: float = 0.01
    max_kl: float = 0.002
    eps: float = 1e-6

    def validate(self) -> None:
        if self.mode != "dual_route_trust":
            raise ValueError("the paper release supports only dual_route_trust")
        if self.candidate_k < 2:
            raise ValueError("candidate_k must be at least 2")
        _finite(
            "asymmetric constraint",
            self.negative_strength,
            self.posterior_temperature,
            self.min_spread,
            self.max_kl,
            self.eps,
        )
        if self.negative_strength < 0 or self.posterior_temperature <= 0:
            raise ValueError("constraint strength or temperature is invalid")
        if self.min_spread <= 0 or self.max_kl <= 0 or self.eps <= 0:
            raise ValueError("spread, KL budget, and epsilon must be positive")


@dataclass
class F1Config:
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    asymmetric_constraint: AsymmetricConstraintConfig = field(
        default_factory=AsymmetricConstraintConfig
    )
    top_k: int = 10

    def validate(self) -> None:
        self.memory.validate()
        self.projection.validate()
        self.masking.validate()
        self.asymmetric_constraint.validate()
        if self.top_k < 1:
            raise ValueError("top_k must be positive")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "F1Config":
        config = _build_dataclass(cls, values)
        config.validate()
        return config

    @classmethod
    def from_path(cls, path: str | Path) -> "F1Config":
        with Path(path).open(encoding="utf-8") as handle:
            values = json.load(handle)
        if not isinstance(values, dict):
            raise ValueError("configuration file must contain a JSON object")
        return cls.from_dict(values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
