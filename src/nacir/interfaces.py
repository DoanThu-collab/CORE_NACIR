"""Structural interfaces used by the core pipeline."""

from __future__ import annotations

from typing import Any, List, Protocol, Sequence, runtime_checkable

import torch

from .schema import BeliefBundle


@runtime_checkable
class TextEncoder(Protocol):
    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        """Return normalized embeddings with shape [len(texts), D]."""


@runtime_checkable
class BeliefSource(Protocol):
    def get_beliefs(self, session_id: Any, turn_index: int) -> BeliefBundle:
        """Return feedback attached to retrieval turn ``turn_index``."""


@runtime_checkable
class ImageScorer(Protocol):
    def score(self, query_text: str, image_refs: List[Any]) -> torch.Tensor:
        """Return one calibrated match score per image reference."""


class NullImageScorer:
    def score(self, query_text: str, image_refs: List[Any]) -> torch.Tensor:
        return torch.zeros(len(image_refs), dtype=torch.float32)

