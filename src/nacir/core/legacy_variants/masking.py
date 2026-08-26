"""Corpus-level penalty for images aligned with negative concepts."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def apply_negative_penalty(
    scores: torch.Tensor,
    corpus_vectors: torch.Tensor,
    negative_vectors: Optional[torch.Tensor],
    negative_weights: Optional[torch.Tensor],
    *,
    threshold: float,
    max_penalty: float,
    temperature: float,
) -> torch.Tensor:
    if scores.ndim != 1 or corpus_vectors.ndim != 2:
        raise ValueError("scores must be [N] and corpus_vectors must be [N, D]")
    if scores.shape[0] != corpus_vectors.shape[0] or corpus_vectors.shape[1] < 1:
        raise ValueError("scores and corpus_vectors must align on N")
    if not -1 <= threshold <= 1 or not 0 <= max_penalty <= 1 or temperature <= 0:
        raise ValueError("invalid masking threshold, max_penalty, or temperature")
    if not torch.isfinite(scores).all() or not torch.isfinite(corpus_vectors).all():
        raise ValueError("scores and corpus_vectors must be finite")
    if negative_vectors is None or negative_vectors.numel() == 0 or max_penalty == 0:
        return scores
    if negative_vectors.ndim != 2 or negative_vectors.shape[1] != corpus_vectors.shape[1]:
        raise ValueError("negative_vectors must have shape [K, D]")
    if not torch.isfinite(negative_vectors).all() or bool(
        (negative_vectors.float().norm(dim=-1) <= 1e-8).any()
    ):
        raise ValueError("negative_vectors must be finite and non-zero")

    if negative_weights is None:
        weights = torch.ones(
            negative_vectors.shape[0], device=scores.device, dtype=scores.dtype
        )
    else:
        if negative_weights.shape != (negative_vectors.shape[0],):
            raise ValueError("negative_weights must have shape [K]")
        if not torch.isfinite(negative_weights).all() or bool((negative_weights < 0).any()):
            raise ValueError("negative_weights must be finite and non-negative")
        weights = negative_weights.to(device=scores.device, dtype=scores.dtype)
    weight_sum = weights.sum()
    if float(weight_sum) <= 1e-8:
        return scores
    weights = weights / weight_sum
    normalized_negative = F.normalize(negative_vectors.float(), dim=-1)
    normalized_corpus = F.normalize(corpus_vectors.float(), dim=-1)
    similarity = normalized_negative @ normalized_corpus.T
    concept_activation = torch.sigmoid((similarity - threshold) / temperature)
    # max_penalty is the maximum amplitude, not a clamp on raw sigmoid values.
    penalty = max_penalty * (weights[:, None] * concept_activation).sum(dim=0).clamp(0, 1)
    return scores - penalty.to(scores)
