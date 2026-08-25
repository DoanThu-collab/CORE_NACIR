"""Weighted removal of query components aligned with negative concepts."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def orthogonal_remove(
    query: torch.Tensor,
    negative_vectors: Optional[torch.Tensor],
    negative_weights: Optional[torch.Tensor] = None,
    *,
    strength: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    if not 0 <= strength <= 1 or eps <= 0:
        raise ValueError("strength must be in [0, 1] and eps must be positive")
    if query.ndim != 1 or not torch.isfinite(query).all() or float(query.norm()) <= eps:
        raise ValueError("query must be a finite non-zero vector")
    normalized_query = F.normalize(query.float(), dim=0)
    if negative_vectors is None or negative_vectors.numel() == 0 or strength == 0:
        return normalized_query
    if negative_vectors.ndim != 2 or negative_vectors.shape[1] != query.numel():
        raise ValueError("negative_vectors must have shape [K, D]")
    if not torch.isfinite(negative_vectors).all() or bool(
        (negative_vectors.float().norm(dim=-1) <= eps).any()
    ):
        raise ValueError("negative_vectors must be finite and non-zero")

    vectors = F.normalize(negative_vectors.float(), dim=-1)
    if negative_weights is None:
        weights = torch.ones(
            vectors.shape[0], device=vectors.device, dtype=vectors.dtype
        )
    else:
        if negative_weights.shape != (vectors.shape[0],):
            raise ValueError("negative_weights must have shape [K]")
        if not torch.isfinite(negative_weights).all() or bool((negative_weights < 0).any()):
            raise ValueError("negative_weights must be finite and non-negative")
        weights = negative_weights.to(vectors)
    weight_sum = weights.sum()
    if float(weight_sum) <= eps:
        return normalized_query
    weights = weights / weight_sum

    # Confidence-weighted covariance removal. This intentionally keeps weights;
    # normalizing each weighted vector first would erase them.
    coefficients = vectors @ normalized_query
    component = (weights * coefficients) @ vectors
    updated = normalized_query - strength * component
    if float(updated.norm()) <= eps:
        return normalized_query
    return F.normalize(updated, dim=0)
