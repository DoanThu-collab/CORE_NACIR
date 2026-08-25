"""Metric definitions matching the original PlugIR evaluation protocol."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
import torch


def _validated_rank_matrix(
    ranks_per_round: Sequence[Sequence[int]],
) -> torch.Tensor:
    if len(ranks_per_round) == 0:
        raise ValueError("ranks_per_round must not be empty")
    lengths = {len(round_ranks) for round_ranks in ranks_per_round}
    if len(lengths) != 1 or next(iter(lengths), 0) == 0:
        raise ValueError("Every round must contain the same non-zero number of ranks")
    raw = np.asarray(ranks_per_round)
    if raw.ndim != 2 or not np.issubdtype(raw.dtype, np.integer) or raw.dtype == np.bool_:
        raise ValueError("Ranks must be a rectangular integer matrix")
    ranks = torch.as_tensor(raw.astype(np.int64, copy=False))
    if bool((ranks < 0).any()):
        raise ValueError("Ranks must be zero-indexed non-negative integers")
    return ranks


def compute_metrics(
    ranks_per_round: Sequence[Sequence[int]],
    k: int = 10,
) -> Dict[str, object]:
    if k < 1:
        raise ValueError("k must be positive")
    ranks = _validated_rank_matrix(ranks_per_round)
    per_round_hit = ranks < k
    cumulative_hit = torch.cummax(per_round_hit.to(torch.int8), dim=0).values.bool()
    cumulative_hits = cumulative_hit.float().mean(dim=1) * 100.0
    per_round_recall = per_round_hit.float().mean(dim=1) * 100.0

    best_ranks = torch.cummin(ranks, dim=0).values.float()
    if ranks.shape[0] == 1:
        bri = torch.log(best_ranks[0] + 1.0).mean()
    else:
        trapezoids = (
            torch.log(best_ranks[:-1] + 1.0)
            + torch.log(best_ranks[1:] + 1.0)
        ) / 2.0
        bri = trapezoids.mean()

    return {
        "cumulative_hits": cumulative_hits,
        "per_round_recall": per_round_recall,
        "bri": float(bri.item()),
        "num_queries": int(ranks.shape[1]),
        "num_rounds": int(ranks.shape[0]),
        "rank_indexing": "zero",
        "log_base": "e",
        "bri_integration": "mean trapezoid over consecutive dialogue rounds",
    }


def paired_recall_delta_ci(
    baseline_ranks: Sequence[int],
    candidate_ranks: Sequence[int],
    *,
    k: int = 10,
    confidence: float = 0.95,
    samples: int = 10_000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    baseline = np.asarray(baseline_ranks)
    candidate = np.asarray(candidate_ranks)
    if (
        baseline.shape != candidate.shape
        or baseline.ndim != 1
        or baseline.size == 0
        or not np.issubdtype(baseline.dtype, np.integer)
        or not np.issubdtype(candidate.dtype, np.integer)
        or baseline.dtype == np.bool_
        or candidate.dtype == np.bool_
    ):
        raise ValueError("Paired ranks must be non-empty 1-D integer arrays with equal shape")
    if (baseline < 0).any() or (candidate < 0).any():
        raise ValueError("Paired ranks must be non-negative")
    if k < 1 or not 0 < confidence < 1:
        raise ValueError("k must be positive and confidence in (0, 1)")
    if samples < 100:
        raise ValueError("Use at least 100 bootstrap samples")
    differences = (
        (candidate < k).astype(np.float64)
        - (baseline < k).astype(np.float64)
    )
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(samples, dtype=np.float64)
    batch_size = 256
    for start in range(0, samples, batch_size):
        stop = min(samples, start + batch_size)
        indices = rng.integers(
            0,
            len(differences),
            size=(stop - start, len(differences)),
        )
        bootstrap[start:stop] = differences[indices].mean(axis=1) * 100.0
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(bootstrap, [alpha, 1.0 - alpha])
    return float(differences.mean() * 100.0), float(low), float(high)


def format_metrics(metrics: Dict[str, object], k: int = 10) -> str:
    lines = [f"====== Results for Hits@{k} ======"]
    for turn, value in enumerate(metrics["cumulative_hits"]):
        lines.append(f"\tDialog Length: {turn}: {float(value):.2f}%")
    lines.append(f"====== Results for Recall@{k} ======")
    for turn, value in enumerate(metrics["per_round_recall"]):
        lines.append(f"\tDialog Length: {turn}: {float(value):.2f}%")
    lines.append("====== Best log Rank Integral ======")
    lines.append(f"\tBRI: {metrics['bri']:.4f}")
    return "\n".join(lines)
