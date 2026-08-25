"""Paired uncertainty estimates and multiple-testing correction for retrieval runs."""

from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import numpy as np


def _paired_integer_ranks(
    baseline: Sequence[int] | np.ndarray,
    candidate: Sequence[int] | np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    first = np.asarray(baseline)
    second = np.asarray(candidate)
    if (
        first.shape != second.shape
        or first.ndim < 1
        or first.size == 0
        or not np.issubdtype(first.dtype, np.integer)
        or not np.issubdtype(second.dtype, np.integer)
        or first.dtype == np.bool_
        or second.dtype == np.bool_
        or (first < 0).any()
        or (second < 0).any()
    ):
        raise ValueError("Paired ranks must be aligned non-negative integer arrays")
    return first.astype(np.int64, copy=False), second.astype(np.int64, copy=False)


def per_session_bri(ranks_per_round: Sequence[Sequence[int]] | np.ndarray) -> np.ndarray:
    ranks = np.asarray(ranks_per_round)
    if (
        ranks.ndim != 2
        or ranks.size == 0
        or not np.issubdtype(ranks.dtype, np.integer)
        or ranks.dtype == np.bool_
        or (ranks < 0).any()
    ):
        raise ValueError("BRI ranks must be a non-empty [round, session] integer matrix")
    best = np.minimum.accumulate(ranks.astype(np.float64), axis=0)
    logged = np.log(best + 1.0)
    if ranks.shape[0] == 1:
        return logged[0]
    return ((logged[:-1] + logged[1:]) / 2.0).mean(axis=0)


def paired_bri_delta_ci(
    baseline: Sequence[Sequence[int]] | np.ndarray,
    candidate: Sequence[Sequence[int]] | np.ndarray,
    *,
    confidence: float = 0.95,
    samples: int = 10_000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    first, second = _paired_integer_ranks(baseline, candidate)
    if first.ndim != 2:
        raise ValueError("Paired BRI requires [round, session] matrices")
    if not 0 < confidence < 1 or samples < 100:
        raise ValueError("confidence must be in (0,1) and samples at least 100")
    differences = per_session_bri(second) - per_session_bri(first)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 256):
        stop = min(samples, start + 256)
        indices = rng.integers(
            0, len(differences), size=(stop - start, len(differences))
        )
        bootstrap[start:stop] = differences[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(bootstrap, [alpha, 1.0 - alpha])
    return float(differences.mean()), float(low), float(high)


def exact_mcnemar_pvalue(
    baseline_ranks: Sequence[int] | np.ndarray,
    candidate_ranks: Sequence[int] | np.ndarray,
    *,
    k: int,
) -> Dict[str, float | int]:
    baseline, candidate = _paired_integer_ranks(baseline_ranks, candidate_ranks)
    if baseline.ndim != 1 or k < 1:
        raise ValueError("McNemar input must be 1-D and k positive")
    first_hit = baseline < k
    second_hit = candidate < k
    lost = int((first_hit & ~second_hit).sum())
    gained = int((~first_hit & second_hit).sum())
    discordant = lost + gained
    if discordant == 0:
        pvalue = 1.0
    else:
        tail = sum(
            math.comb(discordant, value)
            for value in range(0, min(lost, gained) + 1)
        ) / (2**discordant)
        pvalue = min(1.0, 2.0 * tail)
    return {
        "baseline_only_hits": lost,
        "candidate_only_hits": gained,
        "discordant": discordant,
        "pvalue_exact_two_sided": float(pvalue),
    }


def holm_adjust(pvalues: Sequence[float]) -> list[float]:
    values = np.asarray(pvalues, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("pvalues must be a finite non-empty vector")
    if ((values < 0) | (values > 1)).any():
        raise ValueError("pvalues must be in [0, 1]")
    order = np.argsort(values, kind="stable")
    adjusted_sorted = np.empty(len(values), dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (len(values) - rank) * values[index])
        running = max(running, candidate)
        adjusted_sorted[rank] = running
    adjusted = np.empty(len(values), dtype=np.float64)
    adjusted[order] = adjusted_sorted
    return adjusted.tolist()
