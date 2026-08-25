#!/usr/bin/env python3
"""Compute paired BRI and per-round Recall@10 comparisons from release ranks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from nacir.metrics import compute_metrics, paired_recall_delta_ci
from nacir.statistics import exact_mcnemar_pvalue, holm_adjust, paired_bri_delta_ci


def _ranks(run: Path) -> np.ndarray:
    archive = np.load(run / "ranks.npz")
    ranks = archive["ranks_per_round"]
    if ranks.ndim != 2 or ranks.size == 0 or (ranks < 0).any():
        raise ValueError(f"invalid ranks in {run}")
    return ranks.astype(np.int64, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline, candidate = _ranks(args.baseline), _ranks(args.candidate)
    if baseline.shape != candidate.shape:
        raise ValueError("baseline and candidate ranks must align")
    delta, low, high = paired_bri_delta_ci(baseline, candidate)
    raw_pvalues = [exact_mcnemar_pvalue(baseline[turn], candidate[turn], k=10)["pvalue_exact_two_sided"] for turn in range(baseline.shape[0])]
    holm = holm_adjust(raw_pvalues)
    rounds = []
    for turn in range(baseline.shape[0]):
        recall_delta, recall_low, recall_high = paired_recall_delta_ci(baseline[turn], candidate[turn])
        rounds.append({
            "turn": turn,
            "baseline_recall_at_10": float(compute_metrics([baseline[turn]])["per_round_recall"][0]),
            "candidate_recall_at_10": float(compute_metrics([candidate[turn]])["per_round_recall"][0]),
            "candidate_minus_baseline_recall_at_10": recall_delta,
            "confidence_interval": [recall_low, recall_high],
            "mcnemar": exact_mcnemar_pvalue(baseline[turn], candidate[turn], k=10),
            "holm_adjusted_pvalue": holm[turn],
        })
    report = {
        "status": "complete",
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "bri": {
            "baseline": compute_metrics(baseline)["bri"],
            "candidate": compute_metrics(candidate)["bri"],
            "candidate_minus_baseline": delta,
            "confidence_interval": [low, high],
            "direction": "lower_is_better",
        },
        "rounds": rounds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)


if __name__ == "__main__":
    main()
