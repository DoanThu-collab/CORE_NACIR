#!/usr/bin/env python3
"""Compute paired NACIR statistics only after strict provenance verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from nacir.metrics import compute_metrics, paired_recall_delta_ci
from nacir.provenance import read_rank_archive, verify_pairing
from nacir.statistics import exact_mcnemar_pvalue, holm_adjust, paired_bri_delta_ci


def _resolve(path: Path) -> Path:
    return path / "ranks.npz" if path.is_dir() else path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-legacy-unverified",
        action="store_true",
        help="Debug only; never use for paper paired statistics.",
    )
    args = parser.parse_args()

    baseline_archive = read_rank_archive(_resolve(args.baseline))
    candidate_archive = read_rank_archive(_resolve(args.candidate))
    pairing = verify_pairing(baseline_archive, candidate_archive)

    if not pairing["verified"] and not args.allow_legacy_unverified:
        raise ValueError(
            "paired comparison rejected by provenance guard: "
            + ", ".join(pairing["reasons"])
        )

    baseline = baseline_archive["ranks"]
    candidate = candidate_archive["ranks"]

    if baseline.ndim != 2 or candidate.ndim != 2:
        raise ValueError("rank matrices must be 2D")
    if baseline.size == 0 or candidate.size == 0:
        raise ValueError("rank matrices must be non-empty")
    if (baseline < 0).any() or (candidate < 0).any():
        raise ValueError("ranks must be non-negative")
    if baseline.shape != candidate.shape:
        raise ValueError("baseline and candidate ranks must align")

    delta, low, high = paired_bri_delta_ci(baseline, candidate)
    raw_pvalues = [
        exact_mcnemar_pvalue(
            baseline[turn],
            candidate[turn],
            k=10,
        )["pvalue_exact_two_sided"]
        for turn in range(baseline.shape[0])
    ]
    holm = holm_adjust(raw_pvalues)

    rounds = []
    for turn in range(baseline.shape[0]):
        recall_delta, recall_low, recall_high = paired_recall_delta_ci(
            baseline[turn],
            candidate[turn],
        )
        rounds.append(
            {
                "turn": turn,
                "baseline_recall_at_10": float(
                    compute_metrics([baseline[turn]])["per_round_recall"][0]
                ),
                "candidate_recall_at_10": float(
                    compute_metrics([candidate[turn]])["per_round_recall"][0]
                ),
                "candidate_minus_baseline_recall_at_10": recall_delta,
                "confidence_interval": [recall_low, recall_high],
                "mcnemar": exact_mcnemar_pvalue(
                    baseline[turn],
                    candidate[turn],
                    k=10,
                ),
                "holm_adjusted_pvalue": holm[turn],
            }
        )

    report = {
        "status": "verified" if pairing["verified"] else "legacy_unverified",
        "pairing_provenance": pairing,
        "baseline": str(_resolve(args.baseline)),
        "candidate": str(_resolve(args.candidate)),
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

    print("compare_runs verdict:", report["status"])


if __name__ == "__main__":
    main()
