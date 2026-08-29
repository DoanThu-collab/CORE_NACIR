#!/usr/bin/env python3
"""Strict paired comparison guard for NACIR rank archives.

The old comparator only checked shape / session-target identity and could accept
BLIP-vs-CLIP or different-corpus runs as "paired". This script refuses such
comparisons unless strict provenance matches.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from nacir.provenance import read_rank_archive, verify_pairing


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline", type=Path, help="ranks.npz or run directory")
    ap.add_argument("candidate", type=Path, help="ranks.npz or run directory")
    ap.add_argument("--output", type=Path)
    ap.add_argument(
        "--allow-legacy-unverified",
        action="store_true",
        help="Debug only. Never use for paper paired statistics.",
    )
    args = ap.parse_args()

    def resolve(p: Path) -> Path:
        return p / "ranks.npz" if p.is_dir() else p

    a = read_rank_archive(resolve(args.baseline))
    b = read_rank_archive(resolve(args.candidate))
    verdict = verify_pairing(a, b)

    if not verdict["verified"] and not args.allow_legacy_unverified:
        print("compare_runs verdict: REJECTED")
        print(json.dumps(verdict, indent=2))
        raise SystemExit(2)

    result = {
        "status": "verified" if verdict["verified"] else "legacy_unverified",
        "pairing_provenance": verdict,
        "baseline": str(resolve(args.baseline)),
        "candidate": str(resolve(args.candidate)),
        "rank_shape": list(a["ranks"].shape),
        "num_rank_differences": int(np.sum(a["ranks"] != b["ranks"])),
        "max_abs_rank_difference": int(np.max(np.abs(a["ranks"] - b["ranks"]))) if a["ranks"].size else 0,
    }
    print("compare_runs verdict:", result["status"])
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
