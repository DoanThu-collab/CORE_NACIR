#!/usr/bin/env python3
"""Diagnose why the fast weight-ablation evaluator is not exact.

Compares frozen canonical Persistent ranks with the fast evaluator's `full`
ranks, reports mismatch locations and whether any R@10 decision changes.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np


def load(path: Path):
    x = np.load(path, allow_pickle=False)["ranks"].astype(np.int64)
    if x.shape != (11, 2064):
        raise ValueError(f"{path}: {x.shape}")
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", type=Path, required=True)
    ap.add_argument("--fast", type=Path, required=True)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    a = load(args.frozen)
    b = load(args.fast)
    mask = a != b
    rr, dd = np.where(mask)

    print("=" * 88)
    print(args.name)
    print("=" * 88)
    print("exact:", bool(np.array_equal(a, b)))
    print("num different:", int(mask.sum()))
    print("max abs diff:", int(np.max(np.abs(a-b))) if mask.any() else 0)
    print("R@10 decision changes:", int(np.sum((a < 10) != (b < 10))))
    print("R@1 decision changes:", int(np.sum((a < 1) != (b < 1))))
    print("MRR-ish target rank zero changes:", int(np.sum((a == 0) != (b == 0))))
    print()
    print("first 30 mismatches:")
    for r, d in list(zip(rr, dd))[:30]:
        print({
            "dialog": int(d),
            "turn": int(r),
            "frozen_rank": int(a[r,d]),
            "fast_rank": int(b[r,d]),
            "delta": int(b[r,d]-a[r,d]),
            "frozen_hit10": bool(a[r,d] < 10),
            "fast_hit10": bool(b[r,d] < 10),
        })


if __name__ == "__main__":
    main()
