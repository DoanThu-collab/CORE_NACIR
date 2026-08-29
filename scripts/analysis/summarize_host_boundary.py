#!/usr/bin/env python3
"""Summarize host-boundary results from any existing PlugIR rank files.

Pass named runs explicitly when artifact discovery identifies them.
Example:
  python scripts/analysis/summarize_host_boundary.py \
    --run PlugIR_H0=runs_final/foo_h0/ranks.npz \
    --run PlugIR_Current=runs_final/foo_current/ranks.npz \
    --run PlugIR_Persistent=runs_final/foo_persistent/ranks.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from nacir.metrics import compute_metrics


def load(path: Path):
    r = np.load(path, allow_pickle=False)["ranks"].astype(np.int64)
    m = compute_metrics(r)
    per = [float(x) for x in m["per_round_recall"]]
    cum = [float(x) for x in m["cumulative_hits"]]
    return {
        "shape": r.shape,
        "avg_feedback_r10": float(np.mean(per[1:])),
        "final_r10": per[-1],
        "cum": cum[-1],
        "bri": float(m["bri"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=[], help="NAME=path/to/ranks.npz")
    args = ap.parse_args()

    rows = []
    for spec in args.run:
        if "=" not in spec:
            raise SystemExit(f"bad --run: {spec}")
        name, p = spec.split("=", 1)
        rows.append((name, load(Path(p))))

    print(f"{'Run':30s} {'shape':>12s} {'AvgR10':>9s} {'Final':>9s} {'Cum10':>9s} {'BRI':>9s}")
    print("-"*86)
    for name, m in rows:
        print(
            f"{name:30s} {str(m['shape']):>12s} "
            f"{m['avg_feedback_r10']:9.3f} {m['final_r10']:9.3f} "
            f"{m['cum']:9.3f} {m['bri']:9.4f}"
        )


if __name__ == "__main__":
    main()
