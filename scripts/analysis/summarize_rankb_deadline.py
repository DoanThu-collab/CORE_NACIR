#!/usr/bin/env python3
"""Summarize deadline experiments into one compact JSON/table."""

from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
from nacir.metrics import compute_metrics


def metrics_from_ranks(path: Path):
    r = np.load(path, allow_pickle=False)["ranks"].astype(np.int64)
    m = compute_metrics(r)
    per = [float(x) for x in m["per_round_recall"]]
    cum = [float(x) for x in m["cumulative_hits"]]
    return {
        "avg_feedback_r10": float(np.mean(per[1:])),
        "final_r10": per[-1],
        "cumulative_hits10": cum[-1],
        "bri": float(m["bri"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=Path("artifacts_final/analysis/rankb_deadline_summary.json"))
    args = ap.parse_args()
    R = args.root

    candidates = {
        "BLIP_H0": R/"runs_final/chatir_blip_h0/ranks.npz",
        "BLIP_Current": R/"runs_final/chatir_blip_nacir_current_turn/ranks.npz",
        "BLIP_NACIR": R/"runs_final/chatir_blip_nacir_minus/ranks.npz",
        "CLIP_H0": R/"runs_final/chatir_clip_vitl14_h0/ranks.npz",
        "CLIP_Current": R/"runs_final/chatir_clip_vitl14_nacir_current_turn/ranks.npz",
        "CLIP_NACIR": R/"runs_final/chatir_clip_vitl14_nacir_minus/ranks.npz",
    }

    # Optional newly generated variants.
    for backbone in ["blip", "clip"]:
        d = R/f"runs_deadline/{backbone}_weight_ablation"
        for mode in ["uniform", "confidence", "recency", "full"]:
            candidates[f"{backbone.upper()}_{mode}"] = d/f"{mode}_ranks.npz"
        candidates[f"{backbone.upper()}_TextPersistent"] = (
            R/f"runs_deadline/{backbone}_text_persistent/ranks.npz"
        )

    out = {}
    for name, path in candidates.items():
        if path.exists():
            out[name] = metrics_from_ranks(path)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"{'Run':28s} {'AvgR10':>9s} {'FinalR10':>9s} {'Cum10':>9s} {'BRI':>9s}")
    print("-"*70)
    for name, m in out.items():
        print(
            f"{name:28s} {m['avg_feedback_r10']:9.3f} "
            f"{m['final_r10']:9.3f} {m['cumulative_hits10']:9.3f} "
            f"{m['bri']:9.4f}"
        )
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
