from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

BELIEFS = Path(
    "/mlcv1/WorkingSpace/Personal/core_baotg/thuy/"
    "NACIR_FIX/data/beliefs_v2/"
    "llama3_1_8b_v9_final_20260824.json"
)

BLIP_H0 = ROOT / "runs_final/chatir_blip_h0/ranks.npz"
BLIP_NACIR = ROOT / "runs_final/chatir_blip_nacir_minus/ranks.npz"

CLIP_H0 = ROOT / "runs_final/chatir_clip_vitl14_h0/ranks.npz"
CLIP_NACIR = ROOT / "runs_final/chatir_clip_vitl14_nacir_minus/ranks.npz"

OUT = ROOT / "artifacts_final/analysis"


def norm_attr(x: str) -> str:
    x = x.lower().strip()
    x = re.sub(r"[^a-z0-9\s-]", " ", x)
    x = re.sub(r"\s+", " ", x)
    return x


def load_ranks(path: Path):
    x = np.load(path)["ranks"]
    if x.shape[0] == 11:
        x = x.T
    assert x.shape == (2064, 11), (path, x.shape)
    return x


data = json.load(open(BELIEFS, encoding="utf-8"))
dialogs = data["dialogs"]

assert len(dialogs) == 2064


# ------------------------------------------------------
# 1. Basic belief statistics
# ------------------------------------------------------

num_neg = []
num_pos = []
num_neg_highconf = []
unique_neg = []
turn_neg_counts = np.zeros(10, dtype=int)
turn_pos_counts = np.zeros(10, dtype=int)

all_neg_attrs = Counter()
all_pos_attrs = Counter()

per_dialog = []

for d in dialogs:
    negs = []
    poss = []

    for turn in d["turns"]:
        t = turn["turn"]

        turn_neg_counts[t] += len(turn["negatives"])
        turn_pos_counts[t] += len(turn["positives"])

        for b in turn["negatives"]:
            attr = norm_attr(b["attribute"])
            conf = float(b.get("confidence", 1.0))

            negs.append((t, attr, conf))
            all_neg_attrs[attr] += 1

        for b in turn["positives"]:
            attr = norm_attr(b["attribute"])
            conf = float(b.get("confidence", 1.0))

            poss.append((t, attr, conf))
            all_pos_attrs[attr] += 1

    num_neg.append(len(negs))
    num_pos.append(len(poss))
    num_neg_highconf.append(sum(c >= 0.7 for _, _, c in negs))
    unique_neg.append(len(set(a for _, a, _ in negs)))

    per_dialog.append({
        "dialog_id": d["dialog_id"],
        "num_negative": len(negs),
        "num_positive": len(poss),
        "num_negative_highconf": sum(c >= 0.7 for _, _, c in negs),
        "num_unique_negative": len(set(a for _, a, _ in negs)),
    })


# ------------------------------------------------------
# 2. Candidate contradiction / reversal detection
#
# Conservative lexical detector:
# same normalized attribute appears with both polarities
# in the same dialogue.
# ------------------------------------------------------

contradictions = []

for d in dialogs:
    neg_by_attr = defaultdict(list)
    pos_by_attr = defaultdict(list)

    for turn in d["turns"]:
        t = turn["turn"]

        for b in turn["negatives"]:
            neg_by_attr[norm_attr(b["attribute"])].append({
                "turn": t,
                "confidence": float(b.get("confidence", 1.0)),
                "evidence": b.get("evidence"),
                "question": turn.get("question"),
                "answer": turn.get("answer"),
            })

        for b in turn["positives"]:
            pos_by_attr[norm_attr(b["attribute"])].append({
                "turn": t,
                "confidence": float(b.get("confidence", 1.0)),
                "evidence": b.get("evidence"),
                "question": turn.get("question"),
                "answer": turn.get("answer"),
            })

    overlap = sorted(set(neg_by_attr) & set(pos_by_attr))

    for attr in overlap:
        contradictions.append({
            "dialog_id": d["dialog_id"],
            "attribute": attr,
            "negative_mentions": neg_by_attr[attr],
            "positive_mentions": pos_by_attr[attr],
        })


# ------------------------------------------------------
# 3. Negation density buckets
# ------------------------------------------------------

def bucket(n: int) -> str:
    if n <= 1:
        return "0-1"
    if n <= 3:
        return "2-3"
    if n <= 5:
        return "4-5"
    return "6+"


density_rows = []

for row in per_dialog:
    density_rows.append({
        **row,
        "density_bucket": bucket(row["num_negative"]),
    })


# ------------------------------------------------------
# 4. Retrieval gain analysis
# ------------------------------------------------------

def analyze_gain(h0_path: Path, nacir_path: Path, name: str):
    h0 = load_ranks(h0_path)
    na = load_ranks(nacir_path)

    h0_hit = h0 < 10
    na_hit = na < 10

    # Per-dialog mean gain across feedback turns only
    dialog_gain = 100 * (
        na_hit[:, 1:].mean(axis=1) -
        h0_hit[:, 1:].mean(axis=1)
    )

    final_gain = 100 * (
        na_hit[:, -1].astype(float) -
        h0_hit[:, -1].astype(float)
    )

    rows = []

    for i, meta in enumerate(density_rows):
        rows.append({
            "backbone": name,
            "dialog_id": i,
            "num_negative": meta["num_negative"],
            "num_unique_negative": meta["num_unique_negative"],
            "density_bucket": meta["density_bucket"],
            "avg_feedback_gain": float(dialog_gain[i]),
            "final_gain": float(final_gain[i]),
        })

    depth = []

    for r in range(1, 11):
        gain = 100 * (
            na_hit[:, r].mean() -
            h0_hit[:, r].mean()
        )

        depth.append({
            "backbone": name,
            "round": r,
            "baseline_r10": 100 * h0_hit[:, r].mean(),
            "nacir_r10": 100 * na_hit[:, r].mean(),
            "delta_r10": float(gain),
        })

    return rows, depth


all_gain_rows = []
all_depth_rows = []

for h0, na, name in [
    (BLIP_H0, BLIP_NACIR, "BLIP"),
    (CLIP_H0, CLIP_NACIR, "CLIP_ViT-L14"),
]:
    rows, depth = analyze_gain(h0, na, name)
    all_gain_rows.extend(rows)
    all_depth_rows.extend(depth)


# ------------------------------------------------------
# 5. Aggregate density analysis
# ------------------------------------------------------

density_summary = []

for backbone in ["BLIP", "CLIP_ViT-L14"]:
    subset = [x for x in all_gain_rows if x["backbone"] == backbone]

    for b in ["0-1", "2-3", "4-5", "6+"]:
        rows = [x for x in subset if x["density_bucket"] == b]

        if not rows:
            continue

        density_summary.append({
            "backbone": backbone,
            "density_bucket": b,
            "n": len(rows),
            "mean_num_negative": float(
                np.mean([x["num_negative"] for x in rows])
            ),
            "avg_feedback_delta_r10": float(
                np.mean([x["avg_feedback_gain"] for x in rows])
            ),
            "final_delta_r10": float(
                np.mean([x["final_gain"] for x in rows])
            ),
        })


# ------------------------------------------------------
# 6. Correlation: #negatives vs per-dialog gain
# ------------------------------------------------------

correlations = {}

for backbone in ["BLIP", "CLIP_ViT-L14"]:
    rows = [x for x in all_gain_rows if x["backbone"] == backbone]

    x = np.array([r["num_negative"] for r in rows], dtype=float)
    y = np.array([r["avg_feedback_gain"] for r in rows], dtype=float)

    if np.std(x) > 0 and np.std(y) > 0:
        corr = float(np.corrcoef(x, y)[0, 1])
    else:
        corr = None

    correlations[backbone] = corr


# ------------------------------------------------------
# 7. Save outputs
# ------------------------------------------------------

stats = {
    "num_dialogs": len(dialogs),
    "feedback_turns_per_dialog": 10,

    "negative": {
        "total": int(sum(num_neg)),
        "mean_per_dialog": float(np.mean(num_neg)),
        "median_per_dialog": float(np.median(num_neg)),
        "max_per_dialog": int(np.max(num_neg)),
        "dialogs_with_any_negative": int(np.sum(np.array(num_neg) > 0)),
        "dialogs_with_no_negative": int(np.sum(np.array(num_neg) == 0)),
        "mean_unique_per_dialog": float(np.mean(unique_neg)),
        "high_confidence_total": int(sum(num_neg_highconf)),
    },

    "positive": {
        "total": int(sum(num_pos)),
        "mean_per_dialog": float(np.mean(num_pos)),
        "median_per_dialog": float(np.median(num_pos)),
        "max_per_dialog": int(np.max(num_pos)),
    },

    "negative_per_turn": turn_neg_counts.tolist(),
    "positive_per_turn": turn_pos_counts.tolist(),

    "lexical_contradiction_candidates": len(contradictions),

    "top_negative_attributes": all_neg_attrs.most_common(30),
    "top_positive_attributes": all_pos_attrs.most_common(30),

    "negation_gain_correlation": correlations,
}


with open(OUT / "belief_state_stats.json", "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2)


with open(
    OUT / "contradiction_cases.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(contradictions, f, indent=2)


with open(
    OUT / "negation_density_per_dialog.csv",
    "w",
    newline="",
    encoding="utf-8",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=density_rows[0].keys(),
    )
    writer.writeheader()
    writer.writerows(density_rows)


with open(
    OUT / "negation_density_gain.csv",
    "w",
    newline="",
    encoding="utf-8",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=all_gain_rows[0].keys(),
    )
    writer.writeheader()
    writer.writerows(all_gain_rows)


with open(
    OUT / "negation_density_summary.csv",
    "w",
    newline="",
    encoding="utf-8",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=density_summary[0].keys(),
    )
    writer.writeheader()
    writer.writerows(density_summary)


with open(
    OUT / "depth_gain.csv",
    "w",
    newline="",
    encoding="utf-8",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=all_depth_rows[0].keys(),
    )
    writer.writeheader()
    writer.writerows(all_depth_rows)


print("=" * 90)
print("BELIEF STATE ANALYSIS COMPLETE")
print("=" * 90)

print("\nNegative beliefs")
print("total              :", stats["negative"]["total"])
print("mean/dialog        :", stats["negative"]["mean_per_dialog"])
print("median/dialog      :", stats["negative"]["median_per_dialog"])
print("dialogs with neg   :", stats["negative"]["dialogs_with_any_negative"])
print("dialogs without neg:", stats["negative"]["dialogs_with_no_negative"])

print("\nPositive beliefs")
print("total       :", stats["positive"]["total"])
print("mean/dialog :", stats["positive"]["mean_per_dialog"])

print("\nLexical contradiction candidates:",
      len(contradictions))

print("\nCorrelation #negative vs avg NACIR gain")
for k, v in correlations.items():
    print(f"{k:15s}: {v}")

print("\nDensity summary")
for row in density_summary:
    print(
        f'{row["backbone"]:15s} '
        f'{row["density_bucket"]:>4s} '
        f'n={row["n"]:4d} '
        f'avgΔ={row["avg_feedback_delta_r10"]:+.3f} '
        f'finalΔ={row["final_delta_r10"]:+.3f}'
    )

print("\nOutputs:")
for p in sorted(OUT.iterdir()):
    print(" ", p)
