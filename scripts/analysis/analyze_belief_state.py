#!/usr/bin/env python3
"""Summarize the frozen belief artifact and retrieval gains by negative density."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def norm_attr(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    return re.sub(r"\s+", " ", text)


def load_ranks(path: Path) -> np.ndarray:
    ranks = np.load(path, allow_pickle=False)["ranks"]
    if ranks.shape == (11, 2064):
        ranks = ranks.T
    if ranks.shape != (2064, 11):
        raise ValueError(f"{path}: expected (11,2064) or (2064,11), got {ranks.shape}")
    return ranks.astype(np.int64, copy=False)


def density_bucket(count: int) -> str:
    if count <= 1:
        return "0-1"
    if count <= 3:
        return "2-3"
    if count <= 5:
        return "4-5"
    return "6+"


def analyze_gain(
    metadata: list[dict],
    h0_path: Path,
    persistent_path: Path,
    backbone: str,
) -> tuple[list[dict], list[dict]]:
    h0 = load_ranks(h0_path)
    persistent = load_ranks(persistent_path)
    h0_hit = h0 < 10
    persistent_hit = persistent < 10

    per_dialog_gain = 100 * (
        persistent_hit[:, 1:].mean(axis=1)
        - h0_hit[:, 1:].mean(axis=1)
    )
    final_gain = 100 * (
        persistent_hit[:, -1].astype(float)
        - h0_hit[:, -1].astype(float)
    )

    rows = []
    for index, meta in enumerate(metadata):
        rows.append(
            {
                "backbone": backbone,
                "dialog_id": index,
                "num_negative": meta["num_negative"],
                "num_unique_negative": meta["num_unique_negative"],
                "density_bucket": meta["density_bucket"],
                "avg_feedback_gain": float(per_dialog_gain[index]),
                "final_gain": float(final_gain[index]),
            }
        )

    depth = []
    for round_index in range(1, 11):
        depth.append(
            {
                "backbone": backbone,
                "round": round_index,
                "baseline_r10": float(100 * h0_hit[:, round_index].mean()),
                "persistent_r10": float(100 * persistent_hit[:, round_index].mean()),
                "delta_r10": float(
                    100
                    * (
                        persistent_hit[:, round_index].mean()
                        - h0_hit[:, round_index].mean()
                    )
                ),
            }
        )
    return rows, depth


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--beliefs", type=Path, required=True)
    parser.add_argument(
        "--blip-h0",
        type=Path,
        default=Path("runs_final/chatir_blip_h0/ranks.npz"),
    )
    parser.add_argument(
        "--blip-persistent",
        type=Path,
        default=Path("runs_final/chatir_blip_nacir_minus/ranks.npz"),
    )
    parser.add_argument(
        "--clip-h0",
        type=Path,
        default=Path("runs_final/chatir_clip_vitl14_h0/ranks.npz"),
    )
    parser.add_argument(
        "--clip-persistent",
        type=Path,
        default=Path("runs_final/chatir_clip_vitl14_nacir_minus/ranks.npz"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/analysis/belief_state"),
    )
    args = parser.parse_args()

    document = json.loads(args.beliefs.read_text(encoding="utf-8"))
    dialogs = document.get("dialogs")
    if not isinstance(dialogs, list) or len(dialogs) != 2064:
        raise ValueError("expected canonical belief artifact with 2064 dialogs")

    num_negative = []
    num_positive = []
    num_negative_highconf = []
    unique_negative = []
    negative_per_turn = np.zeros(10, dtype=int)
    positive_per_turn = np.zeros(10, dtype=int)
    all_negative = Counter()
    all_positive = Counter()
    per_dialog = []
    contradictions = []

    for expected_id, dialog in enumerate(dialogs):
        if int(dialog.get("dialog_id", -1)) != expected_id:
            raise ValueError(f"dialog order mismatch at {expected_id}")

        negatives = []
        positives = []
        neg_by_attr = defaultdict(list)
        pos_by_attr = defaultdict(list)

        for turn in dialog.get("turns", []):
            turn_index = int(turn["turn"])
            turn_negatives = turn.get("negatives", []) or []
            turn_positives = turn.get("positives", []) or []
            negative_per_turn[turn_index] += len(turn_negatives)
            positive_per_turn[turn_index] += len(turn_positives)

            for belief in turn_negatives:
                attr = norm_attr(belief["attribute"])
                confidence = float(belief.get("confidence", 1.0))
                negatives.append((turn_index, attr, confidence))
                all_negative[attr] += 1
                neg_by_attr[attr].append(turn_index)

            for belief in turn_positives:
                attr = norm_attr(belief["attribute"])
                confidence = float(belief.get("confidence", 1.0))
                positives.append((turn_index, attr, confidence))
                all_positive[attr] += 1
                pos_by_attr[attr].append(turn_index)

        num_negative.append(len(negatives))
        num_positive.append(len(positives))
        num_negative_highconf.append(sum(c >= 0.7 for _, _, c in negatives))
        unique_negative.append(len({a for _, a, _ in negatives}))

        per_dialog.append(
            {
                "dialog_id": expected_id,
                "num_negative": len(negatives),
                "num_positive": len(positives),
                "num_negative_highconf": sum(c >= 0.7 for _, _, c in negatives),
                "num_unique_negative": len({a for _, a, _ in negatives}),
                "density_bucket": density_bucket(len(negatives)),
            }
        )

        for attr in sorted(set(neg_by_attr) & set(pos_by_attr)):
            contradictions.append(
                {
                    "dialog_id": expected_id,
                    "attribute": attr,
                    "negative_turns": neg_by_attr[attr],
                    "positive_turns": pos_by_attr[attr],
                }
            )

    stats = {
        "num_dialogs": len(dialogs),
        "feedback_turns_per_dialog": 10,
        "negative": {
            "total": int(sum(num_negative)),
            "mean_per_dialog": float(np.mean(num_negative)),
            "median_per_dialog": float(np.median(num_negative)),
            "max_per_dialog": int(np.max(num_negative)),
            "dialogs_with_any_negative": int(np.sum(np.asarray(num_negative) > 0)),
            "dialogs_with_no_negative": int(np.sum(np.asarray(num_negative) == 0)),
            "mean_unique_per_dialog": float(np.mean(unique_negative)),
            "high_confidence_total": int(sum(num_negative_highconf)),
        },
        "positive": {
            "total": int(sum(num_positive)),
            "mean_per_dialog": float(np.mean(num_positive)),
            "median_per_dialog": float(np.median(num_positive)),
            "max_per_dialog": int(np.max(num_positive)),
        },
        "negative_per_turn": negative_per_turn.tolist(),
        "positive_per_turn": positive_per_turn.tolist(),
        "lexical_contradiction_candidates": len(contradictions),
        "top_negative_attributes": all_negative.most_common(30),
        "top_positive_attributes": all_positive.most_common(30),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "belief_state_stats.json").write_text(
        json.dumps(stats, indent=2),
        encoding="utf-8",
    )
    (args.out_dir / "contradiction_cases.json").write_text(
        json.dumps(contradictions, indent=2),
        encoding="utf-8",
    )
    write_csv(args.out_dir / "negation_density_per_dialog.csv", per_dialog)

    gain_rows = []
    depth_rows = []
    for h0_path, persistent_path, backbone in [
        (args.blip_h0, args.blip_persistent, "BLIP"),
        (args.clip_h0, args.clip_persistent, "CLIP_ViT-L14"),
    ]:
        if h0_path.exists() and persistent_path.exists():
            rows, depth = analyze_gain(
                per_dialog,
                h0_path,
                persistent_path,
                backbone,
            )
            gain_rows.extend(rows)
            depth_rows.extend(depth)

    write_csv(args.out_dir / "negation_density_gain.csv", gain_rows)
    write_csv(args.out_dir / "depth_gain.csv", depth_rows)

    print("BELIEF STATE ANALYSIS COMPLETE")
    print("negative total:", stats["negative"]["total"])
    print("positive total:", stats["positive"]["total"])
    print("lexical contradiction candidates:", len(contradictions))
    print("saved:", args.out_dir)


if __name__ == "__main__":
    main()
