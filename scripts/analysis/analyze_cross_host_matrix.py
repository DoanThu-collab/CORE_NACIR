#!/usr/bin/env python3
"""Summarize the 2-host x 2-retrieval-space NACIR matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from nacir.metrics import compute_metrics


def load(path):
    ranks = np.load(path, allow_pickle=False)["ranks"].astype(np.int64)
    if ranks.shape != (11, 2064):
        raise ValueError(f"{path}: {ranks.shape}")
    return ranks


def summary(ranks):
    metrics = compute_metrics(ranks)
    per_round = np.asarray([float(x) for x in metrics["per_round_recall"]])
    cumulative = np.asarray([float(x) for x in metrics["cumulative_hits"]])
    return {
        "avg": float(per_round[1:].mean()),
        "final": float(per_round[-1]),
        "cumulative": float(cumulative[-1]),
        "bri": float(metrics["bri"]),
    }


def boot(baseline, candidate, seed=20260829, n=20000):
    diff = (
        (candidate[-1] < 10).astype(float)
        - (baseline[-1] < 10).astype(float)
    )
    point = 100 * diff.mean()
    rng = np.random.default_rng(seed)
    values = np.empty(n)
    num_dialogs = len(diff)
    for start in range(0, n, 512):
        stop = min(n, start + 512)
        idx = rng.integers(0, num_dialogs, size=(stop - start, num_dialogs))
        values[start:stop] = 100 * diff[idx].mean(1)
    low, high = np.quantile(values, [0.025, 0.975])
    return float(point), float(low), float(high)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chatir-blip-h0", default="runs_final/chatir_blip_h0/ranks.npz")
    parser.add_argument("--chatir-blip-current", default="runs_final/chatir_blip_nacir_current_turn/ranks.npz")
    parser.add_argument("--chatir-blip-persistent", default="runs_final/chatir_blip_nacir_minus/ranks.npz")
    parser.add_argument("--chatir-clip-h0", default="runs_final/chatir_clip_vitl14_h0/ranks.npz")
    parser.add_argument("--chatir-clip-current", default="runs_final/chatir_clip_vitl14_nacir_current_turn/ranks.npz")
    parser.add_argument("--chatir-clip-persistent", default="runs_final/chatir_clip_vitl14_nacir_minus/ranks.npz")
    parser.add_argument("--plugir-blip-h0", default="outputs/cross_host/plugir_cr_blip_h0/ranks.npz")
    parser.add_argument("--plugir-blip-current", default="outputs/cross_host/plugir_cr_blip_current/ranks.npz")
    parser.add_argument("--plugir-blip-persistent", default="outputs/cross_host/plugir_cr_blip_persistent/ranks.npz")
    parser.add_argument("--plugir-clip-h0", default="outputs/cross_host/plugir_cr_clip_h0/ranks.npz")
    parser.add_argument("--plugir-clip-current", default="outputs/cross_host/plugir_cr_clip_current/ranks.npz")
    parser.add_argument("--plugir-clip-persistent", default="outputs/cross_host/plugir_cr_clip_persistent/ranks.npz")
    parser.add_argument("--out", default="outputs/analysis/cross_host_matrix.json")
    args = parser.parse_args()

    groups = {
        "ChatIR×BLIP": (
            args.chatir_blip_h0,
            args.chatir_blip_current,
            args.chatir_blip_persistent,
        ),
        "ChatIR×CLIP": (
            args.chatir_clip_h0,
            args.chatir_clip_current,
            args.chatir_clip_persistent,
        ),
        "PlugIR×BLIP": (
            args.plugir_blip_h0,
            args.plugir_blip_current,
            args.plugir_blip_persistent,
        ),
        "PlugIR×CLIP": (
            args.plugir_clip_h0,
            args.plugir_clip_current,
            args.plugir_clip_persistent,
        ),
    }

    output = {}
    print(
        f"{'Setting':16s} {'H0':>8s} {'Current':>8s} "
        f"{'Persistent':>10s} {'P-C':>8s} {'95% CI':>20s}"
    )
    print("-" * 80)

    for name, (h0_path, current_path, persistent_path) in groups.items():
        if not all(Path(x).exists() for x in (h0_path, current_path, persistent_path)):
            print(f"{name:16s} MISSING")
            continue

        h0, current, persistent = map(
            load,
            (h0_path, current_path, persistent_path),
        )
        h0_summary, current_summary, persistent_summary = map(
            summary,
            (h0, current, persistent),
        )
        delta, low, high = boot(current, persistent)
        output[name] = {
            "H0": h0_summary,
            "Current": current_summary,
            "Persistent": persistent_summary,
            "persistent_minus_current_final_pp": delta,
            "final_bootstrap_ci": [low, high],
        }
        print(
            f"{name:16s} {h0_summary['final']:8.3f} "
            f"{current_summary['final']:8.3f} {persistent_summary['final']:10.3f} "
            f"{delta:+8.3f} [{low:+6.3f},{high:+6.3f}]"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
