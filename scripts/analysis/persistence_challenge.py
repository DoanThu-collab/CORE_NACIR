#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def canon(x: str) -> str:
    return " ".join(str(x).lower().strip().split())


def load_ranks(path: Path) -> np.ndarray:
    x = np.load(path, allow_pickle=False)["ranks"]
    if x.shape != (11, 2064):
        raise ValueError(f"{path}: expected (11,2064), got {x.shape}")
    return x.astype(np.int64, copy=False)


def build_history_only_states(beliefs_path: Path):
    doc = json.loads(beliefs_path.read_text(encoding="utf-8"))
    dialogs = doc["dialogs"]
    rows = []
    for did, d in enumerate(dialogs):
        last_seen = {}
        turns = d["turns"]
        for feedback_turn, t in enumerate(turns):
            retrieval_turn = feedback_turn + 1
            current = t.get("negatives", []) or []

            current_keys = []
            for neg in current:
                key = canon(neg["attribute"])
                current_keys.append(key)
                last_seen[key] = retrieval_turn

            if not current_keys and last_seen:
                ages = [retrieval_turn - r for r in last_seen.values()]
                rows.append({
                    "dialog_id": did,
                    "retrieval_turn": retrieval_turn,
                    "memory_size": len(last_seen),
                    "latest_age": min(ages),
                    "oldest_age": max(ages),
                })
    return rows


def cluster_bootstrap(records, baseline, candidate, samples=20000, seed=20260829, k=10):
    if not records:
        return {"states": 0, "dialogs": 0, "delta_pp": None, "ci_low": None, "ci_high": None}

    by_dialog = defaultdict(lambda: [0.0, 0])
    for r in records:
        d, t = r["dialog_id"], r["retrieval_turn"]
        diff = float(candidate[t, d] < k) - float(baseline[t, d] < k)
        by_dialog[d][0] += diff
        by_dialog[d][1] += 1

    ids = np.array(sorted(by_dialog), dtype=np.int64)
    sums = np.array([by_dialog[int(i)][0] for i in ids], dtype=np.float64)
    counts = np.array([by_dialog[int(i)][1] for i in ids], dtype=np.float64)

    point = 100.0 * sums.sum() / counts.sum()
    rng = np.random.default_rng(seed)
    boots = np.empty(samples, dtype=np.float64)
    batch = 256
    for start in range(0, samples, batch):
        stop = min(samples, start + batch)
        idx = rng.integers(0, len(ids), size=(stop - start, len(ids)))
        bs = sums[idx].sum(axis=1)
        bc = counts[idx].sum(axis=1)
        boots[start:stop] = 100.0 * bs / bc

    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {
        "states": len(records),
        "dialogs": len(ids),
        "delta_pp": float(point),
        "ci_low": float(lo),
        "ci_high": float(hi),
    }


def recall(records, ranks, k=10):
    if not records:
        return None
    return 100.0 * np.mean([ranks[x["retrieval_turn"], x["dialog_id"]] < k for x in records])


def analyze(name, records, h0, current, persistent):
    mismatches = [
        x for x in records
        if h0[x["retrieval_turn"], x["dialog_id"]] != current[x["retrieval_turn"], x["dialog_id"]]
    ]
    if mismatches:
        raise AssertionError(f"{name}: H0/Current mismatch on {len(mismatches)} history-only states")

    groups = [("ALL_HISTORY_ONLY", records)]
    for label, pred in [
        ("AGE_1", lambda x: x["latest_age"] == 1),
        ("AGE_2_3", lambda x: 2 <= x["latest_age"] <= 3),
        ("AGE_4_PLUS", lambda x: x["latest_age"] >= 4),
        ("MEM_1", lambda x: x["memory_size"] == 1),
        ("MEM_2_3", lambda x: 2 <= x["memory_size"] <= 3),
        ("MEM_4_PLUS", lambda x: x["memory_size"] >= 4),
    ]:
        groups.append((label, [x for x in records if pred(x)]))

    out = []
    for group, recs in groups:
        stat = cluster_bootstrap(recs, h0, persistent)
        out.append({
            "backbone": name,
            "group": group,
            "states": stat["states"],
            "dialogs": stat["dialogs"],
            "h0_r10": recall(recs, h0),
            "current_r10": recall(recs, current),
            "persistent_r10": recall(recs, persistent),
            "persistent_minus_h0_pp": stat["delta_pp"],
            "ci_low": stat["ci_low"],
            "ci_high": stat["ci_high"],
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beliefs", type=Path, required=True)
    ap.add_argument("--blip-h0", type=Path, default=Path("runs_final/chatir_blip_h0/ranks.npz"))
    ap.add_argument("--blip-current", type=Path, default=Path("runs_final/chatir_blip_nacir_current_turn/ranks.npz"))
    ap.add_argument("--blip-persistent", type=Path, default=Path("runs_final/chatir_blip_nacir_minus/ranks.npz"))
    ap.add_argument("--clip-h0", type=Path, default=Path("runs_final/chatir_clip_vitl14_h0/ranks.npz"))
    ap.add_argument("--clip-current", type=Path, default=Path("runs_final/chatir_clip_vitl14_nacir_current_turn/ranks.npz"))
    ap.add_argument("--clip-persistent", type=Path, default=Path("runs_final/chatir_clip_vitl14_nacir_minus/ranks.npz"))
    ap.add_argument("--out", type=Path, default=Path("artifacts_final/analysis/persistence_challenge.csv"))
    args = ap.parse_args()

    states = build_history_only_states(args.beliefs)
    if not states:
        raise RuntimeError("No history-only persistence states found")

    results = []
    results += analyze("BLIP", states, load_ranks(args.blip_h0), load_ranks(args.blip_current), load_ranks(args.blip_persistent))
    results += analyze("CLIP_ViT-L14", states, load_ranks(args.clip_h0), load_ranks(args.clip_current), load_ranks(args.clip_persistent))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)

    print("=" * 110)
    print("PERSISTENCE CHALLENGE: current turn has zero negatives, historical negative memory is non-empty")
    print("=" * 110)
    for r in results:
        d = r["persistent_minus_h0_pp"]
        if d is None:
            continue
        print(
            f"{r['backbone']:12s} {r['group']:18s} "
            f"states={r['states']:4d} dialogs={r['dialogs']:4d} "
            f"H0={r['h0_r10']:6.2f} Current={r['current_r10']:6.2f} "
            f"Persistent={r['persistent_r10']:6.2f} "
            f"Δ={d:+6.2f} [{r['ci_low']:+6.2f},{r['ci_high']:+6.2f}]"
        )
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
