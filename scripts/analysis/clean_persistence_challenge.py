#!/usr/bin/env python3
"""Build the clean history-only persistence challenge from frozen artifacts.

This script does NOT run a retrieval model.  It selects retrieval states where:
  1) the current feedback turn contains zero extracted negative beliefs;
  2) at least one historical negative belief is still active;
  3) the historical negative is marked actionable by the frozen structured artifact;
  4) an exact-canonical positive belief observed after that negative invalidates it
     until a later actionable negative re-establishes the concept.

Because condition (1) holds, the frozen Current-turn NACIR run must be exactly
identical to H0 on every selected state.  The Persistent-vs-H0 difference therefore
isolates historical negative state.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def canon(text: str) -> str:
    return " ".join(str(text).lower().strip().split())


def load_ranks(path: Path) -> np.ndarray:
    ranks = np.load(path, allow_pickle=False)["ranks"]
    if ranks.shape != (11, 2064):
        raise ValueError(f"{path}: expected rank matrix (11,2064), got {ranks.shape}")
    if not np.issubdtype(ranks.dtype, np.integer):
        raise ValueError(f"{path}: ranks must be integers")
    return ranks.astype(np.int64, copy=False)


def load_actionability(path: Path) -> dict[tuple[int, int, int], dict]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    items = artifact.get("items")
    if not isinstance(items, list):
        raise ValueError("structured artifact must contain an items list")

    out = {}
    for item in items:
        key = (
            int(item["dialog_id"]),
            int(item["turn"]),
            int(item.get("negative_index", 0)),
        )
        if key in out:
            raise ValueError(f"duplicate structured key: {key}")
        if not isinstance(item.get("actionable_negative"), bool):
            raise ValueError(f"missing/invalid actionable_negative for {key}")
        out[key] = item

    if len(out) != 6464:
        raise ValueError(f"expected 6464 structured negatives, found {len(out)}")
    return out


def build_states(beliefs_path: Path, structured_path: Path) -> tuple[list[dict], dict]:
    beliefs = json.loads(beliefs_path.read_text(encoding="utf-8"))
    dialogs = beliefs.get("dialogs")
    if not isinstance(dialogs, list) or len(dialogs) != 2064:
        raise ValueError("expected canonical schema-v2 beliefs with 2064 dialogs")

    structured = load_actionability(structured_path)

    states: list[dict] = []
    counters = {
        "negative_events": 0,
        "actionable_negative_events": 0,
        "non_actionable_negative_events": 0,
        "exact_positive_invalidations": 0,
        "reactivations_after_positive": 0,
    }

    for did, dialog in enumerate(dialogs):
        if int(dialog.get("dialog_id", -1)) != did:
            raise ValueError(f"belief dialog order mismatch at {did}")

        turns = dialog.get("turns")
        if not isinstance(turns, list) or len(turns) != 10:
            raise ValueError(f"dialog {did}: expected 10 feedback turns")

        # concept -> metadata for currently valid actionable historical negative.
        active: dict[str, dict] = {}
        invalidated_once: set[str] = set()

        for feedback_turn, turn in enumerate(turns):
            retrieval_turn = feedback_turn + 1

            positives = turn.get("positives", []) or []
            negatives = turn.get("negatives", []) or []
            if not isinstance(positives, list) or not isinstance(negatives, list):
                raise ValueError(f"dialog {did}, turn {feedback_turn}: invalid belief lists")

            # Exact positive evidence invalidates a previously active exact-canonical
            # negative before evaluating this retrieval state.
            for pos in positives:
                key = canon(pos["attribute"])
                if key in active:
                    del active[key]
                    counters["exact_positive_invalidations"] += 1
                    invalidated_once.add(key)

            # Add/refresh only frozen-actionable negatives to the CLEAN state.
            for ni, neg in enumerate(negatives):
                counters["negative_events"] += 1
                rec_key = (did, feedback_turn, ni)
                rec = structured.get(rec_key)
                if rec is None:
                    raise KeyError(f"missing structured record for {rec_key}")

                expected = canon(rec["negative_attribute"])
                actual = canon(neg["attribute"])
                if expected != actual:
                    raise ValueError(
                        f"structured/belief mismatch {rec_key}: "
                        f"{rec['negative_attribute']!r} != {neg['attribute']!r}"
                    )

                if not rec["actionable_negative"]:
                    counters["non_actionable_negative_events"] += 1
                    continue

                counters["actionable_negative_events"] += 1
                if actual in invalidated_once:
                    counters["reactivations_after_positive"] += 1
                    invalidated_once.discard(actual)

                active[actual] = {
                    "concept": actual,
                    "last_negative_retrieval_turn": retrieval_turn,
                    "source_feedback_turn": feedback_turn,
                    "confidence": float(neg["confidence"]),
                }

            # Critical condition: RAW current turn has zero negatives.  This guarantees
            # Current-turn NACIR has empty memory and must equal H0 exactly.
            if len(negatives) == 0 and active:
                ages = [
                    retrieval_turn - x["last_negative_retrieval_turn"]
                    for x in active.values()
                ]
                states.append(
                    {
                        "dialog_id": did,
                        "retrieval_turn": retrieval_turn,
                        "active_memory_size": len(active),
                        "latest_age": min(ages),
                        "oldest_age": max(ages),
                        "active_concepts": sorted(active),
                    }
                )

    return states, counters


def recall(records: list[dict], ranks: np.ndarray, k: int = 10) -> float | None:
    if not records:
        return None
    hit = [
        ranks[r["retrieval_turn"], r["dialog_id"]] < k
        for r in records
    ]
    return 100.0 * float(np.mean(hit))


def clustered_bootstrap(
    records: list[dict],
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    k: int = 10,
    samples: int = 20_000,
    seed: int = 20260829,
) -> dict:
    if not records:
        return {
            "states": 0,
            "dialogs": 0,
            "delta_pp": None,
            "ci_low": None,
            "ci_high": None,
        }

    by_dialog: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for r in records:
        did, turn = r["dialog_id"], r["retrieval_turn"]
        delta = float(candidate[turn, did] < k) - float(baseline[turn, did] < k)
        by_dialog[did][0] += delta
        by_dialog[did][1] += 1.0

    ids = np.asarray(sorted(by_dialog), dtype=np.int64)
    sums = np.asarray([by_dialog[int(i)][0] for i in ids], dtype=np.float64)
    counts = np.asarray([by_dialog[int(i)][1] for i in ids], dtype=np.float64)

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


def group_records(states: list[dict]) -> list[tuple[str, list[dict]]]:
    specs = [
        ("ALL_CLEAN_HISTORY_ONLY", lambda x: True),
        ("AGE_1", lambda x: x["latest_age"] == 1),
        ("AGE_2_3", lambda x: 2 <= x["latest_age"] <= 3),
        ("AGE_4_PLUS", lambda x: x["latest_age"] >= 4),
        ("MEM_1", lambda x: x["active_memory_size"] == 1),
        ("MEM_2_3", lambda x: 2 <= x["active_memory_size"] <= 3),
        ("MEM_4_PLUS", lambda x: x["active_memory_size"] >= 4),
    ]
    return [(name, [x for x in states if pred(x)]) for name, pred in specs]


def analyze(
    backbone: str,
    states: list[dict],
    h0: np.ndarray,
    current: np.ndarray,
    persistent: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> list[dict]:
    mismatches = [
        r for r in states
        if h0[r["retrieval_turn"], r["dialog_id"]]
        != current[r["retrieval_turn"], r["dialog_id"]]
    ]
    if mismatches:
        first = mismatches[0]
        raise AssertionError(
            f"{backbone}: H0 != Current on {len(mismatches)} clean history-only states; "
            f"first={first}"
        )

    rows = []
    for name, recs in group_records(states):
        stat = clustered_bootstrap(
            recs, h0, persistent, samples=samples, seed=seed
        )
        rows.append(
            {
                "backbone": backbone,
                "group": name,
                "states": stat["states"],
                "dialogs": stat["dialogs"],
                "h0_r10": recall(recs, h0),
                "current_r10": recall(recs, current),
                "persistent_r10": recall(recs, persistent),
                "persistent_minus_h0_pp": stat["delta_pp"],
                "ci_low": stat["ci_low"],
                "ci_high": stat["ci_high"],
                "h0_current_rank_mismatches": 0,
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--beliefs",
        type=Path,
        default=Path(
            "/mlcv1/WorkingSpace/Personal/core_baotg/thuy/NACIR_FIX/"
            "data/beliefs_v2/llama3_1_8b_v9_final_20260824.json"
        ),
    )
    ap.add_argument(
        "--structured",
        type=Path,
        default=Path(
            "artifacts_final/typed_nacir/chatir_structured_negative_final_v1_1.json"
        ),
    )
    ap.add_argument("--blip-h0", type=Path, default=Path("runs_final/chatir_blip_h0/ranks.npz"))
    ap.add_argument("--blip-current", type=Path, default=Path("runs_final/chatir_blip_nacir_current_turn/ranks.npz"))
    ap.add_argument("--blip-persistent", type=Path, default=Path("runs_final/chatir_blip_nacir_minus/ranks.npz"))
    ap.add_argument("--clip-h0", type=Path, default=Path("runs_final/chatir_clip_vitl14_h0/ranks.npz"))
    ap.add_argument("--clip-current", type=Path, default=Path("runs_final/chatir_clip_vitl14_nacir_current_turn/ranks.npz"))
    ap.add_argument("--clip-persistent", type=Path, default=Path("runs_final/chatir_clip_vitl14_nacir_minus/ranks.npz"))
    ap.add_argument("--bootstrap-samples", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=20260829)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts_final/analysis/clean_persistence"),
    )
    args = ap.parse_args()

    states, counters = build_states(args.beliefs, args.structured)
    if not states:
        raise RuntimeError("clean persistence challenge is empty")

    blip_h0 = load_ranks(args.blip_h0)
    blip_current = load_ranks(args.blip_current)
    blip_persistent = load_ranks(args.blip_persistent)
    clip_h0 = load_ranks(args.clip_h0)
    clip_current = load_ranks(args.clip_current)
    clip_persistent = load_ranks(args.clip_persistent)

    rows = []
    rows += analyze(
        "BLIP", states, blip_h0, blip_current, blip_persistent,
        samples=args.bootstrap_samples, seed=args.seed,
    )
    rows += analyze(
        "CLIP_ViT-L14", states, clip_h0, clip_current, clip_persistent,
        samples=args.bootstrap_samples, seed=args.seed,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    with (args.out_dir / "clean_persistence_states.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            {
                "definition": {
                    "current_raw_negative_count": 0,
                    "historical_negative_actionable": True,
                    "exact_positive_contradiction_invalidates": True,
                    "later_actionable_negative_reactivates": True,
                },
                "num_states": len(states),
                "num_dialogues": len({x["dialog_id"] for x in states}),
                "counters": counters,
                "states": states,
            },
            f,
            indent=2,
        )

    with (args.out_dir / "clean_persistence_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 118)
    print("CLEAN PERSISTENCE CHALLENGE")
    print("current negatives=0; historical negative actionable; exact later positive invalidates")
    print("=" * 118)
    print(
        f"states={len(states)} dialogs={len({x['dialog_id'] for x in states})} "
        f"actionable_events={counters['actionable_negative_events']} "
        f"non_actionable_events={counters['non_actionable_negative_events']} "
        f"positive_invalidations={counters['exact_positive_invalidations']}"
    )
    for r in rows:
        if r["persistent_minus_h0_pp"] is None:
            continue
        print(
            f"{r['backbone']:12s} {r['group']:24s} "
            f"states={r['states']:4d} dialogs={r['dialogs']:4d} "
            f"H0={r['h0_r10']:6.2f} Current={r['current_r10']:6.2f} "
            f"Persistent={r['persistent_r10']:6.2f} "
            f"Δ={r['persistent_minus_h0_pp']:+6.2f} "
            f"[{r['ci_low']:+6.2f},{r['ci_high']:+6.2f}]"
        )
    print("H0 == Current at rank level on every selected state: PASS")
    print("Saved:", args.out_dir)


if __name__ == "__main__":
    main()
