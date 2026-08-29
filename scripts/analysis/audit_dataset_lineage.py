#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import torch


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_tensor(path: Path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return obj.get("vectors") if isinstance(obj, dict) and "vectors" in obj else obj


def summarize_sessions(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sessions = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(sessions, list) or not sessions:
        raise ValueError(f"{path}: expected non-empty list")

    ids, targets, turns, dims = [], [], [], []
    have_texts = True
    for i, s in enumerate(sessions):
        if not isinstance(s, dict):
            raise ValueError(f"{path}: session {i} is not a dict")
        sid = s.get("session_id")
        target = s.get("target_index")
        qv = s.get("query_vectors")
        qt = s.get("query_texts")
        if not isinstance(qv, torch.Tensor) or qv.ndim != 2:
            raise ValueError(f"{path}: invalid query_vectors at session {i}")
        ids.append(int(sid))
        targets.append(int(target))
        turns.append(int(qv.shape[0]))
        dims.append(int(qv.shape[1]))
        have_texts &= isinstance(qt, list) and len(qt) == qv.shape[0]

    summary = {
        "path": str(path),
        "sha256": sha256(path),
        "num_sessions": len(sessions),
        "session_ids_contiguous_0_based": ids == list(range(len(sessions))),
        "num_turns_unique": sorted(set(turns)),
        "embedding_dims_unique": sorted(set(dims)),
        "query_texts_available_for_all": bool(have_texts),
        "target_min": min(targets),
        "target_max": max(targets),
    }
    return summary, sessions


def summarize_corpus(path: Path) -> dict[str, Any]:
    x = load_tensor(path)
    if not isinstance(x, torch.Tensor) or x.ndim != 2:
        raise ValueError(f"{path}: expected [N,D] tensor")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "finite": bool(torch.isfinite(x).all()),
    }


def summarize_beliefs(path: Path) -> dict[str, Any]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    dialogs = doc.get("dialogs")
    if not isinstance(dialogs, list):
        raise ValueError("belief artifact missing dialogs list")

    pos = neg = turns = 0
    bad_ids = 0
    turn_counts = []
    for did, d in enumerate(dialogs):
        if d.get("dialog_id") != did:
            bad_ids += 1
        ts = d.get("turns", [])
        turn_counts.append(len(ts))
        for tid, t in enumerate(ts):
            turns += 1
            pos += len(t.get("positives", []) or [])
            neg += len(t.get("negatives", []) or [])

    return {
        "path": str(path),
        "sha256": sha256(path),
        "schema_version": doc.get("schema_version"),
        "status": doc.get("status"),
        "quality_passed": doc.get("quality", {}).get("passed"),
        "num_dialogues": len(dialogs),
        "turns_per_dialogue_unique": sorted(set(turn_counts)),
        "num_feedback_turns": turns,
        "num_positive_beliefs": pos,
        "num_negative_beliefs": neg,
        "dialog_id_mismatches": bad_ids,
        "provenance": doc.get("provenance", {}),
    }


def git_search(term: str) -> list[str]:
    try:
        out = subprocess.check_output(
            ["git", "log", "--all", "--oneline", "-S", term, "--", "configs", "scripts", "src"],
            text=True,
            stderr=subprocess.STDOUT,
        )
        return [x for x in out.splitlines() if x.strip()][:100]
    except Exception as e:
        return [f"git search failed: {e}"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beliefs", type=Path, default=Path(
        "/mlcv1/WorkingSpace/Personal/core_baotg/thuy/NACIR_FIX/data/beliefs_v2/"
        "llama3_1_8b_v9_final_20260824.json"
    ))
    ap.add_argument("--config", type=Path, default=Path("configs/nacir_minus_frozen.json"))
    ap.add_argument("--blip-sessions", type=Path, default=Path("artifacts_final/sessions_chatir_blip.pt"))
    ap.add_argument("--clip-sessions", type=Path, default=Path("artifacts_final/sessions_chatir_clip_vitl14.pt"))
    ap.add_argument("--blip-corpus", type=Path, default=Path("artifacts_final/corpus_blip_large_vectors.pt"))
    ap.add_argument("--clip-corpus", type=Path, default=Path("artifacts_final/corpus_openai_clip_vitl14_vectors.pt"))
    ap.add_argument("--out", type=Path, default=Path("artifacts_final/analysis/dataset_lineage_audit.json"))
    args = ap.parse_args()

    required = [args.beliefs, args.config, args.blip_sessions, args.clip_sessions, args.blip_corpus, args.clip_corpus]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print("MISSING FILES:")
        for p in missing:
            print(" -", p)
        raise SystemExit(2)

    blip_info, blip_sessions = summarize_sessions(args.blip_sessions)
    clip_info, clip_sessions = summarize_sessions(args.clip_sessions)
    blip_corpus = summarize_corpus(args.blip_corpus)
    clip_corpus = summarize_corpus(args.clip_corpus)
    beliefs = summarize_beliefs(args.beliefs)
    config = json.loads(args.config.read_text(encoding="utf-8"))

    alignment = {
        "same_num_sessions": len(blip_sessions) == len(clip_sessions),
        "session_id_equal": True,
        "target_index_equal": True,
        "query_texts_equal_where_available": None,
        "query_text_sessions_compared": 0,
    }
    text_equal = True
    compared = 0
    for b, c in zip(blip_sessions, clip_sessions):
        if int(b["session_id"]) != int(c["session_id"]):
            alignment["session_id_equal"] = False
        if int(b["target_index"]) != int(c["target_index"]):
            alignment["target_index_equal"] = False
        bt, ct = b.get("query_texts"), c.get("query_texts")
        if isinstance(bt, list) and isinstance(ct, list):
            compared += 1
            if bt != ct:
                text_equal = False
    if compared:
        alignment["query_texts_equal_where_available"] = text_equal
        alignment["query_text_sessions_compared"] = compared

    checks = {
        "beliefs_2064_dialogues": beliefs["num_dialogues"] == 2064,
        "beliefs_10_feedback_turns_each": beliefs["turns_per_dialogue_unique"] == [10],
        "beliefs_expected_negative_count_6464": beliefs["num_negative_beliefs"] == 6464,
        "blip_2064_sessions": blip_info["num_sessions"] == 2064,
        "clip_2064_sessions": clip_info["num_sessions"] == 2064,
        "blip_11_retrieval_states": blip_info["num_turns_unique"] == [11],
        "clip_11_retrieval_states": clip_info["num_turns_unique"] == [11],
        "same_session_ids": alignment["session_id_equal"],
        "same_target_indices": alignment["target_index_equal"],
        "blip_corpus_50000": blip_corpus["shape"][0] == 50000,
        "clip_corpus_50000": clip_corpus["shape"][0] == 50000,
        "blip_session_dim_matches_corpus": blip_info["embedding_dims_unique"] == [blip_corpus["shape"][1]],
        "clip_session_dim_matches_corpus": clip_info["embedding_dims_unique"] == [clip_corpus["shape"][1]],
    }

    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "beliefs": beliefs,
        "config": {"path": str(args.config), "sha256": sha256(args.config), "contents": config},
        "blip_sessions": blip_info,
        "clip_sessions": clip_info,
        "blip_corpus": blip_corpus,
        "clip_corpus": clip_corpus,
        "cross_backbone_alignment": alignment,
        "parameter_history_search": {
            "lambda_0.275": git_search("0.275"),
            "rho_0.1": git_search('"recency_decay": 0.1'),
        },
        "manual_question_required": (
            "Were lambda=0.275 and rho=0.10 selected after inspecting performance on "
            "these same 2064 evaluation dialogues? Files/git history cannot prove this. "
            "If yes or uncertain, create a fixed development/test split before making "
            "held-out claims."
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 80)
    print("DATASET / ARTIFACT LINEAGE AUDIT")
    print("=" * 80)
    for k, v in checks.items():
        print(f"{'PASS' if v else 'FAIL':4s}  {k}")
    print()
    print("Beliefs:", beliefs["num_dialogues"], "dialogs,", beliefs["num_feedback_turns"],
          "feedback turns,", beliefs["num_negative_beliefs"], "negative beliefs")
    print("BLIP sessions:", blip_info["num_sessions"], blip_info["num_turns_unique"],
          "dim", blip_info["embedding_dims_unique"], "corpus", blip_corpus["shape"])
    print("CLIP sessions:", clip_info["num_sessions"], clip_info["num_turns_unique"],
          "dim", clip_info["embedding_dims_unique"], "corpus", clip_corpus["shape"])
    print("Cross-backbone target alignment:", alignment["target_index_equal"])
    print("Cross-backbone query-text alignment:", alignment["query_texts_equal_where_available"])
    print("Overall:", report["status"])
    print("Saved:", args.out)
    print()
    print("MANUAL PARAMETER-SELECTION CHECK:")
    print(report["manual_question_required"])


if __name__ == "__main__":
    main()
