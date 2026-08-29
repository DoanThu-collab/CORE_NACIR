#!/usr/bin/env python3
"""Discover and summarize existing PlugIR / stronger-host artifacts.

Goal: avoid regenerating anything under deadline. The script scans existing
artifacts/runs, inspects likely session/rank files, and reports what can be used
immediately for a host-boundary experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nacir.metrics import compute_metrics


KEYWORDS = ("plugir", "recon", "active", "context", "reform", "256")


def safe_torch_summary(path: Path) -> dict[str, Any]:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        return {"kind": "torch", "load_error": str(e)}

    if isinstance(obj, torch.Tensor):
        return {"kind": "tensor", "shape": list(obj.shape), "dtype": str(obj.dtype)}

    if isinstance(obj, dict):
        out = {"kind": "dict", "keys": sorted(map(str, obj.keys()))[:30]}
        vec = obj.get("vectors")
        if isinstance(vec, torch.Tensor):
            out["vectors_shape"] = list(vec.shape)
        return out

    if isinstance(obj, list):
        out = {"kind": "list", "len": len(obj)}
        if obj and isinstance(obj[0], dict):
            first = obj[0]
            out["first_keys"] = sorted(map(str, first.keys()))
            qv = first.get("query_vectors")
            qt = first.get("query_texts")
            if isinstance(qv, torch.Tensor):
                out["first_query_vectors_shape"] = list(qv.shape)
            if isinstance(qt, list):
                out["first_query_texts_len"] = len(qt)
            if "target_index" in first:
                out["first_target_index"] = int(first["target_index"])
            if "session_id" in first:
                out["first_session_id"] = int(first["session_id"])
        return out

    return {"kind": type(obj).__name__}


def safe_json_summary(path: Path) -> dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"kind": "json", "load_error": str(e)}

    if isinstance(obj, list):
        out = {"kind": "list", "len": len(obj)}
        if obj and isinstance(obj[0], dict):
            out["first_keys"] = sorted(map(str, obj[0].keys()))[:30]
        return out
    if isinstance(obj, dict):
        out = {"kind": "dict", "keys": sorted(map(str, obj.keys()))[:30]}
        if isinstance(obj.get("dialogs"), list):
            out["dialogs_len"] = len(obj["dialogs"])
        if isinstance(obj.get("items"), list):
            out["items_len"] = len(obj["items"])
        return out
    return {"kind": type(obj).__name__}


def rank_summary(path: Path) -> dict[str, Any]:
    try:
        z = np.load(path, allow_pickle=False)
        if "ranks" not in z:
            return {"error": "no ranks key"}
        ranks = z["ranks"].astype(np.int64)
        if ranks.ndim != 2:
            return {"shape": list(ranks.shape), "error": "not 2D"}
        metrics = compute_metrics(ranks)
        per = [float(x) for x in metrics["per_round_recall"]]
        cum = [float(x) for x in metrics["cumulative_hits"]]
        return {
            "shape": list(ranks.shape),
            "avg_feedback_r10": float(np.mean(per[1:])) if len(per) > 1 else None,
            "final_r10": per[-1],
            "final_cumulative_hits10": cum[-1],
            "bri": float(metrics["bri"]),
        }
    except Exception as e:
        return {"error": str(e)}


def candidate_files(root: Path):
    roots = [root / "artifacts_final", root / "runs_final", root / "runs_deadline", root / "scripts"]
    found = []
    for base in roots:
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if not p.is_file():
                continue
            rel = str(p.relative_to(root)).lower()
            if any(k in rel for k in KEYWORDS):
                found.append(p)
    return sorted(found)


def classify_session_candidates(rows):
    sessions_2064 = []
    sessions_256 = []
    for row in rows:
        s = row.get("summary", {})
        if s.get("kind") == "list" and "first_query_vectors_shape" in s:
            n = s.get("len")
            if n == 2064:
                sessions_2064.append(row["path"])
            elif n == 256:
                sessions_256.append(row["path"])
    return sessions_2064, sessions_256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts_final/analysis/plugir_artifact_discovery.json"),
    )
    args = ap.parse_args()
    root = args.root.resolve()

    rows = []
    for p in candidate_files(root):
        rel = str(p.relative_to(root))
        suffix = p.suffix.lower()
        if suffix in {".pt", ".pth"}:
            summary = safe_torch_summary(p)
        elif suffix == ".json":
            summary = safe_json_summary(p)
        elif suffix == ".npz":
            summary = rank_summary(p)
        else:
            summary = {"kind": suffix.lstrip(".")}
        rows.append({"path": rel, "summary": summary})

    s2064, s256 = classify_session_candidates(rows)
    rank_rows = [r for r in rows if r["path"].endswith(".npz") and "shape" in r["summary"]]

    report = {
        "session_candidates_2064": s2064,
        "session_candidates_256": s256,
        "rank_candidates": rank_rows,
        "all_candidates": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 110)
    print("PLUGIR / STRONGER-HOST ARTIFACT DISCOVERY")
    print("=" * 110)

    print("\n2064-session standardized candidates:")
    if s2064:
        for p in s2064:
            print("  ", p)
    else:
        print("   NONE")

    print("\n256-session standardized candidates:")
    if s256:
        for p in s256:
            print("  ", p)
    else:
        print("   NONE")

    print("\nExisting PlugIR/reconstruction rank files:")
    if rank_rows:
        for r in rank_rows:
            m = r["summary"]
            print(
                f"  {r['path']}\n"
                f"    shape={m.get('shape')} "
                f"avgR10={m.get('avg_feedback_r10')} "
                f"finalR10={m.get('final_r10')} "
                f"cum={m.get('final_cumulative_hits10')} "
                f"BRI={m.get('bri')}"
            )
    else:
        print("   NONE")

    print("\nAll relevant artifact files:")
    for r in rows:
        print("  ", r["path"], "=>", r["summary"])

    print("\nSaved:", args.out)


if __name__ == "__main__":
    main()
