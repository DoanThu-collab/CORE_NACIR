#!/usr/bin/env python3
"""Upgrade a legacy ranks-only archive with strict NACIR provenance metadata.

This DOES NOT change ranks. It attaches provenance computed from explicitly
declared source artifacts. For old paper runs this is called "rehydrated" rather
than pretending the metadata was emitted historically by the evaluator.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from nacir.provenance import build_provenance, read_rank_archive


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks", type=Path, required=True)
    ap.add_argument("--sessions", type=Path, required=True)
    ap.add_argument("--corpus-vectors", type=Path, required=True)
    ap.add_argument("--beliefs", type=Path)
    ap.add_argument("--method", required=True)
    ap.add_argument("--config", type=Path)
    ap.add_argument("--adapter-module")
    ap.add_argument("--adapter-func")
    ap.add_argument("--model-revision")
    ap.add_argument("--output", type=Path, help="Default: overwrite --ranks atomically")
    args = ap.parse_args()

    old = read_rank_archive(args.ranks)
    ranks = old["ranks"]

    prov = build_provenance(
        sessions_path=args.sessions,
        corpus_path=args.corpus_vectors,
        beliefs_path=args.beliefs,
        method=args.method,
        config_path=args.config,
        adapter_module=args.adapter_module,
        adapter_func=args.adapter_func,
        model_revision=args.model_revision,
        extra_run_metadata={"archive_migration": "legacy_rank_archive_rehydration"},
    )

    if ranks.ndim != 2:
        raise ValueError(f"rank matrix must be 2D, got {ranks.shape}")
    if ranks.shape[1] != len(prov["session_ids"]):
        raise ValueError(
            f"rank/session mismatch: ranks has {ranks.shape[1]} sessions, "
            f"session file has {len(prov['session_ids'])}"
        )

    metadata = {
        k: v
        for k, v in prov.items()
        if k not in {"session_ids", "target_indices"}
    }
    metadata["provenance_status"] = "rehydrated_from_declared_inputs"
    metadata["source_rank_archive"] = str(args.ranks)

    out = args.output or args.ranks
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp.npz")
    np.savez_compressed(
        tmp,
        ranks=ranks.astype(np.int64, copy=False),
        session_ids=prov["session_ids"],
        target_indices=prov["target_indices"],
        pairing_fingerprint=np.asarray(prov["pairing_fingerprint"]),
        evaluation_fingerprint=np.asarray(prov["evaluation_fingerprint"]),
        provenance_status=np.asarray("rehydrated_from_declared_inputs"),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    actual_tmp = tmp if tmp.exists() else Path(str(tmp) + ".npz")
    actual_tmp.replace(out)

    reread = read_rank_archive(out)
    if not np.array_equal(ranks, reread["ranks"]):
        raise RuntimeError("migration changed rank data -- refusing")
    print("UPGRADED:", out)
    print("rank data unchanged: PASS")
    print("pairing_fingerprint:", prov["pairing_fingerprint"])
    print("evaluation_fingerprint:", prov["evaluation_fingerprint"])
    print("provenance_status: rehydrated_from_declared_inputs")


if __name__ == "__main__":
    main()
