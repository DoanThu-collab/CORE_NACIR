"""Strict provenance helpers for NACIR evaluation archives.

The key design distinction is:

* pairing_fingerprint: proves two rank matrices refer to the same aligned
  evaluation problem (same session file, corpus file, session/target ordering,
  and embedding dimension). This MUST match for paired statistics.

* evaluation_fingerprint: identifies one concrete run. It additionally includes
  method/config/evidence/adapter metadata and is expected to differ across
  H0/Current/Persistent.

This avoids the old bug where hashing only (session_id, target_index) allowed
cross-corpus / cross-space comparisons to be falsely accepted.
"""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def sha256_file(path: str | Path | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _stable_sha256(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _declared_model_revision(method: str, adapter_module: str | None) -> str | None:
    """Read a declared adapter/model revision for non-H0 runs.

    H0 never invokes a belief encoder, so adapter identity is intentionally not
    material to the run. For Current/Persistent, public adapters must expose
    ``MODEL_REVISION`` (preferred) or ``MODEL_ID`` as a stable fallback.
    """
    if method == "h0" or not adapter_module:
        return None

    module = importlib.import_module(adapter_module)
    value = getattr(module, "MODEL_REVISION", None)
    if value is None:
        value = getattr(module, "MODEL_ID", None)
    return None if value is None else str(value)


def load_session_identity(path: str | Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Return session_ids, target_indices, embedding_dim from a standardized session file."""
    loaded = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(loaded, list) or not loaded:
        raise ValueError(f"{path}: expected non-empty list of sessions")

    session_ids: list[int] = []
    target_indices: list[int] = []
    embedding_dim: int | None = None

    for i, raw in enumerate(loaded):
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: session {i} is not a dict")
        sid = raw.get("session_id")
        target = raw.get("target_index")
        qv = raw.get("query_vectors")
        if not isinstance(sid, int) or not isinstance(target, int):
            raise ValueError(f"{path}: session {i} missing integer session_id/target_index")
        if not isinstance(qv, torch.Tensor) or qv.ndim != 2:
            raise ValueError(f"{path}: session {i} query_vectors must be [turns,D]")
        d = int(qv.shape[1])
        if embedding_dim is None:
            embedding_dim = d
        elif embedding_dim != d:
            raise ValueError(f"{path}: inconsistent query embedding dimensions")
        session_ids.append(sid)
        target_indices.append(target)

    assert embedding_dim is not None
    return (
        np.asarray(session_ids, dtype=np.int64),
        np.asarray(target_indices, dtype=np.int64),
        embedding_dim,
    )


def corpus_dimension(path: str | Path) -> int:
    loaded = torch.load(Path(path), map_location="cpu", weights_only=False)
    vectors = loaded.get("vectors") if isinstance(loaded, dict) else loaded
    if not isinstance(vectors, torch.Tensor) or vectors.ndim != 2:
        raise ValueError(f"{path}: corpus vectors must be [N,D]")
    return int(vectors.shape[1])


def build_provenance(
    *,
    sessions_path: str | Path,
    corpus_path: str | Path,
    beliefs_path: str | Path | None,
    method: str,
    config_path: str | Path | None = None,
    adapter_module: str | None = None,
    adapter_func: str | None = None,
    model_revision: str | None = None,
    extra_run_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    session_ids, target_indices, session_dim = load_session_identity(sessions_path)
    corpus_dim = corpus_dimension(corpus_path)
    if session_dim != corpus_dim:
        raise ValueError(
            f"session/corpus dimension mismatch: sessions D={session_dim}, corpus D={corpus_dim}"
        )

    method = str(method)
    if model_revision is None:
        model_revision = _declared_model_revision(method, adapter_module)

    sessions_sha = sha256_file(sessions_path)
    corpus_sha = sha256_file(corpus_path)
    beliefs_sha = sha256_file(beliefs_path)
    config_sha = sha256_file(config_path)

    pairing_payload = {
        "schema": "nacir-pairing-v2",
        "sessions_sha256": sessions_sha,
        "corpus_sha256": corpus_sha,
        "embedding_dim": session_dim,
        "session_ids_sha256": hashlib.sha256(session_ids.tobytes()).hexdigest(),
        "target_indices_sha256": hashlib.sha256(target_indices.tobytes()).hexdigest(),
        "num_sessions": int(len(session_ids)),
    }
    pairing_fp = _stable_sha256(pairing_payload)

    evaluation_payload = {
        "schema": "nacir-evaluation-v2",
        "pairing_fingerprint": pairing_fp,
        "method": method,
        "beliefs_sha256": beliefs_sha,
        "config_sha256": config_sha,
        "adapter_module": adapter_module if method != "h0" else None,
        "adapter_func": adapter_func if method != "h0" else None,
        "model_revision": model_revision,
        "extra_run_metadata": extra_run_metadata or {},
    }
    evaluation_fp = _stable_sha256(evaluation_payload)

    return {
        "provenance_schema": "nacir-evaluation-v2",
        "pairing_fingerprint": pairing_fp,
        "evaluation_fingerprint": evaluation_fp,
        "pairing_payload": pairing_payload,
        "evaluation_payload": evaluation_payload,
        "session_ids": session_ids,
        "target_indices": target_indices,
        "embedding_dim": session_dim,
        "sessions_sha256": sessions_sha,
        "corpus_sha256": corpus_sha,
        "beliefs_sha256": beliefs_sha,
        "config_sha256": config_sha,
        "method": method,
        "adapter_module": adapter_module if method != "h0" else None,
        "adapter_func": adapter_func if method != "h0" else None,
        "model_revision": model_revision,
    }


def npz_scalar_str(archive: np.lib.npyio.NpzFile, key: str) -> str | None:
    if key not in archive.files:
        return None
    value = archive[key]
    if np.asarray(value).size != 1:
        raise ValueError(f"{key} must be a scalar string in ranks.npz")
    return str(np.asarray(value).reshape(-1)[0])


def read_rank_archive(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    with np.load(p, allow_pickle=False) as z:
        rank_key = "ranks" if "ranks" in z.files else ("ranks_per_round" if "ranks_per_round" in z.files else None)
        if rank_key is None:
            raise ValueError(f"{p}: no ranks/ranks_per_round key")
        ranks = np.asarray(z[rank_key], dtype=np.int64)
        out = {
            "path": str(p),
            "keys": list(z.files),
            "ranks": ranks,
            "session_ids": np.asarray(z["session_ids"], dtype=np.int64) if "session_ids" in z.files else None,
            "target_indices": np.asarray(z["target_indices"], dtype=np.int64) if "target_indices" in z.files else None,
            "pairing_fingerprint": npz_scalar_str(z, "pairing_fingerprint"),
            "evaluation_fingerprint": npz_scalar_str(z, "evaluation_fingerprint"),
            "provenance_status": npz_scalar_str(z, "provenance_status"),
            "metadata_json": npz_scalar_str(z, "metadata_json"),
        }
    return out


def verify_pairing(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Strictly verify that paired statistics are legal."""
    reasons: list[str] = []

    if a["ranks"].shape != b["ranks"].shape:
        reasons.append(f"rank_shape_mismatch:{a['ranks'].shape}!={b['ranks'].shape}")

    if not a.get("pairing_fingerprint") or not b.get("pairing_fingerprint"):
        reasons.append("missing_pairing_fingerprint")
    elif a["pairing_fingerprint"] != b["pairing_fingerprint"]:
        reasons.append("pairing_fingerprint_mismatch")

    for key in ("session_ids", "target_indices"):
        av, bv = a.get(key), b.get(key)
        if av is None or bv is None:
            reasons.append(f"missing_{key}")
        elif not np.array_equal(av, bv):
            reasons.append(f"{key}_mismatch")

    return {
        "verified": not reasons,
        "reasons": reasons,
        "baseline_pairing_fingerprint": a.get("pairing_fingerprint"),
        "candidate_pairing_fingerprint": b.get("pairing_fingerprint"),
        "baseline_evaluation_fingerprint": a.get("evaluation_fingerprint"),
        "candidate_evaluation_fingerprint": b.get("evaluation_fingerprint"),
    }
