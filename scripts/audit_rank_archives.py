#!/usr/bin/env python3
"""Audit paper-facing rank archives for strict provenance metadata.

This script is read-only. It fails when a discovered ``ranks.npz`` violates the
release provenance contract, including inconsistencies between scalar archive
fields and ``metadata_json``. Native evaluator outputs for Current/Persistent
must also record a concrete model revision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


REQUIRED_KEYS = {
    "ranks",
    "session_ids",
    "target_indices",
    "pairing_fingerprint",
    "evaluation_fingerprint",
    "provenance_status",
    "metadata_json",
}

VALID_STATUSES = {
    "emitted_by_evaluator",
    "rehydrated_from_declared_inputs",
}


def _scalar_text(z: np.lib.npyio.NpzFile, key: str) -> str | None:
    value = np.asarray(z[key])
    if value.size != 1:
        return None
    text = str(value.reshape(-1)[0]).strip()
    return text or None


def audit(path: Path) -> list[str]:
    problems: list[str] = []

    with np.load(path, allow_pickle=False) as z:
        keys = set(z.files)
        missing = sorted(REQUIRED_KEYS - keys)
        if missing:
            problems.append("missing_keys=" + ",".join(missing))
            return problems

        ranks = np.asarray(z["ranks"])
        session_ids = np.asarray(z["session_ids"])
        target_indices = np.asarray(z["target_indices"])

        if ranks.ndim != 2:
            problems.append(f"ranks_not_2d={ranks.shape}")
        elif ranks.shape[1] != len(session_ids):
            problems.append(
                f"rank_session_mismatch={ranks.shape[1]}!={len(session_ids)}"
            )

        if not np.issubdtype(ranks.dtype, np.integer):
            problems.append(f"ranks_not_integer={ranks.dtype}")
        if ranks.size and (ranks < 0).any():
            problems.append("negative_rank_value")

        if session_ids.ndim != 1:
            problems.append(f"session_ids_not_1d={session_ids.shape}")
        if target_indices.ndim != 1:
            problems.append(f"target_indices_not_1d={target_indices.shape}")
        if len(session_ids) != len(target_indices):
            problems.append(
                f"session_target_mismatch={len(session_ids)}!={len(target_indices)}"
            )
        if session_ids.ndim == 1 and len(np.unique(session_ids)) != len(session_ids):
            problems.append("duplicate_session_ids")

        scalar = {}
        for key in (
            "pairing_fingerprint",
            "evaluation_fingerprint",
            "provenance_status",
            "metadata_json",
        ):
            scalar[key] = _scalar_text(z, key)
            if scalar[key] is None:
                problems.append(f"invalid_scalar={key}")

        if problems and scalar.get("metadata_json") is None:
            return problems

        status = scalar.get("provenance_status")
        if status is not None and status not in VALID_STATUSES:
            problems.append(f"unknown_provenance_status={status}")

        metadata_text = scalar.get("metadata_json")
        if metadata_text is None:
            return problems

        try:
            metadata = json.loads(metadata_text)
        except json.JSONDecodeError:
            problems.append("metadata_json_invalid_json")
            return problems

        if not isinstance(metadata, dict):
            problems.append("metadata_json_not_object")
            return problems

        if metadata.get("provenance_status") != status:
            problems.append("metadata_status_mismatch")
        if metadata.get("pairing_fingerprint") != scalar.get("pairing_fingerprint"):
            problems.append("metadata_pairing_fingerprint_mismatch")
        if metadata.get("evaluation_fingerprint") != scalar.get("evaluation_fingerprint"):
            problems.append("metadata_evaluation_fingerprint_mismatch")

        pairing_payload = metadata.get("pairing_payload")
        evaluation_payload = metadata.get("evaluation_payload")
        if not isinstance(pairing_payload, dict):
            problems.append("missing_pairing_payload")
        if not isinstance(evaluation_payload, dict):
            problems.append("missing_evaluation_payload")
        else:
            if evaluation_payload.get("pairing_fingerprint") != scalar.get(
                "pairing_fingerprint"
            ):
                problems.append("evaluation_payload_pairing_mismatch")

        method = metadata.get("method")
        if method is None and isinstance(evaluation_payload, dict):
            method = evaluation_payload.get("method")

        # Rehydrated archives honestly describe provenance reconstructed from
        # declared historical inputs. Do not retroactively demand native runtime
        # metadata that was not emitted by the old evaluator.
        if status == "emitted_by_evaluator" and method not in (None, "h0"):
            revision = metadata.get("model_revision")
            if revision is None and isinstance(evaluation_payload, dict):
                revision = evaluation_payload.get("model_revision")
            if not isinstance(revision, str) or not revision.strip():
                problems.append("missing_model_revision_for_native_adapted_run")

    return problems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "roots",
        nargs="*",
        type=Path,
        default=[Path("runs_final"), Path("outputs")],
        help="Directories/files to scan recursively for ranks.npz.",
    )
    args = parser.parse_args()

    paths: list[Path] = []
    for root in args.roots:
        if root.is_file():
            if root.name == "ranks.npz":
                paths.append(root)
            continue
        if root.exists():
            paths.extend(sorted(root.rglob("ranks.npz")))

    paths = sorted(set(paths))
    if not paths:
        raise SystemExit("No ranks.npz archives found under requested roots")

    failures = 0
    for path in paths:
        problems = audit(path)
        if problems:
            failures += 1
            print(f"FAIL  {path}")
            for problem in problems:
                print(f"      {problem}")
        else:
            print(f"PASS  {path}")

    print(f"\narchives={len(paths)} failures={failures}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
