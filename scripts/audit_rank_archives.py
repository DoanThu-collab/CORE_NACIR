#!/usr/bin/env python3
"""Audit paper-facing rank archives for strict provenance metadata.

This script does not modify archives. It fails if any discovered ``ranks.npz``
under the requested roots lacks the canonical provenance contract.
"""

from __future__ import annotations

import argparse
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

        if len(session_ids) != len(target_indices):
            problems.append(
                f"session_target_mismatch={len(session_ids)}!={len(target_indices)}"
            )

        for key in (
            "pairing_fingerprint",
            "evaluation_fingerprint",
            "provenance_status",
            "metadata_json",
        ):
            value = np.asarray(z[key])
            if value.size != 1 or not str(value.reshape(-1)[0]).strip():
                problems.append(f"invalid_scalar={key}")

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
