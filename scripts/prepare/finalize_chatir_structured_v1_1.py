#!/usr/bin/env python3
"""Finalize the frozen structured-negative diagnostic artifact (v1.1)."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


PARENT_SHA256 = (
    "2d438c7a9d5f402d5a8b850483118c2cbebfe17cb909df2267d2032bc510e776"
)

TYPE_FIX = {
    "288:3:0": "ATTRIBUTE",
    "304:3:0": "RELATION",
    "871:3:0": "ATTRIBUTE",
    "931:6:0": "ATTRIBUTE",
    "1221:7:1": "GLOBAL",
    "1461:2:0": "ATTRIBUTE",
    "1849:5:0": "ATTRIBUTE",
}

DROP_FIX = {
    "439:7:0",
    "1433:1:0",
}

ALLOWED_TYPES = {
    "EXISTENCE",
    "ATTRIBUTE",
    "RELATION",
    "GLOBAL",
}


def item_key(item: dict) -> str:
    return (
        f"{item['dialog_id']}:"
        f"{item['turn']}:"
        f"{item.get('negative_index', 0)}"
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply the frozen v1.1 structured-negative adjudications."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    if sha256(args.source) != PARENT_SHA256:
        raise ValueError(
            "source artifact SHA256 does not match the frozen v1 parent"
        )

    artifact = json.loads(args.source.read_text(encoding="utf-8"))
    items = artifact.get("items")
    if not isinstance(items, list) or len(items) != 6464:
        raise ValueError("expected 6464 structured negative items")

    seen_type: set[str] = set()
    seen_drop: set[str] = set()

    for item in items:
        key = item_key(item)

        if key in TYPE_FIX:
            if item.get("actionable_negative") is not True:
                raise ValueError(f"{key}: expected actionable negative")
            typing = item.get("typing")
            if not isinstance(typing, dict) or typing.get("type") is not None:
                raise ValueError(f"{key}: expected unresolved semantic type")

            typing["type"] = TYPE_FIX[key]
            typing["status"] = "REJECTED"
            item["semantic_type_source"] = (
                "manual_pre_retrieval_final_adjudication"
            )
            seen_type.add(key)

        if key in DROP_FIX:
            if item.get("actionable_negative") is not True:
                raise ValueError(f"{key}: expected actionable negative before drop")

            item["actionable_negative"] = False
            item["eligibility_source"] = (
                "manual_pre_retrieval_final_null_type_audit_drop"
            )
            typing = item.get("typing")
            if isinstance(typing, dict):
                typing["status"] = "UNRESOLVED"
                typing["type"] = None
            seen_drop.add(key)

    if seen_type != set(TYPE_FIX):
        raise ValueError("not all frozen type repairs were applied")
    if seen_drop != DROP_FIX:
        raise ValueError("not all frozen eligibility drops were applied")

    actionable = [item for item in items if item["actionable_negative"]]
    dropped = [item for item in items if not item["actionable_negative"]]

    actionable_none = [
        item
        for item in actionable
        if not isinstance(item.get("typing"), dict)
        or item["typing"].get("type") is None
    ]
    if actionable_none:
        raise ValueError(
            f"{len(actionable_none)} actionable negatives still have type=None"
        )

    for item in actionable:
        if item["typing"]["type"] not in ALLOWED_TYPES:
            raise ValueError(
                f"invalid actionable semantic type: {item['typing']['type']!r}"
            )

    type_counts = Counter(item["typing"]["type"] for item in actionable)
    drop_type_counts = Counter(
        item["typing"].get("type")
        if isinstance(item.get("typing"), dict)
        else None
        for item in dropped
    )
    source_counts = Counter(item["eligibility_source"] for item in items)

    artifact["items"] = items
    artifact["status"] = "FROZEN_PRE_RETRIEVAL_STRUCTURED_NEGATIVE_V1_1"
    artifact["final_null_type_audit"] = {
        "num_actionable_type_none_before": 9,
        "typed_actionable": 7,
        "newly_dropped_non_actionable": 2,
        "retrieval_results_used": False,
        "reason": (
            "Final pre-retrieval audit of actionable items lacking semantic type. "
            "Seven contained actionable propositions and were manually typed; "
            "two lacked evidence against the exact candidate concept and were dropped."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    output_sha = sha256(args.output)
    manifest = {
        "artifact": str(args.output),
        "sha256": output_sha,
        "parent_artifact": str(args.source),
        "parent_sha256": PARENT_SHA256,
        "num_negatives": len(items),
        "num_actionable": len(actionable),
        "num_non_actionable": len(dropped),
        "actionable_type_counts": dict(type_counts),
        "actionable_type_none": len(actionable_none),
        "final_manual_type_repairs": 7,
        "final_manual_eligibility_drops": 2,
        "retrieval_results_used_for_design_or_adjudication": False,
        "status": "FROZEN_BEFORE_STRUCTURED_RETRIEVAL",
    }

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("FINAL STRUCTURED NEGATIVE V1.1")
    print("total:", len(items))
    print("actionable:", len(actionable))
    print("non_actionable:", len(dropped))
    print("actionable types:", dict(type_counts))
    print("dropped types:", dict(drop_type_counts))
    print("eligibility sources:", dict(source_counts))
    print("sha256:", output_sha)
    print("saved:", args.output)
    print("manifest:", args.manifest)
    print("PASS")


if __name__ == "__main__":
    main()
