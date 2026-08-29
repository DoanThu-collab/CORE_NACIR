"""Frozen structured-negative metadata resolver.

This module never uses retrieval outcomes. It maps runtime ChatIR beliefs
to the pre-retrieval frozen semantic/eligibility artifact.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .core.memory import canonicalize_concept
from .schema import Belief


ALLOWED_MODES = {
    "gate",
    "structure",
    "full",
}

EXPECTED_ARTIFACT_SHA256 = (
    "14b2d452336f1526fd92a2ff1159845cab5cf22aa49dcd26fe004592898d6840"
)


class StructuredNegativeResolver:
    def __init__(
        self,
        path: str | Path,
        *,
        verify_sha256: bool = True,
    ) -> None:
        self.path = Path(path)

        raw = self.path.read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()

        if verify_sha256:
            if self.sha256 != EXPECTED_ARTIFACT_SHA256:
                raise ValueError(
                    "Structured-negative artifact SHA mismatch:\n"
                    f"expected {EXPECTED_ARTIFACT_SHA256}\n"
                    f"actual   {self.sha256}"
                )

        artifact = json.loads(raw.decode("utf-8"))

        items = artifact.get("items")
        if not isinstance(items, list):
            raise ValueError(
                "structured artifact must contain an items list"
            )

        self.records: dict[
            tuple[int, int, int],
            dict[str, Any],
        ] = {}

        for item in items:
            dialog_id = int(item["dialog_id"])
            turn = int(item["turn"])
            negative_index = int(
                item.get("negative_index", 0)
            )

            key = (
                dialog_id,
                turn,
                negative_index,
            )

            if key in self.records:
                raise ValueError(
                    f"duplicate structured-negative key: {key}"
                )

            self.records[key] = item

        if len(self.records) != 6464:
            raise ValueError(
                f"expected 6464 structured negatives, "
                f"found {len(self.records)}"
            )

    def resolve(
        self,
        *,
        session_id: Any,
        source_turn: int,
        negative_index: int,
        belief: Belief,
    ) -> dict[str, Any]:
        if not isinstance(session_id, int):
            raise ValueError(
                "structured ChatIR resolver requires integer session_id"
            )

        key = (
            session_id,
            source_turn,
            negative_index,
        )

        if key not in self.records:
            raise KeyError(
                f"missing structured-negative record for {key}"
            )

        record = self.records[key]

        expected = canonicalize_concept(
            str(record["negative_attribute"])
        )
        actual = canonicalize_concept(
            belief.attribute
        )

        if expected != actual:
            raise ValueError(
                "belief/artifact mismatch for "
                f"{key}: runtime={belief.attribute!r}, "
                f"artifact={record['negative_attribute']!r}"
            )

        return record

    @staticmethod
    def is_actionable(
        record: dict[str, Any],
    ) -> bool:
        value = record.get("actionable_negative")

        if not isinstance(value, bool):
            raise ValueError(
                "artifact actionable_negative must be boolean"
            )

        return value

    @staticmethod
    def semantic_type(
        record: dict[str, Any],
    ) -> str | None:
        typing = record.get("typing")

        if not isinstance(typing, dict):
            return None

        value = typing.get("type")

        if value is None:
            return None

        value = str(value).strip().upper()

        allowed = {
            "EXISTENCE",
            "ATTRIBUTE",
            "RELATION",
            "GLOBAL",
        }

        if value not in allowed:
            raise ValueError(
                f"invalid semantic type: {value}"
            )

        return value
