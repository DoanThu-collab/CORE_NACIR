"""Loader for NACIR versioned belief artifacts used by the release evaluator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import Belief, BeliefBundle


def _canonical(text: str) -> str:
    return " ".join(text.lower().split())


class BeliefStore:
    """Index a complete version-2 belief artifact by dialogue and feedback turn.

    Belief artifacts index feedback turns from zero. Retrieval sessions include an
    initial query at turn zero, so a session turn uses artifact turn ``turn - 1``.
    """

    def __init__(self, turns: dict[int, dict[int, BeliefBundle]]) -> None:
        self._turns = turns

    @classmethod
    def from_path(cls, path: str | Path) -> "BeliefStore":
        source = Path(path)
        with source.open(encoding="utf-8") as handle:
            document = json.load(handle)
        if not isinstance(document, dict):
            raise ValueError("belief artifact must be a JSON object")
        if document.get("schema_version") != 2 or document.get("status") != "complete":
            raise ValueError("belief artifact must be a complete schema-version-2 file")
        if not isinstance(document.get("provenance"), dict):
            raise ValueError("belief artifact is missing provenance")
        if document.get("quality", {}).get("passed") is not True:
            raise ValueError("belief artifact does not have a passing quality report")
        dialogs = document.get("dialogs")
        if not isinstance(dialogs, list):
            raise ValueError("belief artifact dialogs must be a list")

        indexed: dict[int, dict[int, BeliefBundle]] = {}
        for expected_dialog_id, dialog in enumerate(dialogs):
            if not isinstance(dialog, dict) or dialog.get("dialog_id") != expected_dialog_id:
                raise ValueError("belief dialog identifiers must be contiguous")
            raw_turns = dialog.get("turns")
            if not isinstance(raw_turns, list):
                raise ValueError("belief dialogue turns must be a list")
            indexed_turns: dict[int, BeliefBundle] = {}
            for expected_turn, raw_turn in enumerate(raw_turns):
                if not isinstance(raw_turn, dict) or raw_turn.get("turn") != expected_turn:
                    raise ValueError("belief turn identifiers must be contiguous")
                positive = cls._parse_polarity(raw_turn.get("positives"), "positive")
                negative = cls._parse_polarity(raw_turn.get("negatives"), "negative")
                if {_canonical(item.attribute) for item in positive} & {
                    _canonical(item.attribute) for item in negative
                }:
                    raise ValueError("belief turn contains a cross-polarity conflict")
                question = raw_turn.get("question")
                answer = raw_turn.get("answer")
                if not isinstance(question, str) or not isinstance(answer, str):
                    raise ValueError("belief turn question and answer must be strings")
                indexed_turns[expected_turn] = BeliefBundle(
                    positive=positive,
                    negative=negative,
                    source_turn=expected_turn,
                    question=question,
                    answer=answer,
                )
            indexed[expected_dialog_id] = indexed_turns
        return cls(indexed)

    @staticmethod
    def _parse_polarity(raw: Any, polarity: str) -> list[Belief]:
        if not isinstance(raw, list):
            raise ValueError(f"{polarity} beliefs must be a list")
        output: list[Belief] = []
        for item in raw:
            if not isinstance(item, dict):
                raise ValueError(f"{polarity} belief entries must be objects")
            attribute = item.get("attribute")
            confidence = item.get("confidence")
            if not isinstance(attribute, str) or not attribute.strip():
                raise ValueError("belief attributes must be non-empty strings")
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
                raise ValueError("belief confidence must be numeric")
            output.append(
                Belief(
                    attribute=attribute,
                    confidence=float(confidence),
                    fact_type=polarity,
                    metadata={"evidence": item.get("evidence", "")},
                )
            )
        return output

    def bundle(self, dialogue_id: int, retrieval_turn: int) -> BeliefBundle:
        if retrieval_turn == 0:
            return BeliefBundle.empty()
        try:
            return self._turns[dialogue_id][retrieval_turn - 1]
        except KeyError as error:
            raise KeyError(
                f"missing beliefs for dialogue={dialogue_id}, retrieval_turn={retrieval_turn}"
            ) from error
