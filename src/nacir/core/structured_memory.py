"""Structured negative memory without altering canonical ConceptMemory."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F

from .memory import (
    ConceptMemory,
    MemoryEntry,
    canonicalize_concept,
)
from ..schema import BeliefBundle
from ..structured_negative import (
    ALLOWED_MODES,
    StructuredNegativeResolver,
)


class StructuredConceptMemory(ConceptMemory):
    """
    Persistent negative memory with optional:

    gate:
        remove non-actionable negative evidence.

    structure:
        represent RELATION negatives by an anchor-orthogonal
        relation residual.

    full:
        gate + structure.

    Canonical ConceptMemory itself remains untouched.
    """

    def __init__(
        self,
        config,
        encoder,
        *,
        resolver: StructuredNegativeResolver,
        mode: str,
        session_id: Any,
    ) -> None:
        super().__init__(config, encoder)

        if mode not in ALLOWED_MODES:
            raise ValueError(
                f"invalid structured mode {mode!r}; "
                f"expected one of {sorted(ALLOWED_MODES)}"
            )

        self.resolver = resolver
        self.mode = mode
        self.session_id = session_id

        self.structured_stats = {
            "seen": 0,
            "gated": 0,
            "direct": 0,
            "relation_residual": 0,
        }

    def clone(self) -> "StructuredConceptMemory":
        cloned = StructuredConceptMemory(
            self.config,
            self.encoder,
            resolver=self.resolver,
            mode=self.mode,
            session_id=self.session_id,
        )

        import copy

        cloned.entries = copy.deepcopy(self.entries)
        cloned.current_turn = self.current_turn
        cloned.override_log = copy.deepcopy(
            self.override_log
        )
        cloned.structured_stats = copy.deepcopy(
            self.structured_stats
        )

        return cloned

    @staticmethod
    def _normalize_vector(
        vector: torch.Tensor,
    ) -> torch.Tensor:
        vector = vector.float()

        if vector.ndim != 1:
            raise ValueError(
                "structured vector must be 1D"
            )

        if not torch.isfinite(vector).all():
            raise ValueError(
                "structured vector must be finite"
            )

        if float(vector.norm()) <= 1e-8:
            raise ValueError(
                "structured vector must be non-zero"
            )

        return F.normalize(vector, dim=0)

    def _relation_vector(
        self,
        record: dict,
    ) -> torch.Tensor:
        typing = record["typing"]

        subject = str(
            typing.get("subject") or ""
        ).strip()
        predicate = str(
            typing.get("predicate") or ""
        ).strip()
        obj = str(
            typing.get("object") or ""
        ).strip()

        if not subject or not obj:
            raise ValueError(
                "RELATION record requires subject and object"
            )

        relation_text = " ".join(
            x
            for x in (
                subject,
                predicate,
                obj,
            )
            if x
        )

        # IMPORTANT:
        # This is exactly the formulation audited before retrieval.
        texts = [
            relation_text,
            subject,
            obj,
        ]

        vectors = self.encoder.encode(texts)

        if (
            vectors.ndim != 2
            or vectors.shape[0] != 3
            or vectors.shape[1] < 1
        ):
            raise ValueError(
                "invalid relation embedding batch"
            )

        if not torch.isfinite(vectors).all():
            raise ValueError(
                "non-finite relation embeddings"
            )

        vectors = F.normalize(
            vectors.float(),
            dim=-1,
        )

        e_rel = vectors[0]
        e_s = vectors[1]
        e_o = vectors[2]

        # A: [D, 2]
        A = torch.stack(
            [e_s, e_o],
            dim=1,
        )

        projection = (
            A
            @ (
                torch.linalg.pinv(A)
                @ e_rel
            )
        )

        residual = e_rel - projection

        if float(residual.norm()) <= 1e-8:
            raise ValueError(
                "relation residual collapsed to zero"
            )

        residual = F.normalize(
            residual,
            dim=0,
        )

        # Numerical invariant from the frozen geometry audit.
        cos_s = float(
            torch.abs(
                torch.dot(residual, e_s)
            )
        )
        cos_o = float(
            torch.abs(
                torch.dot(residual, e_o)
            )
        )

        if cos_s > 1e-4 or cos_o > 1e-4:
            raise RuntimeError(
                "relation residual is not orthogonal "
                f"to anchors: subject={cos_s}, object={cos_o}"
            )

        return residual

    def add_bundle(
        self,
        bundle: BeliefBundle,
        turn: int,
    ) -> Dict[str, int]:
        self.current_turn = turn

        stats = {
            "added": 0,
            "updated": 0,
            "overridden": 0,
            "evicted": 0,
            "gated": 0,
            "relation_residual": 0,
            "direct": 0,
        }

        beliefs = list(bundle.negative)

        if not beliefs:
            return stats

        if bundle.source_turn is None:
            raise ValueError(
                "structured memory requires "
                "BeliefBundle.source_turn"
            )

        selected = []

        for negative_index, belief in enumerate(
            beliefs
        ):
            record = self.resolver.resolve(
                session_id=self.session_id,
                source_turn=bundle.source_turn,
                negative_index=negative_index,
                belief=belief,
            )

            self.structured_stats["seen"] += 1

            actionable = (
                self.resolver.is_actionable(
                    record
                )
            )

            if (
                self.mode in {"gate", "full"}
                and not actionable
            ):
                stats["gated"] += 1
                self.structured_stats["gated"] += 1
                continue

            semantic_type = (
                self.resolver.semantic_type(
                    record
                )
            )

            use_relation = (
                self.mode in {"structure", "full"}
                and semantic_type == "RELATION"
            )

            selected.append(
                (
                    belief,
                    record,
                    use_relation,
                )
            )

        if not selected:
            return stats

        # Direct embeddings are intentionally encoded exactly
        # from belief.attribute, as in canonical ConceptMemory.
        direct_positions = [
            i
            for i, (_, _, use_relation)
            in enumerate(selected)
            if not use_relation
        ]

        direct_vectors = {}

        if direct_positions:
            texts = [
                selected[i][0].attribute
                for i in direct_positions
            ]

            vectors = self.encoder.encode(texts)

            if (
                vectors.ndim != 2
                or vectors.shape[0] != len(texts)
                or vectors.shape[1] < 1
            ):
                raise ValueError(
                    "Text encoder returned an invalid "
                    "direct embedding batch"
                )

            if not torch.isfinite(vectors).all():
                raise ValueError(
                    "Text encoder returned non-finite "
                    "direct embeddings"
                )

            vectors = F.normalize(
                vectors.float(),
                dim=-1,
            )

            for pos, vector in zip(
                direct_positions,
                vectors,
            ):
                direct_vectors[pos] = vector

        prepared = []

        for i, (
            belief,
            record,
            use_relation,
        ) in enumerate(selected):

            if use_relation:
                vector = self._relation_vector(
                    record
                )
                stats["relation_residual"] += 1
                self.structured_stats[
                    "relation_residual"
                ] += 1
            else:
                vector = direct_vectors[i]
                stats["direct"] += 1
                self.structured_stats["direct"] += 1

            prepared.append(
                (
                    "negative",
                    belief,
                    vector,
                )
            )

        if self.entries:
            dim = next(
                iter(self.entries.values())
            ).vector.numel()

            for _, _, vector in prepared:
                if vector.numel() != dim:
                    raise ValueError(
                        "Concept embedding dimension "
                        "changed within a session"
                    )

        # Keep canonical concept identity/update semantics.
        for polarity, belief, vector in prepared:
            matched = self._match(
                belief.attribute,
                vector,
            )

            if matched is None:
                key = canonicalize_concept(
                    belief.attribute
                )

                self.entries[key] = MemoryEntry(
                    name=belief.attribute,
                    polarity=polarity,
                    vector=vector,
                    confidence=belief.confidence,
                    turn_created=turn,
                    turn_updated=turn,
                )

                stats["added"] += 1
                continue

            entry = self.entries[matched]

            if entry.polarity == polarity:
                entry.confidence = max(
                    entry.confidence,
                    belief.confidence,
                )
                entry.vector = vector
                entry.turn_updated = turn

                stats["updated"] += 1

            else:
                # Retained only for exact behavioral compatibility
                # with ConceptMemory. Negative-only canonical runs
                # should not normally enter this branch.
                old_polarity = entry.polarity

                entry.polarity = polarity
                entry.confidence = min(
                    1.0,
                    max(
                        entry.confidence,
                        belief.confidence,
                    )
                    + self.config.override_boost,
                )
                entry.vector = vector
                entry.turn_updated = turn
                entry.override_count += 1

                self.override_log.append(
                    {
                        "concept": entry.name,
                        "old_polarity": old_polarity,
                        "new_polarity": polarity,
                        "turn": turn,
                    }
                )

                stats["overridden"] += 1

        if (
            len(self.entries)
            > self.config.max_concepts
        ):
            ordered = sorted(
                self.entries,
                key=lambda key: (
                    self.entries[key].turn_updated
                ),
            )

            excess = (
                len(self.entries)
                - self.config.max_concepts
            )

            for key in ordered[:excess]:
                del self.entries[key]
                stats["evicted"] += 1

        return stats

    def diagnostics(self) -> dict:
        out = super().diagnostics()

        out.update(
            {
                "structured_mode": self.mode,
                "structured_seen": (
                    self.structured_stats["seen"]
                ),
                "structured_gated": (
                    self.structured_stats["gated"]
                ),
                "structured_direct": (
                    self.structured_stats["direct"]
                ),
                "structured_relation_residual": (
                    self.structured_stats[
                        "relation_residual"
                    ]
                ),
            }
        )

        return out
