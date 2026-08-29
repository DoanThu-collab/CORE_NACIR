"""Structured NACIR ablation pipeline.

Canonical NACIRMinusPipeline remains untouched.
"""

from __future__ import annotations

import torch

from .core.structured_memory import (
    StructuredConceptMemory,
)
from .pipeline import NACIRMinusPipeline
from .schema import (
    RetrievalSession,
    SessionOutput,
    TurnTrace,
)
from .structured_negative import (
    ALLOWED_MODES,
    StructuredNegativeResolver,
)


class StructuredNACIRMinusPipeline(
    NACIRMinusPipeline
):
    def __init__(
        self,
        *,
        structured_resolver:
            StructuredNegativeResolver,
        structured_mode: str,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        if structured_mode not in ALLOWED_MODES:
            raise ValueError(
                f"invalid structured mode "
                f"{structured_mode!r}"
            )

        self.structured_resolver = (
            structured_resolver
        )
        self.structured_mode = (
            structured_mode
        )

    @torch.inference_mode()
    def run_session(
        self,
        session: RetrievalSession,
    ) -> SessionOutput:

        memory = StructuredConceptMemory(
            self.config.memory,
            self.text_encoder,
            resolver=self.structured_resolver,
            mode=self.structured_mode,
            session_id=session.session_id,
        )

        traces: list[TurnTrace] = []

        for turn in sorted(
            session.turns,
            key=lambda item: item.turn_index,
        ):
            memory.current_turn = turn.turn_index

            if turn.beliefs is not None:
                add_stats = memory.add_bundle(
                    turn.beliefs,
                    turn.turn_index,
                )
            else:
                add_stats = {
                    "added": 0,
                    "updated": 0,
                    "overridden": 0,
                    "evicted": 0,
                    "gated": 0,
                    "relation_residual": 0,
                    "direct": 0,
                }

            if turn.query_vector is None:
                raise ValueError(
                    "paper release requires "
                    "precomputed query vectors"
                )

            scores, diagnostics = self.score(
                turn.query_vector,
                memory,
            )

            ranked = torch.argsort(
                scores,
                descending=True,
                stable=True,
            )

            final_rank = self._rank_of(
                scores,
                session.target_index,
            )

            traces.append(
                TurnTrace(
                    turn_index=turn.turn_index,
                    accepted=True,
                    decision_mode=(
                        "always_accept_"
                        + self.structured_mode
                    ),
                    reject_rank=final_rank,
                    accept_rank=final_rank,
                    final_rank=final_rank,
                    top_k_indices=ranked[
                        : self.config.top_k
                    ].cpu().tolist(),
                    diagnostics={
                        "memory": (
                            memory.diagnostics()
                        ),
                        "memory_add": add_stats,
                        "anchor": diagnostics,
                    },
                )
            )

        return SessionOutput(
            session_id=session.session_id,
            turns=traces,
        )
