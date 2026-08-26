"""Evaluation helpers for the H0, H1, and F1 paper protocol."""

from __future__ import annotations

from typing import Literal

import torch

from .core.memory import ConceptMemory
from .pipeline import NACIRMinusPipeline
from .schema import RetrievalSession, SessionOutput, TurnTrace

RunMode = Literal["h0", "nacir"]


@torch.inference_mode()
def evaluate_session(pipeline: NACIRMinusPipeline, session: RetrievalSession, mode: RunMode) -> SessionOutput:
    """Evaluate one session with either H0 baseline or the NACIR- method.

    Ground-truth target indices are read after scores are formed solely to report
    ranks; they never affect the pipeline state.
    """

    if mode not in {"h0", "nacir"}:
        raise ValueError("mode must be h0 or nacir")
    memory = ConceptMemory(pipeline.config.memory, pipeline.text_encoder)
    traces: list[TurnTrace] = []
    for turn in sorted(session.turns, key=lambda item: item.turn_index):
        if turn.query_vector is None:
            raise ValueError("paper evaluation requires precomputed query vectors")
        memory.current_turn = turn.turn_index
        if mode != "h0" and turn.beliefs is not None:
            memory_add = memory.add_bundle(turn.beliefs, turn.turn_index)
        else:
            memory_add = {"added": 0, "updated": 0, "overridden": 0, "evicted": 0}

        base_query = pipeline._query(turn.query_vector)
        if mode == "h0":
            scores, diagnostics = base_query @ pipeline.corpus_vectors.T, {}
        else:
            scores, diagnostics = pipeline.score(turn.query_vector, memory)
        order = torch.argsort(scores, descending=True, stable=True)
        final_rank = pipeline._rank_of(scores, session.target_index)
        traces.append(
            TurnTrace(
                turn_index=turn.turn_index,
                accepted=True,
                decision_mode=mode,
                reject_rank=final_rank,
                accept_rank=final_rank,
                final_rank=final_rank,
                top_k_indices=order[: pipeline.config.top_k].cpu().tolist(),
                diagnostics={"memory": memory.diagnostics(), "memory_add": memory_add, **diagnostics},
            )
        )
    return SessionOutput(session.session_id, traces)


def rank_matrix(outputs: list[SessionOutput]) -> list[list[int]]:
    """Convert aligned outputs to the [round, session] rank matrix used by BRI."""

    if not outputs:
        raise ValueError("outputs must be non-empty")
    rounds = len(outputs[0].turns)
    if rounds < 1 or any(len(output.turns) != rounds for output in outputs):
        raise ValueError("outputs must contain aligned complete sessions")
    matrix: list[list[int]] = []
    for turn_index in range(rounds):
        ranks = [output.turns[turn_index].final_rank for output in outputs]
        if any(rank is None for rank in ranks):
            raise ValueError("rank metrics require a target_index for every session")
        matrix.append([int(rank) for rank in ranks])
    return matrix
