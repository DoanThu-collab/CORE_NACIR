"""NACIR- (Negative Only) conversational retrieval pipeline."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .config import NACIRMinusConfig
from .core.memory import ConceptMemory
from .interfaces import TextEncoder
from .schema import RetrievalSession, SessionOutput, TurnTrace


class NACIRMinusPipeline:
    """Training-free Negative-Only implementation (Eq 2 of the paper).

    Inputs are precomputed query vectors, corpus vectors, and per-turn signed
    beliefs. ``target_index`` is used only for evaluation traces and never affects
    memory or ranking.
    """

    def __init__(
        self,
        *,
        config: NACIRMinusConfig,
        corpus_vectors: torch.Tensor,
        text_encoder: TextEncoder,
        device: str | None = None,
    ) -> None:
        config.validate()
        if corpus_vectors.ndim != 2 or corpus_vectors.shape[0] < config.top_k:
            raise ValueError("corpus_vectors must have shape [N, D] with N >= top_k")
        if not torch.isfinite(corpus_vectors).all():
            raise ValueError("corpus_vectors must be finite")
        if bool((corpus_vectors.float().norm(dim=-1) <= 1e-8).any()):
            raise ValueError("corpus_vectors must be non-zero")
        
        self.config = config
        self.device = device or str(corpus_vectors.device)
        self.corpus_vectors = F.normalize(corpus_vectors.float().to(self.device), dim=-1)
        self.text_encoder = text_encoder

    def _query(self, vector: torch.Tensor) -> torch.Tensor:
        query = vector.float().to(self.device)
        if query.ndim != 1 or query.numel() != self.corpus_vectors.shape[1]:
            raise ValueError("query vector has an invalid shape")
        if not torch.isfinite(query).all() or float(query.norm()) <= 1e-8:
            raise ValueError("query vector must be finite and non-zero")
        return F.normalize(query, dim=0)

    def score(self, query_vector: torch.Tensor, memory: ConceptMemory) -> tuple[torch.Tensor, dict[str, Any]]:
        """Return NACIR- scores for one dialogue turn."""
        base_query = self._query(query_vector)
        
        # Eq 2: q− = norm(q0 − λ− * norm(Sum(ωj * vj)))
        q_minus = memory.synthesize(base_query)
        scores = q_minus @ self.corpus_vectors.T
        
        return scores, {"anchor_query": q_minus}

    @staticmethod
    def _rank_of(scores: torch.Tensor, target_index: int | None) -> int | None:
        if target_index is None:
            return None
        if not 0 <= target_index < scores.numel():
            raise ValueError("target_index is outside the corpus")
        order = torch.argsort(scores, descending=True, stable=True)
        return int(torch.nonzero(order == target_index, as_tuple=False)[0, 0].item())

    @torch.inference_mode()
    def run_session(self, session: RetrievalSession) -> SessionOutput:
        memory = ConceptMemory(self.config.memory, self.text_encoder)
        traces: list[TurnTrace] = []
        for turn in sorted(session.turns, key=lambda item: item.turn_index):
            memory.current_turn = turn.turn_index
            if turn.beliefs is not None:
                add_stats = memory.add_bundle(turn.beliefs, turn.turn_index)
            else:
                add_stats = {"added": 0, "updated": 0, "overridden": 0, "evicted": 0}
            if turn.query_vector is None:
                raise ValueError("paper release requires precomputed query vectors")
            
            scores, diagnostics = self.score(turn.query_vector, memory)
            ranked = torch.argsort(scores, descending=True, stable=True)
            final_rank = self._rank_of(scores, session.target_index)
            
            traces.append(
                TurnTrace(
                    turn_index=turn.turn_index,
                    accepted=True,
                    decision_mode="always_accept",
                    reject_rank=final_rank,
                    accept_rank=final_rank,
                    final_rank=final_rank,
                    top_k_indices=ranked[: self.config.top_k].cpu().tolist(),
                    diagnostics={
                        "memory": memory.diagnostics(),
                        "memory_add": add_stats,
                        "anchor": diagnostics,
                    },
                )
            )
        return SessionOutput(session.session_id, traces)
