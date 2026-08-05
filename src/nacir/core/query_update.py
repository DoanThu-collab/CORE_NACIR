"""
NACIR++ — Batch Query Updater (Glue: Steps 2 + 3 + 4)
==========================================================
Connects Concept Memory, Orthogonal Projection, and Attention Masking
into a single BatchUpdater interface.

Design Philosophy — Plug-and-Play:
    NACIR++ does NOT replace the retrieval model or the query generation.
    It is a POST-HOC query modifier that sits between:
        PlugIR/ChatIR query → [NACIR++] → same retrieval model
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple
import logging
import copy

from .concept_memory import ConceptMemoryConfig, ConceptMemoryBoard
from .orthogonal_projection import orthogonal_purge
from .attention_masking import apply_enhanced_penalty
from ..config import NACIRPlusPlusConfig

logger = logging.getLogger(__name__)


class NACIRPlusPlusBatchUpdater:
    """
    Batch-level query update for NACIR++.

    Maintains one ConceptMemoryBoard per query in the batch.
    Orchestrates Steps 2 → 3 → 4 for the entire batch.

    Usage:
        updater = NACIRPlusPlusBatchUpdater(config, batch_size, encoder, device)

        for t in range(num_rounds):
            if t > 0:
                q_t = updater.update_query(q_text_t, beliefs_batch, t)
            else:
                q_t = q_text_t

            scores = q_t @ corpus_vectors.T

            if t > 0:
                scores = updater.apply_masking(scores, corpus_vectors)
    """

    def __init__(
        self,
        config: NACIRPlusPlusConfig,
        batch_size: int,
        encoder: Any,
        device: str,
    ):
        self.config = copy.deepcopy(config)
        self.B = batch_size
        self.encoder = encoder
        self.device = device

        mem_config = ConceptMemoryConfig(
            alpha=config.memory_alpha,
            beta=config.memory_beta,
            recency_decay=config.recency_decay,
            concept_match_threshold=config.concept_match_threshold,
            max_concepts=config.max_concepts,
        )

        self.boards: List[ConceptMemoryBoard] = [
            ConceptMemoryBoard(config=copy.deepcopy(mem_config), encoder=encoder)
            for _ in range(batch_size)
        ]

        self._neg_vectors_cache: List[Optional[torch.Tensor]] = [None] * batch_size
        self._neg_weights_cache: List[Optional[torch.Tensor]] = [None] * batch_size

        self._total_overrides = 0
        self._total_concepts = 0

    def update_query(
        self,
        q_text_batch: torch.Tensor,
        beliefs_batch: List[Dict],
        turn: int,
    ) -> torch.Tensor:
        B, D = q_text_batch.shape
        q_list = []

        all_positives: List[List[Dict]] = []
        all_negatives: List[List[Dict]] = []
        for b in range(B):
            beliefs = beliefs_batch[b] if b < len(beliefs_batch) else {}
            pos_beliefs = beliefs.get("positive_beliefs", [])
            neg_beliefs = beliefs.get("negative_beliefs", [])
            all_positives.append([
                {"attribute": b_item.get("attribute", ""),
                 "confidence": b_item.get("confidence", 0.7)}
                for b_item in pos_beliefs if b_item.get("attribute")
            ])
            all_negatives.append([
                {"attribute": b_item.get("attribute", ""),
                 "confidence": b_item.get("confidence", 0.7)}
                for b_item in neg_beliefs if b_item.get("attribute")
            ])

        flat_vecs = None
        offsets: List[Tuple[int, int]] = []
        if self.config.mode != "no_memory":
            flat_names: List[str] = []
            for b in range(B):
                start = len(flat_names)
                flat_names.extend(item["attribute"] for item in all_positives[b])
                flat_names.extend(item["attribute"] for item in all_negatives[b])
                offsets.append((start, len(flat_names)))

            if flat_names:
                with torch.no_grad():
                    flat_vecs = self.encoder(flat_names)
                    flat_vecs = F.normalize(flat_vecs, dim=-1)

        for b in range(B):
            q_t = q_text_batch[b]

            if self.config.mode != "no_memory":
                positives = all_positives[b]
                negatives = all_negatives[b]
                start, end = offsets[b]
                precomputed = [flat_vecs[j] for j in range(start, end)] if flat_vecs is not None else []

                stats = self.boards[b].add_concepts(positives, negatives, turn, precomputed_vectors=precomputed)
                self._total_overrides += stats.get("overridden", 0)
                self._total_concepts += stats.get("added", 0)

                q_t = self.boards[b].synthesize_query(q_t)

            if self.config.mode != "no_ortho":
                neg_vecs, neg_weights = self.boards[b].get_negative_vectors()
                if neg_vecs is not None and neg_vecs.shape[0] > 0:
                    q_t = orthogonal_purge(
                        q_t,
                        neg_vecs,
                        neg_weights,
                        strength=self.config.ortho_strength,
                        use_gram_schmidt=self.config.use_gram_schmidt,
                    )

            q_list.append(q_t)

        q_updated = torch.stack(q_list)

        self._update_neg_cache()

        return q_updated

    def _update_neg_cache(self):
        for b in range(self.B):
            neg_vecs, neg_weights = self.boards[b].get_negative_vectors()
            self._neg_vectors_cache[b] = neg_vecs
            self._neg_weights_cache[b] = neg_weights

    def apply_masking(
        self,
        scores: torch.Tensor,
        corpus_vectors: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.mode == "no_mask":
            return scores

        for b in range(self.B):
            neg_vecs = self._neg_vectors_cache[b]
            neg_weights = self._neg_weights_cache[b]

            if neg_vecs is not None and neg_vecs.shape[0] > 0:
                scores[b:b+1] = apply_enhanced_penalty(
                    scores[b:b+1],
                    corpus_vectors,
                    neg_vecs,
                    neg_weights,
                    tau=self.config.masking_threshold,
                    max_penalty=self.config.masking_penalty_weight,
                    soft=True,
                    temperature=0.1,
                )

        return scores

    def get_batch_stats(self) -> Dict[str, int]:
        return {
            "total_overrides": self._total_overrides,
            "total_concepts": self._total_concepts,
        }

    def get_memory_snapshots(self) -> List[List[Dict]]:
        return [board.get_memory_snapshot() for board in self.boards]
