"""
NACIR++ — Query Update Pipeline (Glue: Steps 2 + 3 + 4)
==========================================================
Connects Concept Memory, Orthogonal Projection, and Attention Masking
into a single BatchUpdater interface consumed by run_nacir_plus.py.

Design Philosophy — Plug-and-Play:
    NACIR++ does NOT replace the retrieval model or the query generation.
    It is a POST-HOC query modifier that sits between:
        PlugIR/ChatIR query → [NACIR++] → same retrieval model

    This means:
    - Same corpus embeddings
    - Same text encoder
    - Same eval metrics (Hits@K, Recall@K, BRI — matching PlugIR exactly)
    - Only the query vector and scores are modified

Architecture:
    beliefs_batch → Step 2 (Concept Memory + Synthesize)
                  → Step 3 (Orthogonal Projection)
                  → Step 4 (Attention Masking on scores)
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import logging
import copy

from core.concept_memory import (
    ConceptMemoryConfig,
    ConceptMemoryBoard,
)
from core.orthogonal_projection import orthogonal_purge
from core.attention_masking import apply_enhanced_penalty

logger = logging.getLogger(__name__)


# ============================================================
# Config
# ============================================================

@dataclass
class NACIRPlusPlusConfig:
    """
    Unified configuration for all NACIR++ steps.

    Ablation modes:
        "full"       — all steps enabled
        "no_ortho"   — disable Step 3 (Orthogonal Projection)
        "no_mask"    — disable Step 4 (Attention Masking)
        "no_memory"  — disable Step 2 (Concept Memory), use raw PlugIR query
    """
    # Step 2: Concept Memory
    memory_alpha: float = 0.30          # Positive blend weight
    memory_beta: float = 0.15           # Negative blend weight
    recency_decay: float = 0.1          # Recency decay for older concepts
    concept_match_threshold: float = 0.85  # Cosine sim threshold for auto-override
    max_concepts: int = 50              # Max concepts per dialog memory

    # Step 3: Orthogonal Projection
    positive_blend_alpha: float = 0.30  # Positive concept blend before projection
    ortho_strength: float = 1.0         # Projection removal strength [0, 1]
    use_gram_schmidt: bool = True       # Gram-Schmidt before projection

    # Step 4: Masking / Penalty
    masking_penalty_weight: float = 0.15  # Max penalty per image
    masking_threshold: float = 0.20       # Similarity threshold for penalty

    # Ablation mode
    mode: str = "full"

    def __repr__(self):
        return (
            f"NACIRPlusPlusConfig(mode={self.mode}, "
            f"α_mem={self.memory_alpha}, β_mem={self.memory_beta}, "
            f"α_pos={self.positive_blend_alpha}, "
            f"ortho_str={self.ortho_strength}, "
            f"mask_w={self.masking_penalty_weight}, "
            f"mask_τ={self.masking_threshold})"
        )


# ============================================================
# Batch Updater — Main Interface
# ============================================================

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
        self.config = config
        self.B = batch_size
        self.encoder = encoder
        self.device = device

        # Create memory config from unified config
        mem_config = ConceptMemoryConfig(
            alpha=config.memory_alpha,
            beta=config.memory_beta,
            recency_decay=config.recency_decay,
            concept_match_threshold=config.concept_match_threshold,
            max_concepts=config.max_concepts,
        )

        # One memory board per query
        self.boards: List[ConceptMemoryBoard] = [
            ConceptMemoryBoard(config=copy.deepcopy(mem_config), encoder=encoder)
            for _ in range(batch_size)
        ]

        # Track accumulated negative vectors per query for masking (Step 4)
        self._neg_vectors_cache: List[Optional[torch.Tensor]] = [None] * batch_size
        self._neg_weights_cache: List[Optional[torch.Tensor]] = [None] * batch_size

        # Stats
        self._total_overrides = 0
        self._total_concepts = 0

    def update_query(
        self,
        q_text_batch: torch.Tensor,   # [B, D] — PlugIR text embedding for this round
        beliefs_batch: List[Dict],     # [B] — beliefs per query
        turn: int,
    ) -> torch.Tensor:
        """
        Steps 2 + 3: Update query vectors using Concept Memory + Orthogonal Projection.

        Args:
            q_text_batch:  [B, D] raw PlugIR text embeddings
            beliefs_batch: list of B dicts, each with:
                           {"positive_beliefs": [...], "negative_beliefs": [...]}
            turn:          current dialog turn (1-indexed for first QA round)

        Returns:
            q_updated: [B, D] updated query vectors
        """
        B, D = q_text_batch.shape
        q_list = []

        for b in range(B):
            q_t = q_text_batch[b]
            beliefs = beliefs_batch[b] if b < len(beliefs_batch) else {}

            pos_beliefs = beliefs.get("positive_beliefs", [])
            neg_beliefs = beliefs.get("negative_beliefs", [])

            # ── Step 2: Concept Memory ──
            if self.config.mode != "no_memory":
                # Convert beliefs to concept format
                positives = [
                    {"attribute": b_item.get("attribute", ""),
                     "confidence": b_item.get("confidence", 0.7)}
                    for b_item in pos_beliefs if b_item.get("attribute")
                ]
                negatives = [
                    {"attribute": b_item.get("attribute", ""),
                     "confidence": b_item.get("confidence", 0.7)}
                    for b_item in neg_beliefs if b_item.get("attribute")
                ]

                # Add to memory board
                stats = self.boards[b].add_concepts(positives, negatives, turn)
                self._total_overrides += stats.get("overridden", 0)
                self._total_concepts += stats.get("added", 0)

                # Synthesize query from memory
                q_t = self.boards[b].synthesize_query(q_t)

            # ── Step 3: Orthogonal Projection ──
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

        # Cache negative vectors for Step 4 (masking)
        self._update_neg_cache()

        return q_updated

    def _update_neg_cache(self):
        """Cache negative vectors from memory boards for masking step."""
        for b in range(self.B):
            neg_vecs, neg_weights = self.boards[b].get_negative_vectors()
            self._neg_vectors_cache[b] = neg_vecs
            self._neg_weights_cache[b] = neg_weights

    def apply_masking(
        self,
        scores: torch.Tensor,       # [B, N]
        corpus_vectors: torch.Tensor,  # [N, D]
    ) -> torch.Tensor:
        """
        Step 4: Apply enhanced penalty scoring using negative concepts.

        Uses Global Fallback mode (Mode 2) since we only have global
        corpus embeddings, not patch-level features.

        Args:
            scores:         [B, N] current similarity scores
            corpus_vectors: [N, D] corpus image embeddings

        Returns:
            scores:         [B, N] adjusted scores
        """
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
        """Return accumulated statistics for this batch."""
        return {
            "total_overrides": self._total_overrides,
            "total_concepts": self._total_concepts,
        }

    def get_memory_snapshots(self) -> List[List[Dict]]:
        """Get memory snapshots for all queries in batch (for debugging)."""
        return [board.get_memory_snapshot() for board in self.boards]
