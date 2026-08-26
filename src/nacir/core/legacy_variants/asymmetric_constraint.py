"""Asymmetric positive-proposal and negative-constraint retrieval."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

from ..config import AsymmetricConstraintConfig
from .memory import ConceptMemory


class AsymmetricProposalConstraintRouter:
    """Prune a positive-memory proposal with complementary negative evidence.

    Candidate discovery happens globally before this router. The router only
    reorders the proposal top-M, so negative evidence cannot hallucinate a new
    candidate or destroy proposal-pool provenance.
    """

    def __init__(self, config: AsymmetricConstraintConfig) -> None:
        config.validate()
        if config.mode == "off":
            raise ValueError(
                "AsymmetricProposalConstraintRouter cannot be constructed in off mode"
            )
        self.config = config

    @staticmethod
    def _kl_divergence(
        logits: torch.Tensor, reference_logits: torch.Tensor
    ) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=0)
        log_q = F.log_softmax(reference_logits, dim=0)
        return torch.sum(log_p.exp() * (log_p - log_q))

    def _negative_adjustment(
        self,
        candidate_vectors: torch.Tensor,
        memory: ConceptMemory,
    ) -> Tuple[torch.Tensor, Dict[str, int]]:
        adjustment = torch.zeros(
            candidate_vectors.shape[0],
            dtype=candidate_vectors.dtype,
            device=candidate_vectors.device,
        )
        positive_vectors, _ = memory.vectors("positive")
        negative_vectors, negative_weights = memory.vectors("negative")
        counts = {
            "positive_total": (
                0 if positive_vectors is None else int(positive_vectors.shape[0])
            ),
            "negative_total": (
                0 if negative_vectors is None else int(negative_vectors.shape[0])
            ),
            "negative_used": 0,
            "negative_skipped": 0,
        }
        if self.config.mode == "proposal" or negative_vectors is None:
            return adjustment, counts

        negative_vectors = negative_vectors.to(
            candidate_vectors.device, dtype=candidate_vectors.dtype
        )
        negative_weights = negative_weights.to(
            candidate_vectors.device, dtype=candidate_vectors.dtype
        )
        similarities = negative_vectors @ candidate_vectors.T
        quantiles = torch.quantile(
            similarities.float(),
            torch.tensor([0.10, 0.50, 0.90], device=similarities.device),
            dim=1,
        ).to(similarities.dtype)
        spread = quantiles[2] - quantiles[0]
        valid = torch.isfinite(spread) & (spread >= self.config.min_spread)
        counts["negative_used"] = int(valid.sum())
        counts["negative_skipped"] = int((~valid).sum())
        if not bool(valid.any()):
            return adjustment, counts

        selected = similarities[valid]
        centers = quantiles[1, valid, None]
        scales = (0.5 * spread[valid, None]).clamp_min(self.config.eps)
        standardized = ((selected - centers) / scales).clamp(-8.0, 8.0)
        log_complement = F.logsigmoid(-standardized)
        log_complement = log_complement - log_complement.mean(
            dim=1, keepdim=True
        )
        adjustment.add_(
            (
                self.config.negative_strength
                * negative_weights[valid, None]
                * log_complement
            ).sum(dim=0)
        )
        return adjustment, counts

    def route(
        self,
        proposal_scores: torch.Tensor,
        corpus_vectors: torch.Tensor,
        memory: ConceptMemory,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if proposal_scores.ndim != 1 or corpus_vectors.ndim != 2:
            raise ValueError(
                "asymmetric routing expects scores [N] and vectors [N, D]"
            )
        if proposal_scores.shape[0] != corpus_vectors.shape[0]:
            raise ValueError("asymmetric score/vector corpus sizes differ")
        if not torch.isfinite(proposal_scores).all() or not torch.isfinite(
            corpus_vectors
        ).all():
            raise ValueError("asymmetric routing inputs must be finite")

        if self.config.mode == "proposal":
            positive_vectors, _ = memory.vectors("positive")
            negative_vectors, _ = memory.vectors("negative")
            diagnostics: Dict[str, Any] = {
                "mode": self.config.mode,
                "proposal": "positive_memory_query",
                "candidate_k": min(
                    self.config.candidate_k, int(proposal_scores.numel())
                ),
                "positive_total": (
                    0
                    if positive_vectors is None
                    else int(positive_vectors.shape[0])
                ),
                "negative_total": (
                    0
                    if negative_vectors is None
                    else int(negative_vectors.shape[0])
                ),
                "negative_used": 0,
                "negative_skipped": 0,
                "evidence_l1": 0.0,
                "evidence_max_abs": 0.0,
                "raw_kl": 0.0,
                "applied_kl": 0.0,
                "max_kl": None,
                "trust_scale": 1.0,
                "entropy_before": None,
                "entropy_after": None,
                "top10_overlap": 1.0,
                "pool_preserved": True,
            }
            return proposal_scores.clone(), diagnostics

        pool_size = min(self.config.candidate_k, int(proposal_scores.numel()))
        candidate_scores, candidate_indices = torch.topk(
            proposal_scores, pool_size, largest=True, sorted=True
        )
        candidate_vectors = corpus_vectors[candidate_indices]
        base_logits = candidate_scores / self.config.posterior_temperature
        evidence, counts = self._negative_adjustment(candidate_vectors, memory)

        raw_logits = base_logits + evidence
        raw_kl = float(self._kl_divergence(raw_logits, base_logits))
        trust_scale = 1.0
        if (
            self.config.mode in {"constraint_trust", "dual_route_trust"}
            and raw_kl > self.config.max_kl
            and bool(torch.any(evidence != 0))
        ):
            low, high = 0.0, 1.0
            for _ in range(24):
                middle = 0.5 * (low + high)
                kl = float(
                    self._kl_divergence(
                        base_logits + middle * evidence, base_logits
                    )
                )
                if kl <= self.config.max_kl:
                    low = middle
                else:
                    high = middle
            trust_scale = low

        posterior_logits = base_logits + trust_scale * evidence
        applied_kl = float(self._kl_divergence(posterior_logits, base_logits))
        before = F.softmax(base_logits, dim=0)
        after = F.softmax(posterior_logits, dim=0)
        entropy_before = float(
            -(before * before.clamp_min(self.config.eps).log()).sum()
        )
        entropy_after = float(
            -(after * after.clamp_min(self.config.eps).log()).sum()
        )

        routed = proposal_scores.clone()
        posterior_order = torch.argsort(
            posterior_logits, descending=True, stable=True
        )
        if bool(torch.any(evidence != 0)):
            ordered_candidates = candidate_indices[posterior_order]
            ordered_values = candidate_scores
            routed[ordered_candidates] = ordered_values

        overlap_k = min(10, pool_size)
        before_top = set(candidate_indices[:overlap_k].cpu().tolist())
        after_top = set(
            candidate_indices[posterior_order[:overlap_k]].cpu().tolist()
        )
        diagnostics: Dict[str, Any] = {
            "mode": self.config.mode,
            "proposal": "positive_memory_query",
            "candidate_k": pool_size,
            **counts,
            "evidence_l1": float(evidence.abs().sum()),
            "evidence_max_abs": float(evidence.abs().max()),
            "raw_kl": raw_kl,
            "applied_kl": applied_kl,
            "max_kl": (
                self.config.max_kl
                if self.config.mode in {"constraint_trust", "dual_route_trust"}
                else None
            ),
            "trust_scale": trust_scale,
            "entropy_before": entropy_before,
            "entropy_after": entropy_after,
            "top10_overlap": len(before_top & after_top) / overlap_k,
            "pool_preserved": True,
        }
        return routed, diagnostics
