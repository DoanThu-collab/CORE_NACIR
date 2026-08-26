"""Target-free fusion of the rank-preserving H1 and recall-expanding APC routes."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import torch
import torch.nn.functional as F


class TrustWeightedDualRouteFusion:
    """Fuse aligned score vectors without learned or target-dependent weights.

    The legacy H1 route is assigned unit reliability. APC reliability is the
    product of its applied trust scale and its internal top-k stability, both
    already bounded to ``[0, 1]``. Normalizing the two reliabilities gives
    ``apc_weight = reliability / (1 + reliability)``, so APC can never receive
    more than half of the mass and H1 remains the rank-preserving anchor.
    """

    def __init__(self, *, eps: float = 1e-6) -> None:
        if not isinstance(eps, float) or not 0 < eps < 1:
            raise ValueError("dual-route fusion eps must be a float in (0, 1)")
        self.eps = eps

    def _standardize(
        self, scores: torch.Tensor
    ) -> Tuple[torch.Tensor, float, float]:
        mean = scores.mean()
        scale = scores.std(unbiased=False).clamp_min(self.eps)
        normalized = ((scores - mean) / scale).clamp(-12.0, 12.0)
        return normalized, float(mean), float(scale)

    @staticmethod
    def _topk_overlap(
        first: torch.Tensor, second: torch.Tensor, k: int = 10
    ) -> float:
        size = min(k, int(first.numel()))
        first_top = set(
            torch.topk(first, size, sorted=False).indices.cpu().tolist()
        )
        second_top = set(
            torch.topk(second, size, sorted=False).indices.cpu().tolist()
        )
        return len(first_top & second_top) / size

    def fuse(
        self,
        anchor_scores: torch.Tensor,
        apc_scores: torch.Tensor,
        apc_diagnostics: Mapping[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if (
            anchor_scores.ndim != 1
            or apc_scores.ndim != 1
            or anchor_scores.shape != apc_scores.shape
            or anchor_scores.numel() == 0
        ):
            raise ValueError(
                "dual-route fusion expects aligned non-empty scores [N]"
            )
        if not torch.isfinite(anchor_scores).all() or not torch.isfinite(
            apc_scores
        ).all():
            raise ValueError("dual-route fusion scores must be finite")

        trust_scale = apc_diagnostics.get("trust_scale")
        apc_stability = apc_diagnostics.get("top10_overlap")
        for name, value in (
            ("trust_scale", trust_scale),
            ("top10_overlap", apc_stability),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"APC diagnostic {name!r} must be numeric")
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"APC diagnostic {name!r} must be in [0, 1]")

        reliability = float(trust_scale) * float(apc_stability)
        apc_weight = reliability / (1.0 + reliability)
        anchor_weight = 1.0 - apc_weight
        anchor_normalized, anchor_mean, anchor_scale = self._standardize(
            anchor_scores
        )
        apc_normalized, apc_mean, apc_scale = self._standardize(apc_scores)
        fused = anchor_weight * anchor_normalized + apc_weight * apc_normalized
        if not torch.isfinite(fused).all():
            raise RuntimeError("dual-route fusion produced non-finite scores")

        diagnostics: Dict[str, Any] = {
            "mode": "dual_route_trust",
            "target_free": True,
            "anchor": "legacy_h1_projection_masking",
            "reliability": reliability,
            "anchor_weight": anchor_weight,
            "apc_weight": apc_weight,
            "weight_rule": "r/(1+r), r=trust_scale*apc_top10_stability",
            "apc_trust_scale": float(trust_scale),
            "apc_internal_top10_stability": float(apc_stability),
            "anchor_apc_top10_overlap": self._topk_overlap(
                anchor_scores, apc_scores
            ),
            "anchor_fused_top10_overlap": self._topk_overlap(
                anchor_scores, fused
            ),
            "apc_fused_top10_overlap": self._topk_overlap(apc_scores, fused),
            "standardized_score_cosine": float(
                F.cosine_similarity(
                    anchor_normalized.unsqueeze(0),
                    apc_normalized.unsqueeze(0),
                )[0]
            ),
            "anchor_score_mean": anchor_mean,
            "anchor_score_scale": anchor_scale,
            "apc_score_mean": apc_mean,
            "apc_score_scale": apc_scale,
        }
        return fused, diagnostics
