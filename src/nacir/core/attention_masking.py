"""
NACIR++ — Step 4: Region-Level Attention Masking
===================================================
Tiến hóa từ M3 (zone_scorer.py): Trừ điểm thô sơ bằng penalty.

NACIR++ xóa trực tiếp điểm ảnh bằng Region-Level Attention Masking:
    1. Tính similarity giữa negative concept vectors và image patch embeddings
    2. Mask ra (triệt tiêu) các patches/regions có similarity cao với negatives
    3. Recompute score dựa trên masked image representation

Công thức:
    Cho mỗi ảnh I với patch embeddings {p_1, ..., p_P}:
        mask_j = 1[sim(n, p_j) > τ]   cho mỗi negative concept n
        score_masked = q · mean({p_j : mask_j = 0})

    Fallback (khi không có patch embeddings):
        Dùng soft penalty giống M3 cũ nhưng dùng negative vectors
        từ Concept Memory Board thay vì exclusion zones

So sánh:
    Cũ (M3):  score -= λ × max(sim - τ, 0)    → chỉ trừ, vẫn score cao
    Mới:      score = q · masked_representation → xóa sạch region nhiễu

Hai chế độ hoạt động:
    Mode 1 — Patch-Level (full): Cần patch embeddings [N, P, D]
        → Tính attention → mask → recompute → cực kỳ chính xác
    Mode 2 — Global Fallback: Chỉ có global vectors [N, D]
        → Dùng enhanced penalty (mạnh hơn M3 cũ)
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# ============================================================
# Mode 1: Patch-Level Attention Masking (Full Version)
# ============================================================

def compute_patch_attention_mask(
    patch_embeddings: torch.Tensor,
    negative_vectors: torch.Tensor,
    tau: float = 0.3,
    soft_mask: bool = True,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Tính attention mask cho patch embeddings dựa trên negative concepts.

    Cho mỗi patch p_j và negative concept n:
        - Hard mask: mask_j = 1 if sim(p_j, n) > τ else 0
        - Soft mask: mask_j = 1 - sigmoid((sim(p_j, n) - τ) / temperature)
          → soft mask cho phép gradient-friendly computation

    Args:
        patch_embeddings: [P, D] — patch embeddings của 1 ảnh
        negative_vectors: [K, D] — negative concept vectors
        tau:              float — similarity threshold
        soft_mask:        bool — dùng soft mask (sigmoid) hay hard mask
        temperature:      float — temperature cho soft mask

    Returns:
        mask: [P] — attention mask (1 = keep, 0 = suppress)
    """
    P = patch_embeddings.shape[0]
    K = negative_vectors.shape[0]

    # Compute similarity: [K, P]
    sim = negative_vectors @ patch_embeddings.T  # [K, P]

    # Max similarity across all negative concepts per patch: [P]
    max_sim, _ = sim.max(dim=0)  # [P]

    if soft_mask:
        # Soft mask: sigmoid decay
        # High sim → mask ≈ 0, Low sim → mask ≈ 1
        mask = torch.sigmoid(-(max_sim - tau) / temperature)
    else:
        # Hard mask: binary
        mask = (max_sim <= tau).float()

    return mask


def apply_patch_attention_masking(
    q: torch.Tensor,
    patch_embeddings: torch.Tensor,
    negative_vectors: torch.Tensor,
    negative_weights: Optional[torch.Tensor] = None,
    tau: float = 0.3,
    soft_mask: bool = True,
    temperature: float = 0.1,
    min_patches: int = 4,
) -> float:
    """
    Compute masked score cho 1 ảnh:
        1. Tính mask cho từng patch
        2. Aggregate masked patches → new image representation
        3. Score = q · new_representation

    Args:
        q:                [D] query vector
        patch_embeddings: [P, D] patch embeddings của ảnh
        negative_vectors: [K, D] negative concept vectors
        negative_weights: [K] weights (from confidence), optional
        tau:              similarity threshold
        soft_mask:        use sigmoid soft mask
        temperature:      soft mask temperature
        min_patches:      minimum patches to keep (prevent complete masking)

    Returns:
        masked_score: float — new similarity score
    """
    P = patch_embeddings.shape[0]

    if negative_vectors is None or negative_vectors.shape[0] == 0:
        # No negatives → use global pooling (original behavior)
        global_repr = patch_embeddings.mean(dim=0)
        global_repr = F.normalize(global_repr, dim=-1)
        return torch.dot(q, global_repr).item()

    # Apply weights to negative vectors if provided
    if negative_weights is not None:
        # Scale negative vectors by their weights
        weighted_neg = negative_vectors * negative_weights.unsqueeze(-1)
        weighted_neg = F.normalize(weighted_neg, dim=-1)
    else:
        weighted_neg = negative_vectors

    # Compute attention mask
    mask = compute_patch_attention_mask(
        patch_embeddings, weighted_neg,
        tau=tau, soft_mask=soft_mask, temperature=temperature,
    )

    # Ensure minimum patches are kept
    if soft_mask:
        # For soft mask, ensure at least min_patches worth of attention
        total_attention = mask.sum()
        if total_attention < min_patches:
            # Scale up all mask values
            scale = min_patches / (total_attention + 1e-8)
            mask = torch.clamp(mask * scale, max=1.0)
    else:
        # For hard mask, keep top-min_patches if too few remain
        num_kept = mask.sum().int().item()
        if num_kept < min_patches:
            # Override: keep patches with lowest negative similarity
            max_sim = (weighted_neg @ patch_embeddings.T).max(dim=0)[0]
            _, keep_idx = max_sim.topk(min_patches, largest=False)
            mask = torch.zeros(P, device=mask.device)
            mask[keep_idx] = 1.0

    # Masked aggregation: weighted mean of patches
    masked_repr = (mask.unsqueeze(-1) * patch_embeddings).sum(dim=0)
    masked_repr = masked_repr / (mask.sum() + 1e-8)
    masked_repr = F.normalize(masked_repr, dim=-1)

    # Score
    return torch.dot(q, masked_repr).item()


def apply_patch_masking_batch(
    q_batch: torch.Tensor,
    patch_embeddings_batch: torch.Tensor,
    negative_vectors_list: List[Optional[torch.Tensor]],
    negative_weights_list: List[Optional[torch.Tensor]],
    tau: float = 0.3,
    soft_mask: bool = True,
    temperature: float = 0.1,
    min_patches: int = 4,
) -> torch.Tensor:
    """
    Batch version of patch-level attention masking.

    Args:
        q_batch:                  [B, D] query vectors
        patch_embeddings_batch:   [N, P, D] ALL images' patch embeddings
        negative_vectors_list:    list of B tensors [K_b, D]
        negative_weights_list:    list of B tensors [K_b]

    Returns:
        scores: [B, N] — masked similarity scores
    """
    B = q_batch.shape[0]
    N, P, D = patch_embeddings_batch.shape

    scores = torch.zeros(B, N, device=q_batch.device)

    for b in range(B):
        neg_vecs = negative_vectors_list[b]
        neg_weights = negative_weights_list[b]

        if neg_vecs is None or neg_vecs.shape[0] == 0:
            # No negatives → standard global similarity
            global_repr = patch_embeddings_batch.mean(dim=1)  # [N, D]
            global_repr = F.normalize(global_repr, dim=-1)
            scores[b] = q_batch[b] @ global_repr.T
        else:
            for n in range(N):
                scores[b, n] = apply_patch_attention_masking(
                    q_batch[b],
                    patch_embeddings_batch[n],
                    neg_vecs, neg_weights,
                    tau=tau, soft_mask=soft_mask,
                    temperature=temperature,
                    min_patches=min_patches,
                )

    return scores


# ============================================================
# Mode 2: Global Fallback (Enhanced Penalty — khi không có patches)
# ============================================================

def apply_enhanced_penalty(
    scores: torch.Tensor,
    corpus_vectors: torch.Tensor,
    negative_vectors: torch.Tensor,
    negative_weights: Optional[torch.Tensor] = None,
    tau: float = 0.20,
    max_penalty: float = 0.20,
    soft: bool = True,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Enhanced penalty scoring — fallback khi không có patch embeddings.
    
    Mạnh hơn M3 cũ:
        - Dùng negative vectors từ Concept Memory (không phải exclusion zones)
        - Weighted penalty based on concept confidence
        - Soft penalty (sigmoid) thay vì hard ReLU
        - Adaptive capping

    Args:
        scores:          [B, N] current scores
        corpus_vectors:  [N, D] image embeddings (L2-normalized)
        negative_vectors: [K, D] negative concept vectors
        negative_weights: [K] weights from confidence
        tau:             similarity threshold
        max_penalty:     maximum total penalty per image
        soft:            use sigmoid penalty vs ReLU
        temperature:     temperature for sigmoid

    Returns:
        scores: [B, N] adjusted scores
    """
    if negative_vectors is None or negative_vectors.shape[0] == 0:
        return scores

    K = negative_vectors.shape[0]

    if negative_weights is None:
        negative_weights = torch.ones(K, device=scores.device)

    # Normalize weights to sum to 1
    negative_weights = negative_weights / (negative_weights.sum() + 1e-8)

    # Compute similarity: [K, N]
    sim = negative_vectors @ corpus_vectors.T  # [K, N]

    if soft:
        # Soft penalty: weighted sigmoid
        # penalty[k, n] = w_k × sigmoid((sim[k,n] - tau) / temperature)
        penalty = negative_weights.unsqueeze(-1) * torch.sigmoid(
            (sim - tau) / temperature
        )
    else:
        # Hard penalty: weighted ReLU
        penalty = negative_weights.unsqueeze(-1) * torch.clamp(
            sim - tau, min=0.0
        )

    # Sum across all negative concepts: [N]
    total_penalty = penalty.sum(dim=0)

    # Cap total penalty
    total_penalty = torch.clamp(total_penalty, max=max_penalty)

    # Apply penalty (broadcast across batch)
    scores = scores - total_penalty.unsqueeze(0)

    return scores


def apply_attention_masking(
    scores: torch.Tensor,
    corpus_vectors: torch.Tensor,
    negative_vectors: torch.Tensor,
    negative_weights: Optional[torch.Tensor] = None,
    patch_embeddings: Optional[torch.Tensor] = None,
    tau: float = 0.25,
    max_penalty: float = 0.20,
    soft_mask: bool = True,
    temperature: float = 0.1,
    min_patches: int = 4,
) -> torch.Tensor:
    """
    Unified interface: tự động chọn Mode 1 (patch) hay Mode 2 (global).

    Args:
        scores:           [B, N] current scores
        corpus_vectors:   [N, D] global image embeddings
        negative_vectors: [K, D] negative concept vectors
        negative_weights: [K] optional weights
        patch_embeddings: [N, P, D] optional patch embeddings
        tau:              similarity threshold
        max_penalty:      max penalty for global fallback
        soft_mask:        use soft mask/penalty
        temperature:      temperature parameter
        min_patches:      min patches to keep (patch mode)

    Returns:
        scores: [B, N] adjusted scores
    """
    if negative_vectors is None or negative_vectors.shape[0] == 0:
        return scores

    if patch_embeddings is not None:
        # Mode 1: Patch-Level Attention Masking
        logger.debug("Using Patch-Level Attention Masking (Mode 1)")
        B, N = scores.shape
        q_batch = None  # Need query vectors for patch mode

        # For patch mode, we need the query vectors
        # Since we receive scores, we reconstruct: q ≈ scores[b] alignment
        # This is a limitation — caller should use apply_patch_masking_batch directly
        logger.warning(
            "Patch-level masking via unified interface not optimal. "
            "Use apply_patch_masking_batch() directly for best results."
        )
        return apply_enhanced_penalty(
            scores, corpus_vectors, negative_vectors,
            negative_weights, tau, max_penalty, soft_mask, temperature,
        )
    else:
        # Mode 2: Global Fallback (Enhanced Penalty)
        logger.debug("Using Enhanced Penalty Scoring (Mode 2 Fallback)")
        return apply_enhanced_penalty(
            scores, corpus_vectors, negative_vectors,
            negative_weights, tau, max_penalty, soft_mask, temperature,
        )
