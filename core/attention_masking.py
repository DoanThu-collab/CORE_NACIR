import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# Mode 1: Patch-Level Attention Masking
def compute_patch_attention_mask(
    patch_embeddings: torch.Tensor,
    negative_vectors: torch.Tensor,
    tau: float = 0.3,
    soft_mask: bool = True,
    temperature: float = 0.1,
) -> torch.Tensor:
   
    P = patch_embeddings.shape[0]
    K = negative_vectors.shape[0]

    # Compute similarity: [K, P]
    sim = negative_vectors @ patch_embeddings.T  # [K, P]

    # Max similarity across all negative concepts per patch: [P]
    max_sim, _ = sim.max(dim=0)  # [P]

    if soft_mask:
        # Soft mask: sigmoid decay
        # High similarity means lower attention; low similarity keeps attention.
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
    
    P = patch_embeddings.shape[0]

    if negative_vectors is None or negative_vectors.shape[0] == 0:
        # No negatives: use global pooling.
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
    
    B = q_batch.shape[0]
    N, P, D = patch_embeddings_batch.shape

    scores = torch.zeros(B, N, device=q_batch.device)

    for b in range(B):
        neg_vecs = negative_vectors_list[b]
        neg_weights = negative_weights_list[b]

        if neg_vecs is None or neg_vecs.shape[0] == 0:
            # No negatives: use standard global similarity.
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


# Mode 2: Global fallback when patch embeddings are unavailable

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
        # penalty[k, n] = w_k * sigmoid((sim[k,n] - tau) / temperature)
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
   
    if negative_vectors is None or negative_vectors.shape[0] == 0:
        return scores

    if patch_embeddings is not None:
        # Mode 1: Patch-Level Attention Masking
        logger.debug("Using Patch-Level Attention Masking (Mode 1)")
        B, N = scores.shape
        q_batch = None  # Need query vectors for patch mode

        # For patch mode, we need the query vectors
        # This path receives scores only, so patch mode should call
        # apply_patch_masking_batch directly.
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
