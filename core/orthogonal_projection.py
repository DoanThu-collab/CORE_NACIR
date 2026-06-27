import logging
from typing import List, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def orthogonal_project_single(
    q: torch.Tensor,
    n: torch.Tensor,
    strength: float = 1.0,
) -> torch.Tensor:
    
    dot = torch.dot(q, n)
    n_norm_sq = torch.dot(n, n)

    if n_norm_sq < 1e-8:
        return q

    projection = (dot / n_norm_sq) * n
    return q - strength * projection


def gram_schmidt_orthonormalize(
    vectors: torch.Tensor,
) -> torch.Tensor:
    
    basis = []
    for v in vectors:
        w = v.clone()
        for b in basis:
            w = w - torch.dot(w, b) * b

        norm = w.norm()
        if norm > 1e-6:
            basis.append(w / norm)

    if not basis:
        return torch.zeros(0, vectors.shape[1], device=vectors.device)

    return torch.stack(basis)


def orthogonal_purge(
    q: torch.Tensor,
    negative_vectors: torch.Tensor,
    negative_weights: Optional[torch.Tensor] = None,
    strength: float = 1.0,
    use_gram_schmidt: bool = True,
) -> torch.Tensor:
    
    if negative_vectors is None or negative_vectors.shape[0] == 0:
        return q

    k = negative_vectors.shape[0]

    if negative_weights is None:
        negative_weights = torch.ones(k, device=q.device)

    max_w = negative_weights.max()
    if max_w > 0:
        negative_weights = negative_weights / max_w

    if use_gram_schmidt and k > 1:
        basis = gram_schmidt_orthonormalize(negative_vectors)
        basis_weights = torch.ones(basis.shape[0], device=q.device)
    else:
        basis = negative_vectors
        basis_weights = negative_weights

    q_clean = q.clone()
    for i in range(basis.shape[0]):
        weight = basis_weights[i] if i < len(basis_weights) else 1.0
        q_clean = orthogonal_project_single(
            q_clean,
            basis[i],
            strength=strength * weight,
        )

    return F.normalize(q_clean.unsqueeze(0), dim=-1).squeeze(0)


def orthogonal_purge_batch(
    q_batch: torch.Tensor,
    negative_vectors_list: List[Optional[torch.Tensor]],
    negative_weights_list: List[Optional[torch.Tensor]],
    strength: float = 1.0,
    use_gram_schmidt: bool = True,
) -> torch.Tensor:
    
    q_clean_list = []

    for b in range(q_batch.shape[0]):
        neg_vecs = negative_vectors_list[b]
        neg_weights = negative_weights_list[b]

        if neg_vecs is not None and neg_vecs.shape[0] > 0:
            q_clean = orthogonal_purge(
                q_batch[b],
                neg_vecs,
                neg_weights,
                strength=strength,
                use_gram_schmidt=use_gram_schmidt,
            )
        else:
            q_clean = q_batch[b]

        q_clean_list.append(q_clean)

    return torch.stack(q_clean_list)


def purge_query_from_memory(
    q: torch.Tensor,
    memory_board,
    strength: float = 1.0,
    use_gram_schmidt: bool = True,
) -> torch.Tensor:
    
    neg_vecs, neg_weights = memory_board.get_negative_vectors()

    if neg_vecs is None:
        return q

    return orthogonal_purge(
        q,
        neg_vecs,
        neg_weights,
        strength=strength,
        use_gram_schmidt=use_gram_schmidt,
    )
