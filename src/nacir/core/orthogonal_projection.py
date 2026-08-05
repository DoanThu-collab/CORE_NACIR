"""
NACIR++ — Step 3: Orthogonal Projection (Phép chiếu trực giao)
================================================================
Tiến hóa từ M2 (exclusion_zone.py): Tạo "Vùng cấm" penalty thô sơ.

NACIR++ dùng toán học thượng thừa: Phép chiếu trực giao để GỌT SẠCH
vector nhiễu khỏi query, thay vì chỉ tạo zone phạt điểm.

Nguyên lý toán học:
    Cho q là query vector, n là negative concept vector.
    Thành phần của q chiếu lên n:
        proj_n(q) = (q · n / ||n||²) · n

    Gọt bỏ thành phần nhiễu:
        q_clean = q - proj_n(q)

    Với nhiều negative vectors {n_1, ..., n_k}:
        Áp dụng Gram-Schmidt để tạo orthonormal basis
        q_clean = q - Σ_i proj_{n_i}(q)

So sánh:
    Cũ (M2): Tạo zone → phạt score → vẫn còn nhiễu trong query vector
    Mới:     Gọt trực tiếp query vector → loại bỏ HOÀN TOÀN thành phần nhiễu

Ưu điểm:
    - Toán học chặt chẽ, deterministic
    - Không cần tuning threshold τ
    - Loại bỏ nhiễu ở gốc (query level) thay vì ở ngọn (score level)
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def orthogonal_project_single(
    q: torch.Tensor,
    n: torch.Tensor,
    strength: float = 1.0,
) -> torch.Tensor:
    """
    Chiếu trực giao: loại bỏ thành phần của q theo hướng n.

    q_clean = q - strength × (q·n / ||n||²) × n

    Args:
        q: query vector [D]
        n: negative vector [D] (should be L2-normalized)
        strength: hệ số điều chỉnh mức độ gọt [0, 1]
                  0 = không gọt, 1 = gọt hoàn toàn

    Returns:
        q_clean: [D] query vector đã gọt bỏ thành phần nhiễu
    """
    # proj_n(q) = (q·n / n·n) * n
    # Nếu n đã normalized thì n·n = 1
    dot = torch.dot(q, n)
    n_norm_sq = torch.dot(n, n)

    if n_norm_sq < 1e-8:
        return q  # Zero vector, skip

    projection = (dot / n_norm_sq) * n
    q_clean = q - strength * projection

    return q_clean


def gram_schmidt_orthonormalize(
    vectors: torch.Tensor,
) -> torch.Tensor:
    """
    Áp dụng Gram-Schmidt orthonormalization trên tập negative vectors.
    Tạo orthonormal basis để phép chiếu không bị overlap.

    Args:
        vectors: [K, D] tensor — K negative vectors

    Returns:
        basis: [K', D] tensor — orthonormal basis (K' ≤ K, loại bỏ
               linearly dependent vectors)
    """
    basis = []
    for v in vectors:
        # Trừ projection lên các basis vectors đã có
        w = v.clone()
        for b in basis:
            w = w - torch.dot(w, b) * b

        # Kiểm tra norm: nếu quá nhỏ → linearly dependent, bỏ qua
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
    """
    Gọt sạch query vector bằng phép chiếu trực giao lên TẤT CẢ
    negative concept vectors.

    Workflow:
        1. (Optional) Gram-Schmidt orthonormalize negative vectors
        2. Cho mỗi negative direction:
           q = q - strength × weight × proj(q onto n)
        3. Re-normalize q

    Args:
        q:                 [D] query vector
        negative_vectors:  [K, D] negative concept vectors (L2-normalized)
        negative_weights:  [K] optional weights per concept (from confidence)
        strength:          float [0, 1] — overall projection strength
        use_gram_schmidt:  bool — apply Gram-Schmidt trước khi project

    Returns:
        q_clean: [D] query vector đã gọt sạch
    """
    if negative_vectors is None or negative_vectors.shape[0] == 0:
        return q

    K = negative_vectors.shape[0]

    # Default weights = uniform
    if negative_weights is None:
        negative_weights = torch.ones(K, device=q.device)

    # Normalize weights to [0, 1]
    max_w = negative_weights.max()
    if max_w > 0:
        negative_weights = negative_weights / max_w

    # Step 1: Gram-Schmidt (optional)
    if use_gram_schmidt and K > 1:
        basis = gram_schmidt_orthonormalize(negative_vectors)
        # After GS, all weights become uniform (basis is orthonormal)
        gs_weights = torch.ones(basis.shape[0], device=q.device)
    else:
        basis = negative_vectors
        gs_weights = negative_weights

    # Step 2: Iterative projection removal
    q_clean = q.clone()
    for i in range(basis.shape[0]):
        w = gs_weights[i] if i < len(gs_weights) else 1.0
        q_clean = orthogonal_project_single(
            q_clean, basis[i], strength=strength * w
        )

    # Step 3: Re-normalize
    q_clean = F.normalize(q_clean.unsqueeze(0), dim=-1).squeeze(0)

    return q_clean


def orthogonal_purge_batch(
    q_batch: torch.Tensor,
    negative_vectors_list: List[Optional[torch.Tensor]],
    negative_weights_list: List[Optional[torch.Tensor]],
    strength: float = 1.0,
    use_gram_schmidt: bool = True,
) -> torch.Tensor:
    """
    Batch version: gọt query vector cho cả batch.

    Args:
        q_batch:               [B, D] query vectors
        negative_vectors_list: list of B tensors, mỗi cái [K_b, D] hoặc None
        negative_weights_list: list of B tensors, mỗi cái [K_b] hoặc None
        strength:              float [0, 1]
        use_gram_schmidt:      bool

    Returns:
        q_clean_batch: [B, D]
    """
    B = q_batch.shape[0]
    q_clean_list = []

    for b in range(B):
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


# ============================================================
# Convenience: Combine with Concept Memory Board
# ============================================================

def purge_query_from_memory(
    q: torch.Tensor,
    memory_board,
    strength: float = 1.0,
    use_gram_schmidt: bool = True,
) -> torch.Tensor:
    """
    Gọt query vector dựa trên negative concepts từ Concept Memory Board.

    Đây là hàm kết nối Step 3 (Orthogonal Projection) với Step 2
    (Concept Memory Board).

    Args:
        q:              [D] query vector
        memory_board:   ConceptMemoryBoard instance
        strength:       float [0, 1]
        use_gram_schmidt: bool

    Returns:
        q_clean: [D]
    """
    neg_vecs, neg_weights = memory_board.get_negative_vectors()

    if neg_vecs is None:
        return q

    return orthogonal_purge(
        q, neg_vecs, neg_weights,
        strength=strength,
        use_gram_schmidt=use_gram_schmidt,
    )
