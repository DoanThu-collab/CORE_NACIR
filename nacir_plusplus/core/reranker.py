"""
NACIR++ — Re-ranking Module (Plug-and-Play version)
=======================================================
Đây là bản TỔNG QUÁT HÓA của core/reranker.py gốc (ITMReranker gắn cứng
BLIP). Công thức kết hợp điểm (combine cosine + cross-encoder score) được
GIỮ NGUYÊN 100%:

    cos_norm = min-max-normalize(cosine_scores)
    itm_norm = min-max-normalize(scorer_scores)
    combined = itm_weight * itm_norm + (1 - itm_weight) * cos_norm

Điểm khác biệt: thay vì gọi thẳng `self.model(...)` (BLIP), hàm dưới đây gọi
qua interface `ImageScorer` (xem interfaces.py) — bất kỳ cross-encoder nào
(BLIP ITM, một VLM khác, hay module chấm điểm riêng của bạn) implement đúng
`.score(query_text, image_refs) -> Tensor` đều dùng được ở đây mà không phải
đổi lấy một dòng công thức nào.
"""

from typing import Any, Callable, List, Optional, Tuple

import torch

from ..interfaces import ImageScorer


def rerank_topk(
    query_text: str,
    top_k_corpus_indices: torch.Tensor,      # [K] index vào corpus
    image_refs: List[Any],                   # [K] tham chiếu ảnh tương ứng (path/id/obj)
    image_scorer: ImageScorer,
    cosine_scores: Optional[torch.Tensor] = None,  # [K] điểm cosine gốc
    itm_weight: float = 0.7,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Re-rank Top-K candidates bằng cross-encoder score (ITM hoặc tương đương).

    Args:
        query_text:            câu truy vấn gốc (không phải vector đã phẫu thuật)
        top_k_corpus_indices:  [K] index của top-K ảnh trong corpus
        image_refs:            [K] tham chiếu ảnh (đã map sẵn theo top_k_corpus_indices)
        image_scorer:          bất kỳ ImageScorer nào (BLIP ITM, hay khác)
        cosine_scores:         [K] điểm cosine similarity gốc (tuỳ chọn)
        itm_weight:            trọng số cho cross-encoder score trong kết hợp

    Returns:
        reranked_indices: [K] index corpus đã re-order
        reranked_scores:  [K] điểm kết hợp
    """
    scorer_scores = image_scorer.score(query_text, image_refs)

    if cosine_scores is not None:
        cos_norm = (cosine_scores - cosine_scores.min()) / (
            cosine_scores.max() - cosine_scores.min() + 1e-8
        )
        itm_norm = (scorer_scores - scorer_scores.min()) / (
            scorer_scores.max() - scorer_scores.min() + 1e-8
        )
        combined = itm_weight * itm_norm + (1 - itm_weight) * cos_norm
    else:
        combined = scorer_scores

    rerank_order = torch.argsort(combined, descending=True)
    reranked_indices = top_k_corpus_indices[rerank_order]
    reranked_scores = combined[rerank_order]

    return reranked_indices, reranked_scores


def rerank_topk_with_lookup(
    query_text: str,
    top_k_corpus_indices: torch.Tensor,
    corpus_ref_lookup: Callable[[int], Any],   # index -> path/id/obj
    image_scorer: ImageScorer,
    cosine_scores: Optional[torch.Tensor] = None,
    itm_weight: float = 0.7,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tiện ích: tự map index -> ref bằng corpus_ref_lookup rồi gọi rerank_topk."""
    image_refs = [corpus_ref_lookup(idx.item()) for idx in top_k_corpus_indices]
    return rerank_topk(
        query_text=query_text,
        top_k_corpus_indices=top_k_corpus_indices,
        image_refs=image_refs,
        image_scorer=image_scorer,
        cosine_scores=cosine_scores,
        itm_weight=itm_weight,
    )
