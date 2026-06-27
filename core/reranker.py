"""
NACIR++ — ITM Re-ranking Module
=================================
Sau khi cosine similarity tìm ra Top-K ảnh ứng viên,
dùng BLIP Image-Text Matching (ITM) head để re-rank lại.

ITM head tính toán cross-attention giữa image patches và text tokens,
cho điểm matching chính xác hơn rất nhiều so với cosine similarity
trên global vectors.

Usage:
    from core.reranker import ITMReranker
    reranker = ITMReranker(model, processor, device)
    new_ranks = reranker.rerank_batch(queries_text, top_k_indices, corpus_paths, K=10)
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple
from PIL import Image
import logging
import concurrent.futures

logger = logging.getLogger(__name__)


class ITMReranker:
    """
    BLIP ITM-based re-ranker.
    
    After first-stage retrieval (cosine similarity on global vectors),
    this re-ranker loads the actual images for top-K candidates and
    computes precise Image-Text Matching scores using BLIP's ITM head.
    """

    def __init__(self, model, processor, device: str, rerank_k: int = 50):
        """
        Args:
            model:     BlipForImageTextRetrieval model (with ITM head)
            processor: AutoProcessor for BLIP
            device:    'cuda' or 'cpu'
            rerank_k:  Number of top candidates to re-rank (default 50)
        """
        self.model = model
        self.processor = processor
        self.device = device
        self.rerank_k = rerank_k

    @torch.no_grad()
    def compute_itm_score(
        self,
        text: str,
        image_paths: List[str],
        batch_size: int = 16,
    ) -> torch.Tensor:
        """
        Compute ITM scores for one query against multiple images.

        Args:
            text:        Query text string
            image_paths: List of image file paths

        Returns:
            scores: [N] ITM matching scores (higher = better match)
        """
        all_scores = []

        def load_image(p):
            try:
                return Image.open(p).convert('RGB'), True
            except Exception as e:
                # logger.warning(f"Failed to load image {p}: {e}") # Bỏ warning cho đỡ rác log
                return Image.new('RGB', (384, 384)), False

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            
            # Đọc ảnh song song bằng Multi-threading (Giải quyết nghẽn NAS)
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, len(batch_paths))) as executor:
                results = list(executor.map(load_image, batch_paths))
                images = [res[0] for res in results]
                valid_mask = torch.tensor([res[1] for res in results], dtype=torch.bool, device=self.device)

            # Process text + images
            inputs = self.processor(
                text=[text] * len(images),
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Forward pass through ITM head
            outputs = self.model(**inputs)

            # ITM score: probability of "match" class
            # outputs.itm_score has shape [batch, 2] (not_match, match)
            itm_logits = outputs.itm_score
            itm_probs = F.softmax(itm_logits, dim=-1)
            match_scores = itm_probs[:, 1]  # probability of "match"
            
            # Phạt nặng các ảnh bị lỗi load để đẩy chúng xuống bét bảng
            match_scores = torch.where(valid_mask, match_scores, torch.tensor(-100.0, device=self.device))

            all_scores.append(match_scores.cpu())

        return torch.cat(all_scores)

    def rerank_topk(
        self,
        query_text: str,
        top_k_corpus_indices: torch.Tensor,  # [K] indices into corpus
        corpus_paths: List[str],             # Full corpus paths
        cosine_scores: Optional[torch.Tensor] = None,  # [K] original scores
        itm_weight: float = 0.7,             # Weight for ITM vs cosine
        batch_size: int = 128,               # Batch size for ITM
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Re-rank top-K candidates using ITM scores.

        Args:
            query_text:           Query text string
            top_k_corpus_indices: [K] indices of top-K images in corpus
            corpus_paths:         List of all corpus image paths
            cosine_scores:        [K] original cosine similarity scores
            itm_weight:           Weight for ITM score in final ranking

        Returns:
            reranked_indices: [K] corpus indices, re-ordered
            reranked_scores:  [K] combined scores
        """
        K = len(top_k_corpus_indices)

        # Get image paths for top-K candidates
        image_paths = [corpus_paths[idx.item()] for idx in top_k_corpus_indices]

        # Compute ITM scores
        itm_scores = self.compute_itm_score(query_text, image_paths, batch_size=batch_size)

        if cosine_scores is not None:
            # Normalize both scores to [0, 1]
            cos_norm = (cosine_scores - cosine_scores.min()) / (cosine_scores.max() - cosine_scores.min() + 1e-8)
            itm_norm = (itm_scores - itm_scores.min()) / (itm_scores.max() - itm_scores.min() + 1e-8)

            # Combined score
            combined = itm_weight * itm_norm + (1 - itm_weight) * cos_norm
        else:
            combined = itm_scores

        # Re-rank
        rerank_order = torch.argsort(combined, descending=True)
        reranked_indices = top_k_corpus_indices[rerank_order]
        reranked_scores = combined[rerank_order]

        return reranked_indices, reranked_scores
