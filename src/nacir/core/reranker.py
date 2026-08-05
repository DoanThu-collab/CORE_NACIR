"""
NACIR++ — ITM Re-ranking Module
=================================
Sau khi cosine similarity tìm ra Top-K ảnh ứng viên,
dùng BLIP Image-Text Matching (ITM) head để re-rank lại.

ITM head tính toán cross-attention giữa image patches và text tokens,
cho điểm matching chính xác hơn rất nhiều so với cosine similarity
trên global vectors.

Usage:
    from nacir_plus.reranker import ITMReranker
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
        text,  # str or List[str]
        image_paths: List[str],
        batch_size: int = 16,
    ) -> torch.Tensor:
        """
        Compute ITM scores for multiple query-image pairs.

        Args:
            text:        Query text string or List of query texts
            image_paths: List of image file paths

        Returns:
            scores: [N] ITM matching scores (higher = better match)
        """
        all_scores = []

        def load_image(p):
            try:
                return Image.open(p).convert('RGB')
            except Exception as e:
                logger.warning(f"Failed to load image {p}: {e}")
                return Image.new('RGB', (384, 384))

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            
            # Đọc ảnh song song bằng Multi-threading (Giải quyết nghẽn NAS)
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, len(batch_paths))) as executor:
                images = list(executor.map(load_image, batch_paths))

            # Prepare texts
            if isinstance(text, str):
                text_list = [text] * len(images)
            else:
                text_list = text[i:i + len(images)]

            # Process text + images
            inputs = self.processor(
                text=text_list,
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

    def rerank_batch(
        self,
        query_texts: List[str],
        top_k_corpus_indices: torch.Tensor,  # [B, K]
        corpus_paths: List[str],
        cosine_scores: Optional[torch.Tensor] = None,  # [B, K]
        itm_weight: float = 0.7,
        batch_size: int = 128,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Re-rank top-K candidates for an entire batch of queries using ITM scores.
        This is MUCH faster than looping over queries individually because it
        batches the image loading and ITM forward passes.
        """
        B, K = top_k_corpus_indices.shape
        
        # Flatten image paths and text
        flat_image_paths = []
        flat_texts = []
        for b in range(B):
            for k in range(K):
                idx = top_k_corpus_indices[b, k].item()
                flat_image_paths.append(corpus_paths[idx])
                flat_texts.append(query_texts[b])
                
        # Compute ITM scores in batch
        itm_scores = self.compute_itm_score(flat_texts, flat_image_paths, batch_size=batch_size)
        itm_scores = itm_scores.view(B, K).to(top_k_corpus_indices.device)
        
        if cosine_scores is not None:
            cosine_scores = cosine_scores.to(top_k_corpus_indices.device)
        
        # Rerank
        reranked_indices = torch.zeros_like(top_k_corpus_indices)
        reranked_scores = torch.zeros_like(top_k_corpus_indices, dtype=torch.float32)
        
        for b in range(B):
            if cosine_scores is not None:
                cos_s = cosine_scores[b]
                itm_s = itm_scores[b]
                cos_norm = (cos_s - cos_s.min()) / (cos_s.max() - cos_s.min() + 1e-8)
                itm_norm = (itm_s - itm_s.min()) / (itm_s.max() - itm_s.min() + 1e-8)
                combined = itm_weight * itm_norm + (1 - itm_weight) * cos_norm
            else:
                combined = itm_scores[b]
                
            rerank_order = torch.argsort(combined, descending=True)
            reranked_indices[b] = top_k_corpus_indices[b][rerank_order]
            reranked_scores[b] = combined[rerank_order]
            
        return reranked_indices, reranked_scores

def rerank_topk_with_lookup(
    query_text: str,
    top_k_corpus_indices: torch.Tensor,
    corpus_ref_lookup,
    image_scorer,
    cosine_scores: Optional[torch.Tensor] = None,
    itm_weight: float = 0.7,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generic re-ranking function that uses the ImageScorer adapter interface.
    """
    K = len(top_k_corpus_indices)
    
    image_paths = [corpus_ref_lookup(idx.item()) for idx in top_k_corpus_indices]
    
    itm_scores = image_scorer.score(query_text, image_paths)
    itm_scores = itm_scores.squeeze(0).to(device)
    
    if cosine_scores is not None:
        cosine_scores = cosine_scores.to(device)
        cos_norm = (cosine_scores - cosine_scores.min()) / (cosine_scores.max() - cosine_scores.min() + 1e-8)
        itm_norm = (itm_scores - itm_scores.min()) / (itm_scores.max() - itm_scores.min() + 1e-8)
        combined = itm_weight * itm_norm + (1 - itm_weight) * cos_norm
    else:
        combined = itm_scores
        
    rerank_order = torch.argsort(combined, descending=True)
    reranked_indices = top_k_corpus_indices.to(device)[rerank_order]
    reranked_scores = combined[rerank_order]
    
    return reranked_indices, reranked_scores
