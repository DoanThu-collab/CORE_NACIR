"""
Visual-Grounded Belief Refinement — Đề xuất Novelty #2
=======================================================
Biến pipeline NACIR++ từ **Open-loop** thành **Closed-loop**.

Vấn đề:
    Toàn bộ pipeline hiện tại hoạt động theo kiểu "bắn rồi quên":
    1. Trích xuất beliefs từ text dialog
    2. Phẫu thuật query vector
    3. Tìm kiếm ảnh
    4. KHÔNG HỀ nhìn lại kết quả tìm kiếm để đánh giá beliefs!

    Hệ quả: Một belief sai lệch (ví dụ LLM hallucinate "red car" trong
    khi user chỉ nói "vehicle") sẽ kéo query đi sai hướng suốt phần
    còn lại của cuộc hội thoại, không có cơ chế tự sửa.

Giải pháp:
    Sau khi tính score retrieval, NHÌN NGƯỢC vào Top-K ảnh để đánh giá
    từng belief trong memory:

    relevance(c) = mean( c.vector · V_topk^T )

    Với V_topk = visual features của K ảnh xếp hạng cao nhất.

    Quy tắc điều chỉnh:
    ┌─────────────┬──────────────┬───────────────────────────────────┐
    │ Polarity    │ Relevance    │ Hành động                         │
    ├─────────────┼──────────────┼───────────────────────────────────┤
    │ POSITIVE    │ Thấp (< τ_l) │ Suppress: concept này KHÔNG      │
    │             │              │ hiện diện trong top results →     │
    │             │              │ nó đang gây nhiễu → giảm weight  │
    ├─────────────┼──────────────┼───────────────────────────────────┤
    │ NEGATIVE    │ Cao (> τ_h)  │ Boost: concept này VẪN CÒN       │
    │             │              │ trong top results → Vector Surgery│
    │             │              │ chưa loại bỏ hết → tăng penalty  │
    └─────────────┴──────────────┴───────────────────────────────────┘

    Sau khi điều chỉnh, pipeline re-synthesize query và re-score.

Paper references:
    - Relevance Feedback in IR (Rocchio, 1971) — ý tưởng gốc
    - Pseudo-Relevance Feedback (Lavrenko & Croft, 2001) — PRF
    - Visual Feedback trong CBIR (Rui et al., 1998)
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class VisualFeedbackRefiner:
    """
    Closed-loop Visual Feedback cho Concept Memory.

    Nhìn vào top-K ảnh được trả về để đánh giá chất lượng
    của từng belief, sau đó điều chỉnh trọng số.
    """

    def __init__(
        self,
        feedback_top_k: int = 50,
        suppress_threshold: float = 0.15,
        boost_threshold: float = 0.25,
        suppress_factor: float = 0.3,
        boost_factor: float = 0.2,
        max_refinement_iterations: int = 1,
    ):
        """
        Args:
            feedback_top_k: Số lượng ảnh top-K để phân tích.
            suppress_threshold: Ngưỡng relevance THẤP.
                Positive beliefs có relevance < ngưỡng này sẽ bị suppress.
            boost_threshold: Ngưỡng relevance CAO.
                Negative beliefs có relevance > ngưỡng này sẽ được boost.
            suppress_factor: Mức giảm confidence khi suppress (0.0-1.0).
            boost_factor: Mức tăng confidence khi boost (0.0-1.0).
            max_refinement_iterations: Số vòng refine (thường 1 là đủ).
        """
        self.top_k = feedback_top_k
        self.suppress_threshold = suppress_threshold
        self.boost_threshold = boost_threshold
        self.suppress_factor = suppress_factor
        self.boost_factor = boost_factor
        self.max_iters = max_refinement_iterations

    def compute_relevance(
        self,
        concept_vectors: torch.Tensor,
        topk_image_vectors: torch.Tensor,
    ) -> torch.Tensor:
        """
        Tính relevance score cho mỗi concept dựa trên top-K ảnh.

        relevance(c) = mean( c · V_topk^T )

        Args:
            concept_vectors: [C, D] các concept vectors
            topk_image_vectors: [K, D] visual features của top-K ảnh

        Returns:
            relevance: [C] — mỗi giá trị trong [−1, 1]
        """
        # [C, K] = concept_vectors @ topk_image_vectors.T
        sim_matrix = concept_vectors @ topk_image_vectors.T
        # Mean over top-K images
        relevance = sim_matrix.mean(dim=1)  # [C]
        return relevance

    def refine(
        self,
        board,
        corpus_vectors: torch.Tensor,
        scores: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Phân tích top-K ảnh và trả về dict các điều chỉnh confidence.

        Args:
            board: ConceptMemoryBoard hiện tại
            corpus_vectors: [N_corpus, D] toàn bộ corpus vectors
            scores: [N_corpus] similarity scores vừa tính

        Returns:
            adjustments: Dict[concept_name -> delta_confidence]
                Giá trị > 0 = tăng confidence (boost negative)
                Giá trị < 0 = giảm confidence (suppress positive)
                {} nếu không có gì cần điều chỉnh
        """
        entries = list(board.memory.values())
        if not entries:
            return {}

        # ── Lấy top-K image vectors ──
        k = min(self.top_k, scores.shape[0])
        topk_indices = torch.topk(scores, k).indices  # [K]
        topk_vectors = corpus_vectors[topk_indices]    # [K, D]

        # ── Tính relevance cho mỗi concept ──
        concept_vectors = torch.stack([e.vector for e in entries])  # [C, D]
        relevance = self.compute_relevance(concept_vectors, topk_vectors)  # [C]

        # ── Quyết định điều chỉnh ──
        adjustments: Dict[str, float] = {}

        for idx, entry in enumerate(entries):
            rel = relevance[idx].item()

            if entry.polarity == "positive" and rel < self.suppress_threshold:
                # ┌──────────────────────────────────────────────────┐
                # │ SUPPRESS: Positive belief nhưng KHÔNG hiện diện │
                # │ trong top results → nó đang kéo query đi sai    │
                # │ hướng → giảm confidence.                        │
                # └──────────────────────────────────────────────────┘
                delta = -self.suppress_factor
                adjustments[entry.name] = delta
                logger.debug(
                    f"[VISUAL FB] SUPPRESS positive '{entry.name}': "
                    f"relevance={rel:.3f} < {self.suppress_threshold} "
                    f"→ Δconf={delta:.2f}"
                )

            elif entry.polarity == "negative" and rel > self.boost_threshold:
                # ┌──────────────────────────────────────────────────┐
                # │ BOOST: Negative belief nhưng VẪN CÒN hiện diện │
                # │ mạnh trong top results → Vector Surgery chưa    │
                # │ đủ mạnh → tăng confidence để phạt nặng hơn.    │
                # └──────────────────────────────────────────────────┘
                delta = +self.boost_factor
                adjustments[entry.name] = delta
                logger.debug(
                    f"[VISUAL FB] BOOST negative '{entry.name}': "
                    f"relevance={rel:.3f} > {self.boost_threshold} "
                    f"→ Δconf=+{delta:.2f}"
                )

        return adjustments

    def apply_adjustments(
        self, board, adjustments: Dict[str, float]
    ) -> int:
        """
        Áp dụng các điều chỉnh confidence lên ConceptMemoryBoard.

        Args:
            board: ConceptMemoryBoard
            adjustments: Dict[name -> delta]

        Returns:
            Số concept đã bị điều chỉnh
        """
        count = 0
        for name, delta in adjustments.items():
            if name in board.memory:
                entry = board.memory[name]
                old_conf = entry.confidence
                entry.confidence = max(0.01, min(1.0, old_conf + delta))
                count += 1
                logger.debug(
                    f"[VISUAL FB] '{name}' confidence: "
                    f"{old_conf:.3f} → {entry.confidence:.3f}"
                )
        return count
