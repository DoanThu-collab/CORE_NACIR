"""
Dynamic Concept Graph (DCG) — Đề xuất Novelty #1
=================================================
Giải quyết vấn đề **Semantic Leakage** trong Concept Memory.

Vấn đề:
    Khi user phủ định "dog", hệ thống hiện tại CHỈ loại bỏ vector "dog".
    Nhưng các concept liên quan ngữ nghĩa ("puppy", "animal", "pet") vẫn
    lọt lưới vì chúng có vector khác biệt đủ nhỏ để thoát khỏi
    concept_match_threshold.

Giải pháp:
    Xây đồ thị ngữ nghĩa G = (V, E) trong đó:
      - V = tất cả concepts trong ConceptMemoryBoard
      - E[i,j] = max(0, cos(v_i, v_j) - τ)    (cạnh có trọng số)
    
    Khi concept "dog" bị đánh dấu negative, tín hiệu negative được
    LAN TRUYỀN qua các cạnh tới các node lân cận (graph convolution):
    
        w'  =  (1 − α) · w  +  α · Ã · w

    trong đó Ã = D⁻¹A là ma trận kề chuẩn hóa theo hàng.

    Kết quả: "puppy" (gần "dog") bị dính hình phạt lan truyền,
             "mountain" (xa "dog") không bị ảnh hưởng.

Paper references:
    - Kipf & Welling (2017): Semi-Supervised Classification with GCN
    - Zhu et al. (2003): Semi-Supervised Learning Using Gaussian Fields (Label Propagation)
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ConceptGraph:
    """
    Đồ thị khái niệm động — xây dựng mỗi turn từ bảng nhớ,
    lan truyền tín hiệu negative/positive qua các cạnh ngữ nghĩa.

    Phiên bản V2 bổ sung:
      - Turn-Evolving Graph: Giữ lại đồ thị từ turn trước, pha trộn
        với đồ thị mới bằng temporal smoothing để theo dõi sự tiến hóa
        của ý định người dùng.
      - Bimodal Concept Node: Trộn vector Text của mỗi concept với
        Visual evidence từ top-K ảnh kết quả, giúp đồ thị hoạt động
        trên cả 2 chiều thông tin (ngôn ngữ + hình ảnh).
    """

    def __init__(
        self,
        propagation_alpha: float = 0.3,
        similarity_threshold: float = 0.50,
        num_hops: int = 1,
        # ── Turn-Evolving Graph ──
        evolving: bool = False,
        temporal_gamma: float = 0.3,
        # ── Bimodal Concept Node ──
        bimodal: bool = False,
        bimodal_lambda: float = 0.2,
        bimodal_top_k: int = 10,
    ):
        """
        Args:
            propagation_alpha: Cường độ lan truyền.
                0.0 = không lan truyền (tương đương bảng phẳng gốc).
                1.0 = chỉ dùng tín hiệu từ neighbor (quá mạnh, không khuyến khích).
                0.2-0.4 = vùng hoạt động tốt.
            similarity_threshold: Ngưỡng cosine similarity tối thiểu
                để tạo cạnh giữa 2 concepts. Dưới ngưỡng → không có cạnh.
            num_hops: Số bước lan truyền.
                1 = chỉ neighbor trực tiếp.
                2 = neighbor của neighbor (lan xa hơn nhưng mờ hơn).
            evolving: Bật Turn-Evolving Graph (temporal smoothing).
            temporal_gamma: Tỷ lệ pha trộn đồ thị mới vào đồ thị cũ.
                A_blended = (1 - γ) · A_prev + γ · A_current
                γ nhỏ → đồ thị "nhớ" cấu trúc cũ lâu hơn.
                γ lớn → đồ thị thay đổi nhanh theo turn mới.
            bimodal: Bật Bimodal Concept Node (visual grounding).
            bimodal_lambda: Tỷ lệ pha trộn visual vào text vector.
                v_bimodal = (1 - λ) · v_text + λ · v_visual
            bimodal_top_k: Số ảnh top-K dùng để tính visual grounding.
        """
        self.alpha = propagation_alpha
        self.threshold = similarity_threshold
        self.num_hops = num_hops

        # Turn-Evolving
        self.evolving = evolving
        self.temporal_gamma = temporal_gamma
        self._prev_A: Optional[torch.Tensor] = None
        self._prev_concept_keys: Optional[list] = None

        # Bimodal
        self.bimodal = bimodal
        self.bimodal_lambda = bimodal_lambda
        self.bimodal_top_k = bimodal_top_k

    def _build_adjacency(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        Xây ma trận kề A từ concept vectors.

        A[i,j] = max(0, cos(v_i, v_j) - τ)

        Args:
            vectors: [N, D] — đã L2-normalized

        Returns:
            A: [N, N] — thresholded, self-loop = 0
        """
        N = vectors.shape[0]
        if N <= 1:
            return torch.zeros(N, N, device=vectors.device)

        # Pairwise cosine similarity (vectors đã normalize)
        sim = vectors @ vectors.T  # [N, N]

        # Threshold + loại self-loop
        A = torch.clamp(sim - self.threshold, min=0.0)
        A.fill_diagonal_(0.0)

        return A

    def _row_normalize(self, A: torch.Tensor) -> torch.Tensor:
        """
        Chuẩn hóa theo hàng:  Ã = D⁻¹ · A

        Returns:
            Ã: [N, N]
        """
        degree = A.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return A / degree

    def _propagate(
        self, A_norm: torch.Tensor, signal: torch.Tensor
    ) -> torch.Tensor:
        """
        Graph convolution K bước:

            s⁰ = signal
            s^{k+1} = (1−α) · s⁰ + α · Ã · sᵏ

        Lưu ý: Dùng Personalized PageRank style (giữ lại gốc s⁰)
        thay vì GCN thuần túy, để tránh tín hiệu bị loãng hoàn toàn
        sau nhiều hop.

        Args:
            A_norm: [N, N] ma trận kề đã chuẩn hóa
            signal: [N] vector tín hiệu gốc

        Returns:
            propagated: [N] tín hiệu sau K bước lan truyền
        """
        s = signal.clone()
        s0 = signal.clone()  # Giữ bản gốc (anchor)

        for _ in range(self.num_hops):
            neighbor = A_norm @ s         # Tổng hợp tín hiệu từ neighbor
            s = (1 - self.alpha) * s0 + self.alpha * neighbor

        return s

    def _visual_ground(
        self,
        vectors: torch.Tensor,
        corpus_vectors: Optional[torch.Tensor],
        scores: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Bimodal Concept Node: Pha trộn text vector với visual evidence.

        Cho mỗi concept c_i có text vector v_i:
          1. Lấy top-K ảnh từ bảng điểm hiện tại
          2. Tính visual_relevance = cos(v_i, img_j) cho mỗi ảnh j ∈ top-K
          3. visual_grounding = weighted_mean(img_vectors, softmax(relevance))
          4. v_bimodal = (1 - λ) · v_text + λ · Normalize(visual_grounding)

        Args:
            vectors: [N, D] concept text vectors (đã L2-normalized)
            corpus_vectors: [M, D] toàn bộ corpus image vectors
            scores: [M] điểm retrieval hiện tại (dùng để chọn top-K)

        Returns:
            blended: [N, D] bimodal vectors (đã L2-normalized)
        """
        if corpus_vectors is None or scores is None:
            return vectors

        # Lấy top-K ảnh
        k = min(self.bimodal_top_k, scores.shape[0])
        top_k_indices = torch.topk(scores, k).indices  # [K]
        top_k_imgs = corpus_vectors[top_k_indices]       # [K, D]

        # Tính relevance giữa mỗi concept và top-K ảnh
        # vectors: [N, D], top_k_imgs: [K, D]
        relevance = vectors @ top_k_imgs.T  # [N, K]
        attn_weights = torch.softmax(relevance * 5.0, dim=-1)  # [N, K], temperature=0.2

        # Visual grounding = weighted mean of top-K image vectors
        visual = attn_weights @ top_k_imgs  # [N, D]
        visual = F.normalize(visual, dim=-1)

        # Pha trộn
        blended = (1 - self.bimodal_lambda) * vectors + self.bimodal_lambda * visual
        blended = F.normalize(blended, dim=-1)

        num_grounded = (attn_weights.max(dim=-1).values > 0.2).sum().item()
        if num_grounded > 0:
            logger.debug(
                f"[DCG-Bimodal] {num_grounded}/{vectors.shape[0]} concepts "
                f"strongly grounded by visual evidence"
            )

        return blended

    def _evolve_adjacency(
        self, A: torch.Tensor, concept_keys: list
    ) -> torch.Tensor:
        """
        Turn-Evolving Graph: Pha trộn đồ thị hiện tại với đồ thị từ turn trước.

        Công thức:
            A_blended = (1 - γ) · A_prev_aligned + γ · A_current

        Các concept mới (chưa có ở turn trước) sẽ chỉ dùng A_current.
        Các concept cũ (đã có ở turn trước) sẽ được pha trộn temporal.

        Args:
            A: [N, N] ma trận kề hiện tại
            concept_keys: list of concept keys (dùng để align giữa 2 turn)

        Returns:
            A_blended: [N, N]
        """
        if self._prev_A is None or self._prev_concept_keys is None:
            # Turn đầu tiên → không có gì để pha trộn
            self._prev_A = A.clone()
            self._prev_concept_keys = concept_keys[:]
            return A

        N = len(concept_keys)
        A_blended = A.clone()

        # Tìm mapping: concept hiện tại → index trong turn trước
        prev_key_to_idx = {k: i for i, k in enumerate(self._prev_concept_keys)}

        for i, ki in enumerate(concept_keys):
            for j, kj in enumerate(concept_keys):
                if ki in prev_key_to_idx and kj in prev_key_to_idx:
                    pi = prev_key_to_idx[ki]
                    pj = prev_key_to_idx[kj]
                    if pi < self._prev_A.shape[0] and pj < self._prev_A.shape[1]:
                        # Pha trộn: cạnh cũ và cạnh mới
                        A_blended[i, j] = (
                            (1 - self.temporal_gamma) * self._prev_A[pi, pj]
                            + self.temporal_gamma * A[i, j]
                        )

        # Lưu lại cho turn sau
        self._prev_A = A_blended.clone()
        self._prev_concept_keys = concept_keys[:]

        num_evolved = sum(1 for k in concept_keys if k in prev_key_to_idx)
        if num_evolved > 0:
            logger.debug(
                f"[DCG-Evolving] {num_evolved}/{N} concepts evolved "
                f"from previous turn (γ={self.temporal_gamma})"
            )

        return A_blended

    def propagate(
        self, board,
        corpus_vectors: Optional[torch.Tensor] = None,
        scores: Optional[torch.Tensor] = None,
    ) -> Tuple[
        Optional[torch.Tensor], Optional[torch.Tensor],
        Optional[torch.Tensor], Optional[torch.Tensor],
    ]:
        """
        Trích xuất concepts từ ConceptMemoryBoard, xây graph,
        lan truyền tín hiệu, và trả về vectors/weights đã điều chỉnh.

        Cơ chế chính (Cross-Polarity Propagation):
        ─────────────────────────────────────────────
        1. Thu thập TẤT CẢ concepts (cả pos lẫn neg) vào một graph chung.
        2. Lan truyền tín hiệu NEGATIVE qua toàn bộ graph.
        3. Các positive concepts nằm GẦN negative concepts sẽ bị
           "nhiễm" (infected) → giảm trọng số positive.
        4. Các negative concepts được TĂNG trọng số nhờ tín hiệu
           cộng hưởng từ các neighbor cùng chiều.

        Ví dụ:
            Memory = {"dog": neg(0.8), "puppy": pos(0.7), "mountain": pos(0.6)}
            Graph: "dog" ←0.7→ "puppy", "dog" ←0.0→ "mountain"
            
            Sau propagation:
              "puppy".pos_weight  = 0.7 - 0.3*0.8*0.7 = 0.532  (bị giảm!)
              "mountain".pos_weight = 0.6               (không đổi)
              "dog".neg_weight    = 0.8 + boost          (có thể tăng)

        Args:
            board: ConceptMemoryBoard

        Returns:
            (pos_vectors, pos_weights, neg_vectors, neg_weights)
            — Tất cả đã qua graph propagation.
            — Trả (None, None, None, None) nếu memory trống.
        """
        entries = list(board.memory.values())
        if not entries:
            return None, None, None, None

        # ── Thu thập vectors, weights, polarities ──
        concept_keys = list(board.memory.keys())  # Dùng cho Turn-Evolving
        vectors = torch.stack([e.vector for e in entries])  # [N, D]

        # ── Bimodal: Visual Grounding (nếu bật) ──
        if self.bimodal and corpus_vectors is not None and scores is not None:
            vectors = self._visual_ground(vectors, corpus_vectors, scores)
        N = vectors.shape[0]

        raw_weights = []
        is_negative = []
        for e in entries:
            recency = 1.0 / (
                1.0 + board.config.recency_decay
                * (board.current_turn - e.turn_updated)
            )
            raw_weights.append(e.confidence * recency)
            is_negative.append(e.polarity == "negative")

        weights = torch.tensor(raw_weights, device=vectors.device)  # [N]
        neg_mask = torch.tensor(is_negative, device=vectors.device)  # [N]
        pos_mask = ~neg_mask

        # ── Xây graph ──
        A = self._build_adjacency(vectors)       # [N, N]

        # ── Turn-Evolving: Pha trộn temporal (nếu bật) ──
        if self.evolving:
            A = self._evolve_adjacency(A, concept_keys)

        A_norm = self._row_normalize(A)           # [N, N]

        # ══════════════════════════════════════════════════════════
        # CROSS-POLARITY PROPAGATION (Novelty chính)
        # ══════════════════════════════════════════════════════════

        # 1. Lan truyền tín hiệu NEGATIVE
        neg_signal = torch.zeros(N, device=vectors.device)
        if neg_mask.any():
            neg_signal[neg_mask] = weights[neg_mask]

        propagated_neg = self._propagate(A_norm, neg_signal)  # [N]

        # 2. Điều chỉnh trọng số
        adjusted_weights = weights.clone()

        if pos_mask.any():
            # Positive concepts bị "nhiễm" → giảm weight
            infection = propagated_neg[pos_mask]
            adjusted_weights[pos_mask] = torch.clamp(
                weights[pos_mask] - infection, min=0.01
            )

            num_suppressed = (infection > 0.01).sum().item()
            if num_suppressed > 0:
                logger.debug(
                    f"[DCG] {num_suppressed} positive concept(s) suppressed "
                    f"by cross-polarity propagation"
                )

        if neg_mask.any():
            # Negative concepts: dùng tín hiệu đã lan truyền
            # (cộng hưởng từ neighbor cùng chiều negative)
            adjusted_weights[neg_mask] = propagated_neg[neg_mask]

        # ── Tách kết quả theo polarity ──
        pos_vecs = vectors[pos_mask] if pos_mask.any() else None
        pos_w = adjusted_weights[pos_mask] if pos_mask.any() else None
        neg_vecs = vectors[neg_mask] if neg_mask.any() else None
        neg_w = adjusted_weights[neg_mask] if neg_mask.any() else None

        return pos_vecs, pos_w, neg_vecs, neg_w


def synthesize_query_from_graph(
    q_t: torch.Tensor,
    pos_vecs: Optional[torch.Tensor],
    pos_weights: Optional[torch.Tensor],
    neg_vecs: Optional[torch.Tensor],
    neg_weights: Optional[torch.Tensor],
    alpha: float,
    beta: float,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Tổng hợp query vector từ kết quả graph propagation.

    Công thức y hệt ConceptMemoryBoard.synthesize_query() nhưng nhận
    vectors/weights từ bên ngoài (từ ConceptGraph.propagate) thay vì
    tính nội bộ.

        q_new = Normalize(
            q_t
            + α · Normalize(Σ w_i · v_i)   cho i ∈ Positive
            − β · Normalize(Σ w_j · v_j)   cho j ∈ Negative
        )

    Args:
        q_t: [D] query vector hiện tại
        pos_vecs: [P, D] positive concept vectors (hoặc None)
        pos_weights: [P] positive weights (hoặc None)
        neg_vecs: [Q, D] negative concept vectors (hoặc None)
        neg_weights: [Q] negative weights (hoặc None)
        alpha: trọng số positive
        beta: trọng số negative
        normalize: L2-normalize output

    Returns:
        q_new: [D]
    """
    q_new = q_t.clone()

    if pos_vecs is not None and pos_weights is not None and pos_vecs.shape[0] > 0:
        weighted_pos = (pos_weights.unsqueeze(-1) * pos_vecs).sum(dim=0)
        weighted_pos = F.normalize(weighted_pos, dim=-1)
        q_new = q_new + alpha * weighted_pos

    if neg_vecs is not None and neg_weights is not None and neg_vecs.shape[0] > 0:
        weighted_neg = (neg_weights.unsqueeze(-1) * neg_vecs).sum(dim=0)
        weighted_neg = F.normalize(weighted_neg, dim=-1)
        q_new = q_new - beta * weighted_neg

    if normalize:
        q_new = F.normalize(q_new.unsqueeze(0), dim=-1).squeeze(0)

    return q_new
