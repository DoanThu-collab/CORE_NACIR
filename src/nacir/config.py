"""
NACIR++ Improved — Config (Phiên bản V2)
==========================================
Cấu hình hợp nhất cho toàn bộ pipeline NACIR++ gồm:
  - Core: Concept Memory, Orthogonal Projection, Masking, Reranking
  - Đề xuất cũ: Semantic-Aware Scheduling, Memory Roll-back
  - ĐỀ XUẤT MỚI 1: Dynamic Concept Graph (DCG)
  - ĐỀ XUẤT MỚI 2: Visual-Grounded Belief Refinement
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional


@dataclass
class NACIRPlusPlusConfig:
    """
    Cấu hình hợp nhất cho toàn bộ các bước của NACIR++.

    Ablation modes:
        "full"       — bật tất cả các bước
        "no_ortho"   — tắt Step 3 (Orthogonal Projection)
        "no_mask"    — tắt Step 4 (Attention Masking)
        "no_memory"  — tắt Step 2 (Concept Memory), dùng thẳng query gốc
    """
    # Step 2: Concept Memory
    memory_alpha: float = 0.30
    memory_beta: float = 0.15
    recency_decay: float = 0.1
    concept_match_threshold: float = 0.85
    max_concepts: int = 50

    # Step 3: Orthogonal Projection
    positive_blend_alpha: float = 0.30
    ortho_strength: float = 1.0
    use_gram_schmidt: bool = True

    # Step 4: Masking / Penalty
    masking_penalty_weight: float = 0.15
    masking_threshold: float = 0.20

    # Ablation mode
    mode: str = "full"

    # ── Semantic-Aware Scheduling ──
    use_semantic_scheduler: bool = False

    # ── Memory Roll-back ──
    use_memory_rollback: bool = False
    rollback_score_drop: float = 0.05
    rollback_top_k: int = 50

    # ══════════════════════════════════════════════════
    # ĐỀ XUẤT MỚI 1: Dynamic Concept Graph (DCG)
    # ══════════════════════════════════════════════════
    use_concept_graph: bool = False
    graph_propagation_alpha: float = 0.3     # Cường độ lan truyền (0=tắt, 1=chỉ neighbor)
    graph_similarity_threshold: float = 0.50  # Cosine sim tối thiểu để tạo cạnh
    graph_num_hops: int = 1                   # Số bước lan truyền
    graph_evolving: bool = False              # Turn-Evolving Graph (temporal smoothing)
    graph_temporal_gamma: float = 0.3         # Tỷ lệ pha trộn đồ thị mới vào cũ
    graph_bimodal: bool = False               # Bimodal Concept Node (visual grounding)
    graph_bimodal_lambda: float = 0.2         # Tỷ lệ pha trộn visual vào text
    graph_bimodal_top_k: int = 10             # Số ảnh top-K để visual grounding

    # ══════════════════════════════════════════════════
    # ĐỀ XUẤT MỚI 2: Visual-Grounded Belief Refinement
    # ══════════════════════════════════════════════════
    use_visual_feedback: bool = False
    vf_top_k: int = 50                        # Số ảnh top-K để phân tích
    vf_suppress_threshold: float = 0.15       # Positive relevance < τ → suppress
    vf_boost_threshold: float = 0.25          # Negative relevance > τ → boost
    vf_suppress_factor: float = 0.3           # Mức giảm confidence khi suppress
    vf_boost_factor: float = 0.2              # Mức tăng confidence khi boost

    def __repr__(self):
        return (
            f"NACIRPlusPlusConfig(mode={self.mode}, "
            f"α_mem={self.memory_alpha}, β_mem={self.memory_beta}, "
            f"ortho_str={self.ortho_strength}, "
            f"mask_w={self.masking_penalty_weight}, "
            f"DCG={self.use_concept_graph}, "
            f"VisualFB={self.use_visual_feedback})"
        )


@dataclass
class DynamicScheduleConfig:
    """
    Tham số hóa lịch trình động.
    Công thức giữ nguyên 100% logic gốc trong main.py.
    """
    enabled: bool = True

    alpha_start: float = 0.20
    alpha_end: float = 0.60
    beta_ratio: float = 0.5
    ortho_start: float = 0.05
    ortho_end: float = 0.25
    penalty_start: float = 0.05
    penalty_end: float = 0.20
    warmup_turns: float = 9.0

    itm_start: float = 0.2
    itm_end: float = 0.7
    itm_warmup_turns: float = 10.0


def default_dynamic_schedule(
    turn: int, schedule: DynamicScheduleConfig
) -> Dict[str, float]:
    """
    Trả về override cho turn hiện tại.
    Công thức giữ nguyên 100% logic gốc trong main.py.
    """
    progress_itm = min(turn / schedule.itm_warmup_turns, 1.0)
    itm_weight = schedule.itm_start + (schedule.itm_end - schedule.itm_start) * progress_itm

    overrides: Dict[str, float] = {"itm_weight": itm_weight}

    if turn > 0:
        progress = min((turn - 1) / schedule.warmup_turns, 1.0)
        alpha = schedule.alpha_start + (schedule.alpha_end - schedule.alpha_start) * progress
        beta = alpha * schedule.beta_ratio
        ortho = schedule.ortho_start + (schedule.ortho_end - schedule.ortho_start) * progress
        penalty = schedule.penalty_start + (schedule.penalty_end - schedule.penalty_start) * progress

        overrides.update(
            {
                "memory_alpha": alpha,
                "memory_beta": beta,
                "ortho_strength": ortho,
                "masking_penalty_weight": penalty,
            }
        )

    return overrides


def semantic_aware_schedule(
    turn: int,
    schedule: DynamicScheduleConfig,
    beliefs=None,
) -> Dict[str, float]:
    """Semantic-Aware Schedule: scale theo confidence từ beliefs."""
    overrides = default_dynamic_schedule(turn, schedule)

    if beliefs is None or turn == 0:
        return overrides

    all_confidences = []
    for b in beliefs.positive_beliefs:
        all_confidences.append(b.confidence)
    for b in beliefs.negative_beliefs:
        all_confidences.append(b.confidence)

    if not all_confidences:
        return overrides

    max_conf = max(all_confidences)
    for key in ("memory_alpha", "memory_beta", "ortho_strength", "masking_penalty_weight"):
        if key in overrides:
            overrides[key] *= max_conf

    return overrides


ScheduleFn = Callable[[int], Dict[str, float]]


# Tham số tối ưu gốc
OPTIMAL_CONFIG = NACIRPlusPlusConfig(
    memory_alpha=0.55,
    memory_beta=0.275,
    positive_blend_alpha=0.55,
    ortho_strength=0.2,
    masking_penalty_weight=0.18,
    masking_threshold=0.25,
    recency_decay=0.1,
    concept_match_threshold=0.75,
    mode="full",
)

OPTIMAL_SCHEDULE = DynamicScheduleConfig()
