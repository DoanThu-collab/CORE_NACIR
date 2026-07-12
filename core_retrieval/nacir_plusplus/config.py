"""
NACIR++ Plug-and-Play — Config
================================
GIỮ NGUYÊN 100% các tham số & công thức gốc của NACIR++ (memory_alpha, beta,
positive_blend_alpha, ortho_strength, masking_penalty_weight, masking_threshold,
recency_decay, concept_match_threshold, mode).

Điểm khác biệt so với bản gốc (query_update.py cũ): tách phần "lịch trình động"
(dynamic scheduling của alpha/beta/ortho/penalty/itm_weight theo turn — vốn bị
hardcode ngay trong main.py) ra thành một đối tượng cấu hình riêng
(`DynamicScheduleConfig`) + hàm thuần túy `default_dynamic_schedule`.

=> Công thức TOÁN HỌC không đổi một chữ, chỉ đổi chỗ ở (từ "in-line trong vòng lặp"
   sang "một hàm pluggable"), để bất kỳ pipeline nào khác cũng có thể:
      - dùng đúng lịch trình gốc của NACIR++ (mặc định), hoặc
      - tự cấp lịch trình khác (ví dụ số vòng hội thoại khác 11, hoặc muốn tắt
        dynamic scheduling hoàn toàn) mà KHÔNG phải sửa core logic.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional


@dataclass
class NACIRPlusPlusConfig:
    """
    Cấu hình hợp nhất cho toàn bộ các bước của NACIR++.
    (Giữ nguyên 1:1 so với core/query_update.py bản gốc)

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

    def __repr__(self):
        return (
            f"NACIRPlusPlusConfig(mode={self.mode}, "
            f"α_mem={self.memory_alpha}, β_mem={self.memory_beta}, "
            f"α_pos={self.positive_blend_alpha}, "
            f"ortho_str={self.ortho_strength}, "
            f"mask_w={self.masking_penalty_weight}, "
            f"mask_τ={self.masking_threshold})"
        )


@dataclass
class DynamicScheduleConfig:
    """
    Tham số hóa lịch trình động y hệt main.py gốc (KHÔNG đổi công thức):

        progress   = min((t - 1) / warmup_turns, 1.0)
        alpha(t)   = alpha_start + (alpha_end - alpha_start) * progress
        beta(t)    = alpha(t) * beta_ratio
        ortho(t)   = ortho_start + (ortho_end - ortho_start) * progress
        penalty(t) = penalty_start + (penalty_end - penalty_start) * progress

        progress_itm = min(t / itm_warmup_turns, 1.0)
        itm_weight(t) = itm_start + (itm_end - itm_start) * progress_itm

    Mặc định các số y hệt bản hardcode trong main.py (0.20→0.60, 0.05→0.25,
    0.05→0.20, 0.2→0.7, warmup 9/10 turns).
    """
    enabled: bool = True

    alpha_start: float = 0.20
    alpha_end: float = 0.60
    beta_ratio: float = 0.5
    ortho_start: float = 0.05
    ortho_end: float = 0.25
    penalty_start: float = 0.05
    penalty_end: float = 0.20
    warmup_turns: float = 9.0  # (t-1)/warmup_turns, turn 1 -> progress 0

    itm_start: float = 0.2
    itm_end: float = 0.7
    itm_warmup_turns: float = 10.0  # t/itm_warmup_turns


def default_dynamic_schedule(
    turn: int, schedule: DynamicScheduleConfig
) -> Dict[str, float]:
    """
    Trả về override cho turn hiện tại: {"memory_alpha", "memory_beta",
    "ortho_strength", "masking_penalty_weight", "itm_weight"}.

    turn=0 luôn trả None-ish (không override, vì turn 0 chưa có beliefs).
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


# Kiểu hàm schedule tùy biến — bất kỳ phương pháp nào khác cũng có thể cắm
# hàm lịch trình của riêng họ vào Pipeline mà không đụng tới core logic.
ScheduleFn = Callable[[int], Dict[str, float]]


# Tham số tối ưu gốc của NACIR++ (đã tune qua Optuna, BRI = 0.6861)
# Giữ nguyên y hệt main.py để đảm bảo tái lập kết quả.
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
