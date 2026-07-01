"""
NACIR++ Plug-and-Play — Ví dụ 2: Cắm vào một phương pháp HOÀN TOÀN KHÁC
============================================================================
Mục đích của file này: CHỨNG MINH rằng NACIR++ giờ là plug-and-play thật sự.
Ở đây ta không dùng BLIP, không dùng PlugIR, không dùng VisDial — chỉ dùng
một "text encoder" giả lập (random hash embedding) và một corpus ảnh giả lập,
đại diện cho MỘT PHƯƠNG PHÁP TÌM KIẾM ẢNH TƯƠNG TÁC KHÁC BẤT KỲ.

Miễn phương pháp đó cung cấp đúng theo interfaces.py / schema.py:
    - TextEncoder    (encode câu hỏi -> vector)
    - BeliefSource    (positive/negative concept mỗi turn)
    - corpus_vectors [N, D]

... thì NACIR++ hoạt động y hệt, KHÔNG cần sửa một dòng nào trong core/.

Chạy:
    python examples/run_generic_demo.py
"""

import os
import sys
from typing import Any, List

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nacir_plusplus import NACIRPlusPlusPipeline, NACIRPlusPlusConfig
from nacir_plusplus.adapters.belief_sources import RuleBasedBeliefSource
from nacir_plusplus.schema import DialogTurn, RetrievalSession


# ============================================================
# "Phương pháp khác" giả lập — thay bằng backbone thật của bạn ở đây
# ============================================================

class DummyHashTextEncoder:
    """TextEncoder giả lập: hash từng từ -> vector cố định (deterministic).
    Trong thực tế, đây chính là chỗ bạn cắm CLIP/SigLIP/hoặc encoder nội bộ
    của phương pháp bạn đang có."""

    def __init__(self, dim: int = 64, seed: int = 0):
        self.dim = dim
        self.seed = seed

    def _word_vector(self, word: str) -> torch.Tensor:
        g = torch.Generator().manual_seed(hash((word, self.seed)) % (2**31))
        return torch.randn(self.dim, generator=g)

    def encode(self, texts: List[str]) -> torch.Tensor:
        vecs = []
        for text in texts:
            words = text.lower().split()
            if not words:
                v = torch.zeros(self.dim)
            else:
                v = torch.stack([self._word_vector(w) for w in words]).mean(dim=0)
            vecs.append(F.normalize(v, dim=-1))
        return torch.stack(vecs)


def build_dummy_corpus(num_images: int, dim: int, encoder: DummyHashTextEncoder, seed: int = 42):
    """Corpus ảnh giả lập: mỗi 'ảnh' thực chất là vector sinh từ 1 caption giả,
    để ta biết trước ảnh nào khớp với concept nào (phục vụ demo có ý nghĩa)."""
    captions = [
        "a red backpack on a wooden table",
        "a black backpack near a window",
        "a blue bicycle parked outside",
        "a red bicycle leaning on a wall",
        "a wooden chair in an empty room",
        "a metal chair next to a red backpack",
        "a black bicycle without a basket",
        "a red backpack with no zipper",
    ]
    torch.manual_seed(seed)
    reps = (num_images // len(captions)) + 1
    all_captions = (captions * reps)[:num_images]
    vectors = encoder.encode(all_captions)
    return all_captions, vectors


def main():
    dim = 64
    encoder = DummyHashTextEncoder(dim=dim)
    captions, corpus_vectors = build_dummy_corpus(num_images=8, dim=dim, encoder=encoder)

    print("Corpus giả lập (mỗi dòng = 1 'ảnh'):")
    for i, c in enumerate(captions):
        print(f"  [{i}] {c}")

    # Target: ảnh index 0 — "a red backpack on a wooden table"
    target_index = 0

    # Hội thoại tương tác giả lập:
    #   Turn 0: caption ban đầu
    #   Turn 1: "Is it black?" -> "No, it's not black" (negative: black)
    #   Turn 2: "Is it on a table?" -> "Yes, on a wooden table" (positive: table)
    dialog_qas = [
        ("", "a backpack"),  # turn 0: caption
        ("Is it black?", "No, it's not black"),
        ("Is it on a table?", "Yes, on a wooden table"),
    ]

    turns = []
    for t, (q, a) in enumerate(dialog_qas):
        turns.append(DialogTurn(turn_index=t, query_text=a, question=q, answer=a))

    session = RetrievalSession(session_id="demo-0", turns=turns, target_index=target_index)

    config = NACIRPlusPlusConfig(
        memory_alpha=0.5, memory_beta=0.25, ortho_strength=0.3,
        masking_penalty_weight=0.2, masking_threshold=0.2, mode="full",
    )

    pipeline = NACIRPlusPlusPipeline(
        config=config,
        corpus_vectors=corpus_vectors,
        text_encoder=encoder,               # <- backbone khác hoàn toàn với BLIP
        belief_source=RuleBasedBeliefSource(),  # <- rule-based, không cần LLM
        image_scorer=None,                  # <- phương pháp này không có bước re-rank
        top_k=5,
    )

    result = pipeline.run_session(session)

    print("\nKết quả từng turn (rank 0 = khớp nhất với ảnh đích):")
    for turn_out in result.turns:
        print(
            f"  Turn {turn_out.turn_index}: target_rank={turn_out.target_rank}, "
            f"top_k={turn_out.top_k_indices}"
        )

    print(
        "\n(Đây là demo 1 session để minh hoạ pipeline hoạt động; muốn tính "
        "Hits@K/Recall@K/BRI thật, hãy dùng compute_metrics() trên nhiều "
        "session như trong examples/run_plugir_visdial.py)"
    )


if __name__ == "__main__":
    main()
