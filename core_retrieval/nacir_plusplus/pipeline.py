"""
NACIR++ Plug-and-Play — Pipeline (bộ điều phối chính)
=========================================================
Đây là phiên bản Plug-and-Play của `core/query_update.py::NACIRPlusPlusBatchUpdater`
+ vòng lặp retrieval trong `main.py` gốc, GIỮ NGUYÊN 100% công thức toán học:

    Step 2 (Concept Memory)        -> core/concept_memory.py       (không đổi)
    Step 3 (Orthogonal Projection) -> core/orthogonal_projection.py (không đổi)
    Step 4 (Attention Masking)     -> core/attention_masking.py     (không đổi)
    Step 5 (ITM/Cross-Encoder rerank) -> core/reranker.py (đã tổng quát hoá scorer)

Điều đã thay đổi: KHÔNG còn hardcode BLIP / PlugIR / VisDial / 11 rounds.
Thay vào đó Pipeline nhận:
    - corpus_vectors:  [N, D]  (từ BẤT KỲ retrieval backbone nào)
    - text_encoder:    TextEncoder  (bất kỳ)
    - belief_source:   BeliefSource (tuỳ chọn — có thể tự cấp beliefs sẵn)
    - image_scorer:    ImageScorer  (tuỳ chọn — bỏ qua nếu method không cần rerank)
    - schedule_fn:      hàm lịch trình động (mặc định = lịch trình gốc NACIR++)

=> Method nào cũng cắm được, miễn tuân theo schema.py / interfaces.py.
"""

import copy
import logging
from typing import Any, Callable, List, Optional

import torch

from .config import (
    DynamicScheduleConfig,
    NACIRPlusPlusConfig,
    ScheduleFn,
    default_dynamic_schedule,
)
from .core.attention_masking import apply_enhanced_penalty
from .core.concept_memory import ConceptMemoryBoard, ConceptMemoryConfig
from .core.orthogonal_projection import orthogonal_purge
from .core.reranker import rerank_topk_with_lookup
from .interfaces import BeliefSource, ImageScorer, NullImageScorer, TextEncoder
from .schema import BeliefBundle, RetrievalSession, SessionOutput, TurnOutput

logger = logging.getLogger(__name__)


class NACIRPlusPlusPipeline:
    """
    Bộ điều phối Plug-and-Play. Xử lý MỘT session hội thoại tại một thời điểm
    (gọi `run_session`), hoặc nhiều session cùng lúc (gọi `run_batch`, xử lý
    tuần tự bên trong nhưng dùng chung corpus_vectors đã nằm sẵn trên device).

    Kiến trúc pipeline giữ nguyên các bước gốc:
        query_text/query_vector -> [Step 2: Concept Memory]
                                  -> [Step 3: Orthogonal Projection]
                                  -> scores = q · corpus_vectors.T
                                  -> [Step 4: Attention Masking trên scores]
                                  -> [Step 5 (tuỳ chọn): ITM/cross-encoder rerank Top-K]
    """

    def __init__(
        self,
        config: NACIRPlusPlusConfig,
        corpus_vectors: torch.Tensor,
        text_encoder: Optional[TextEncoder] = None,
        belief_source: Optional[BeliefSource] = None,
        image_scorer: Optional[ImageScorer] = None,
        corpus_ref_lookup: Optional[Callable[[int], Any]] = None,
        rerank_k: int = 50,
        top_k: int = 10,
        schedule_fn: Optional[ScheduleFn] = None,
        device: Optional[str] = None,
    ):
        self.config = config
        self.corpus_vectors = corpus_vectors
        self.device = device or str(corpus_vectors.device)
        self.text_encoder = text_encoder
        self.belief_source = belief_source
        self.image_scorer = image_scorer or NullImageScorer()
        self.corpus_ref_lookup = corpus_ref_lookup
        self.rerank_k = rerank_k
        self.top_k = top_k

        # Mặc định dùng đúng lịch trình động gốc của NACIR++ (main.py).
        if schedule_fn is None:
            sched_cfg = DynamicScheduleConfig()
            self.schedule_fn: ScheduleFn = lambda t: default_dynamic_schedule(t, sched_cfg)
        else:
            self.schedule_fn = schedule_fn

    # ------------------------------------------------------------------
    # API chính
    # ------------------------------------------------------------------

    def run_batch(self, sessions: List[RetrievalSession]) -> List[SessionOutput]:
        return [self.run_session(s) for s in sessions]

    def run_session(self, session: RetrievalSession) -> SessionOutput:
        mem_config = ConceptMemoryConfig(
            alpha=self.config.memory_alpha,
            beta=self.config.memory_beta,
            recency_decay=self.config.recency_decay,
            concept_match_threshold=self.config.concept_match_threshold,
            max_concepts=self.config.max_concepts,
        )
        board = ConceptMemoryBoard(
            config=copy.deepcopy(mem_config),
            encoder=self._encoder_fn(),
        )
        # Config cục bộ cho session này (để dynamic schedule không rò rỉ giữa các session)
        cfg = copy.deepcopy(self.config)

        turn_outputs: List[TurnOutput] = []

        for turn in sorted(session.turns, key=lambda t: t.turn_index):
            t = turn.turn_index

            overrides = self.schedule_fn(t)
            itm_weight = overrides.pop("itm_weight", 0.7)
            for key, val in overrides.items():
                if key in ("memory_alpha", "memory_beta"):
                    setattr(board.config, key.replace("memory_", ""), val)
                elif hasattr(cfg, key):
                    setattr(cfg, key, val)

            # ── Lấy query vector (đã sẵn hoặc cần encode) ──
            q_t = self._resolve_query_vector(turn)

            # ── Lấy beliefs (đã sẵn hoặc cần trích xuất) ──
            beliefs = self._resolve_beliefs(session.session_id, turn)

            # ── Step 2: Concept Memory ──
            if cfg.mode != "no_memory" and t > 0:
                positives = [
                    {"attribute": b.attribute, "confidence": b.confidence}
                    for b in beliefs.positive_beliefs
                ]
                negatives = [
                    {"attribute": b.attribute, "confidence": b.confidence}
                    for b in beliefs.negative_beliefs
                ]
                board.add_concepts(positives, negatives, t)
                q_t = board.synthesize_query(q_t)

            # ── Step 3: Orthogonal Projection ──
            if cfg.mode != "no_ortho" and t > 0:
                neg_vecs, neg_weights = board.get_negative_vectors()
                if neg_vecs is not None and neg_vecs.shape[0] > 0:
                    q_t = orthogonal_purge(
                        q_t,
                        neg_vecs,
                        neg_weights,
                        strength=cfg.ortho_strength,
                        use_gram_schmidt=cfg.use_gram_schmidt,
                    )

            # ── Scoring ──
            scores = (q_t.unsqueeze(0) @ self.corpus_vectors.T).squeeze(0)

            # ── Step 4: Attention Masking (Global Fallback penalty) ──
            if cfg.mode != "no_mask" and t > 0:
                neg_vecs, neg_weights = board.get_negative_vectors()
                if neg_vecs is not None and neg_vecs.shape[0] > 0:
                    scores = apply_enhanced_penalty(
                        scores.unsqueeze(0),
                        self.corpus_vectors,
                        neg_vecs,
                        neg_weights,
                        tau=cfg.masking_threshold,
                        max_penalty=cfg.masking_penalty_weight,
                        soft=True,
                        temperature=0.1,
                    ).squeeze(0)

            ranked = torch.argsort(scores, descending=True)

            # ── Step 5 (tuỳ chọn): Re-ranking bằng cross-encoder ──
            top_rerank = ranked[: self.rerank_k]
            if self.corpus_ref_lookup is not None and not isinstance(self.image_scorer, NullImageScorer):
                reranked_indices, _ = rerank_topk_with_lookup(
                    query_text=turn.query_text,
                    top_k_corpus_indices=top_rerank.cpu(),
                    corpus_ref_lookup=self.corpus_ref_lookup,
                    image_scorer=self.image_scorer,
                    cosine_scores=scores[top_rerank].cpu(),
                    itm_weight=itm_weight,
                )
                full_ranked = torch.cat([reranked_indices.to(ranked.device), ranked[self.rerank_k:]])
            else:
                full_ranked = ranked

            target_rank = None
            if session.target_index is not None:
                target_rank = (full_ranked == session.target_index).nonzero(as_tuple=True)[0].item()

            turn_outputs.append(
                TurnOutput(
                    turn_index=t,
                    query_vector=q_t.detach().cpu(),
                    scores=scores.detach().cpu(),
                    ranked_indices=full_ranked.detach().cpu(),
                    top_k_indices=full_ranked[: self.top_k].cpu().tolist(),
                    target_rank=target_rank,
                    memory_snapshot=board.get_memory_snapshot(),
                )
            )

        return SessionOutput(session_id=session.session_id, turns=turn_outputs)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _encoder_fn(self):
        """Adapter nội bộ: ConceptMemoryBoard cần encoder(List[str]) -> Tensor[B,D]."""
        if self.text_encoder is None:
            return None
        return lambda texts: self.text_encoder.encode(texts)

    def _resolve_query_vector(self, turn) -> torch.Tensor:
        if turn.query_vector is not None:
            return turn.query_vector.to(self.corpus_vectors.device)
        if self.text_encoder is None:
            raise ValueError(
                "DialogTurn không có query_vector và Pipeline không được cấp text_encoder. "
                "Hãy truyền query_vector trực tiếp hoặc cấp một TextEncoder."
            )
        vec = self.text_encoder.encode([turn.query_text])[0]
        return vec.to(self.corpus_vectors.device)

    def _resolve_beliefs(self, session_id: Any, turn) -> BeliefBundle:
        if turn.beliefs is not None:
            return turn.beliefs
        if self.belief_source is None:
            return BeliefBundle.empty()
        return self.belief_source.get_beliefs(
            session_id=session_id,
            turn_index=turn.turn_index,
            question=turn.question,
            answer=turn.answer,
        )
