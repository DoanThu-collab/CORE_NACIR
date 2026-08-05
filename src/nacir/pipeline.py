"""
NACIR++ Improved — Pipeline V2 (bộ điều phối chính)
=====================================================
Tích hợp trực tiếp 4 đề xuất cải tiến VÀO BÊN TRONG vòng lặp run_session:

  1. Semantic-Aware Dynamic Scheduling (từ V1)
  2. Memory Roll-back (từ V1)
  3. ĐỀ XUẤT MỚI: Dynamic Concept Graph (DCG) — chống Semantic Leakage
  4. ĐỀ XUẤT MỚI: Visual-Grounded Belief Refinement — closed-loop feedback
"""

import copy
import logging
from typing import Any, Callable, List, Optional

import torch
import torch.nn.functional as F

from .config import (
    DynamicScheduleConfig,
    NACIRPlusPlusConfig,
    ScheduleFn,
    default_dynamic_schedule,
    semantic_aware_schedule,
)
from .core.attention_masking import apply_enhanced_penalty
from .core.concept_memory import ConceptMemoryBoard, ConceptMemoryConfig
from .core.concept_graph import ConceptGraph, synthesize_query_from_graph
from .core.visual_feedback import VisualFeedbackRefiner
from .core.orthogonal_projection import orthogonal_purge
from .core.reranker import rerank_topk_with_lookup
from .interfaces import BeliefSource, ImageScorer, NullImageScorer, TextEncoder
from .schema import BeliefBundle, RetrievalSession, SessionOutput, TurnOutput

logger = logging.getLogger(__name__)


class NACIRPlusPlusPipeline:
    """
    Bộ điều phối Plug-and-Play (Phiên bản V2).

    Tích hợp trực tiếp:
      - Dynamic Concept Graph: thay thế bảng phẳng bằng đồ thị ngữ nghĩa
      - Visual Feedback: closed-loop refinement từ kết quả retrieval
      - (Tùy chọn) Semantic-Aware Scheduling & Memory Roll-back
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
        cfg = copy.deepcopy(self.config)

        # ── Khởi tạo các module mới (nếu bật) ──
        concept_graph = None
        if cfg.use_concept_graph:
            concept_graph = ConceptGraph(
                propagation_alpha=cfg.graph_propagation_alpha,
                similarity_threshold=cfg.graph_similarity_threshold,
                num_hops=cfg.graph_num_hops,
                evolving=cfg.graph_evolving,
                temporal_gamma=cfg.graph_temporal_gamma,
                bimodal=cfg.graph_bimodal,
                bimodal_lambda=cfg.graph_bimodal_lambda,
                bimodal_top_k=cfg.graph_bimodal_top_k,
            )

        visual_refiner = None
        if cfg.use_visual_feedback:
            visual_refiner = VisualFeedbackRefiner(
                feedback_top_k=cfg.vf_top_k,
                suppress_threshold=cfg.vf_suppress_threshold,
                boost_threshold=cfg.vf_boost_threshold,
                suppress_factor=cfg.vf_suppress_factor,
                boost_factor=cfg.vf_boost_factor,
            )

        turn_outputs: List[TurnOutput] = []
        prev_top_k_mean: Optional[float] = None
        rollback_count = 0
        feedback_count = 0

        for turn in sorted(session.turns, key=lambda t: t.turn_index):
            t = turn.turn_index

            # ── Lấy query vector và beliefs ──
            q_t_original = self._resolve_query_vector(turn)
            q_t = q_t_original.clone()
            beliefs = self._resolve_beliefs(session.session_id, turn)

            # ── Dynamic Schedule ──
            if self.schedule_fn is None:
                sched_cfg = DynamicScheduleConfig()
                if cfg.use_semantic_scheduler:
                    overrides = semantic_aware_schedule(t, sched_cfg, beliefs)
                else:
                    overrides = default_dynamic_schedule(t, sched_cfg)
            else:
                try:
                    overrides = self.schedule_fn(t, q_t, beliefs, self._encoder_fn())
                except TypeError:
                    overrides = self.schedule_fn(t)

                if cfg.use_semantic_scheduler and beliefs is not None and t > 0:
                    all_confs = [b.confidence for b in beliefs.positive_beliefs] + \
                                [b.confidence for b in beliefs.negative_beliefs]
                    if all_confs:
                        conf_scale = max(all_confs)
                        for key in ("memory_alpha", "memory_beta",
                                    "ortho_strength", "masking_penalty_weight"):
                            if key in overrides:
                                overrides[key] *= conf_scale

            itm_weight = overrides.pop("itm_weight", 0.7)
            for key, val in overrides.items():
                if key in ("memory_alpha", "memory_beta"):
                    setattr(board.config, key.replace("memory_", ""), val)
                elif hasattr(cfg, key):
                    setattr(cfg, key, val)

            # ── Memory Roll-back: Lưu snapshot ──
            board_snapshot = None
            if cfg.use_memory_rollback and t > 0:
                board_snapshot = board.save_state()

            # ══════════════════════════════════════════════
            # Step 2: Concept Memory — thêm concepts vào bảng nhớ
            # ══════════════════════════════════════════════
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

                # ══════════════════════════════════════════════
                # ĐỀ XUẤT MỚI 1: Dynamic Concept Graph (DCG)
                # ══════════════════════════════════════════════
                if concept_graph is not None:
                    # Dùng graph propagation thay vì bảng phẳng
                    # Bimodal cần scores từ turn trước để lấy top-K ảnh
                    prev_scores = None
                    if concept_graph.bimodal and t > 1:
                        # Tính scores nhanh (chưa có ortho/mask) để lấy top-K cho bimodal
                        prev_scores = (q_t.unsqueeze(0) @ self.corpus_vectors.T).squeeze(0)
                    pos_vecs, pos_w, neg_vecs, neg_w = concept_graph.propagate(
                        board,
                        corpus_vectors=self.corpus_vectors if concept_graph.bimodal else None,
                        scores=prev_scores,
                    )

                    # Synthesize query từ kết quả graph
                    q_t = synthesize_query_from_graph(
                        q_t, pos_vecs, pos_w, neg_vecs, neg_w,
                        alpha=board.config.alpha,
                        beta=board.config.beta,
                        normalize=board.config.normalize_query,
                    )
                else:
                    # Dùng bảng phẳng gốc
                    q_t = board.synthesize_query(q_t)
                    neg_vecs, neg_w = board.get_negative_vectors()
            else:
                neg_vecs, neg_w = None, None

            # ══════════════════════════════════════════════
            # Step 3: Orthogonal Projection
            # ══════════════════════════════════════════════
            if cfg.mode != "no_ortho" and t > 0:
                # Lấy neg vectors: ưu tiên từ DCG (đã propagate), fallback sang board
                if concept_graph is not None:
                    # neg_vecs, neg_w đã có từ graph propagation ở trên
                    pass
                else:
                    neg_vecs, neg_w = board.get_negative_vectors()

                if neg_vecs is not None and neg_vecs.shape[0] > 0:
                    q_t = orthogonal_purge(
                        q_t, neg_vecs, neg_w,
                        strength=cfg.ortho_strength,
                        use_gram_schmidt=cfg.use_gram_schmidt,
                    )

            # ── Scoring ──
            scores = (q_t.unsqueeze(0) @ self.corpus_vectors.T).squeeze(0)

            # ══════════════════════════════════════════════
            # Step 4: Attention Masking
            # ══════════════════════════════════════════════
            if cfg.mode != "no_mask" and t > 0:
                if concept_graph is None:
                    neg_vecs, neg_w = board.get_negative_vectors()
                # neg_vecs, neg_w từ DCG đã có ở trên nếu DCG bật

                if neg_vecs is not None and neg_vecs.shape[0] > 0:
                    scores = apply_enhanced_penalty(
                        scores.unsqueeze(0),
                        self.corpus_vectors,
                        neg_vecs, neg_w,
                        tau=cfg.masking_threshold,
                        max_penalty=cfg.masking_penalty_weight,
                        soft=True, temperature=0.1,
                    ).squeeze(0)

            # ══════════════════════════════════════════════════════════
            # ĐỀ XUẤT MỚI 2: Visual-Grounded Belief Refinement
            # ══════════════════════════════════════════════════════════
            if visual_refiner is not None and t > 0 and cfg.mode != "no_memory":
                adjustments = visual_refiner.refine(
                    board, self.corpus_vectors, scores
                )
                if adjustments:
                    feedback_count += len(adjustments)

                    # Áp dụng điều chỉnh confidence
                    visual_refiner.apply_adjustments(board, adjustments)

                    # ── Re-compute: synthesize → ortho → score → mask ──
                    q_t = q_t_original.clone()

                    if concept_graph is not None:
                        pos_vecs, pos_w, neg_vecs, neg_w = concept_graph.propagate(board)
                        q_t = synthesize_query_from_graph(
                            q_t, pos_vecs, pos_w, neg_vecs, neg_w,
                            alpha=board.config.alpha,
                            beta=board.config.beta,
                            normalize=board.config.normalize_query,
                        )
                    else:
                        q_t = board.synthesize_query(q_t)
                        neg_vecs, neg_w = board.get_negative_vectors()

                    if cfg.mode != "no_ortho":
                        if concept_graph is None:
                            neg_vecs, neg_w = board.get_negative_vectors()
                        if neg_vecs is not None and neg_vecs.shape[0] > 0:
                            q_t = orthogonal_purge(
                                q_t, neg_vecs, neg_w,
                                strength=cfg.ortho_strength,
                                use_gram_schmidt=cfg.use_gram_schmidt,
                            )

                    scores = (q_t.unsqueeze(0) @ self.corpus_vectors.T).squeeze(0)

                    if cfg.mode != "no_mask":
                        if concept_graph is None:
                            neg_vecs, neg_w = board.get_negative_vectors()
                        if neg_vecs is not None and neg_vecs.shape[0] > 0:
                            scores = apply_enhanced_penalty(
                                scores.unsqueeze(0),
                                self.corpus_vectors,
                                neg_vecs, neg_w,
                                tau=cfg.masking_threshold,
                                max_penalty=cfg.masking_penalty_weight,
                                soft=True, temperature=0.1,
                            ).squeeze(0)

            # ── Memory Roll-back: Kiểm tra & Phục hồi ──
            if cfg.use_memory_rollback and t > 0 and board_snapshot is not None:
                top_k_scores = torch.topk(scores, cfg.rollback_top_k).values
                current_top_k_mean = top_k_scores.mean().item()

                if prev_top_k_mean is not None:
                    score_drop = prev_top_k_mean - current_top_k_mean
                    if score_drop > cfg.rollback_score_drop:
                        logger.info(
                            f"[ROLLBACK] Turn {t}: score drop {score_drop:.4f} > "
                            f"threshold {cfg.rollback_score_drop:.4f}. "
                            f"Rolling back memory."
                        )
                        rollback_count += 1
                        board.restore_state(board_snapshot)

                        q_t = q_t_original.clone()
                        if cfg.mode != "no_memory":
                            q_t = board.synthesize_query(q_t)

                        if cfg.mode != "no_ortho":
                            nv, nw = board.get_negative_vectors()
                            if nv is not None and nv.shape[0] > 0:
                                q_t = orthogonal_purge(
                                    q_t, nv, nw,
                                    strength=cfg.ortho_strength,
                                    use_gram_schmidt=cfg.use_gram_schmidt,
                                )

                        scores = (q_t.unsqueeze(0) @ self.corpus_vectors.T).squeeze(0)

                        if cfg.mode != "no_mask":
                            nv, nw = board.get_negative_vectors()
                            if nv is not None and nv.shape[0] > 0:
                                scores = apply_enhanced_penalty(
                                    scores.unsqueeze(0),
                                    self.corpus_vectors,
                                    nv, nw,
                                    tau=cfg.masking_threshold,
                                    max_penalty=cfg.masking_penalty_weight,
                                    soft=True, temperature=0.1,
                                ).squeeze(0)

                        top_k_scores = torch.topk(scores, cfg.rollback_top_k).values
                        current_top_k_mean = top_k_scores.mean().item()

                prev_top_k_mean = current_top_k_mean
            else:
                if t == 0:
                    rk = cfg.rollback_top_k if hasattr(cfg, 'rollback_top_k') else 50
                    top_k_scores = torch.topk(scores, rk).values
                    prev_top_k_mean = top_k_scores.mean().item()

            ranked = torch.argsort(scores, descending=True)

            # ── Step 5: Re-ranking bằng cross-encoder ──
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

        if rollback_count > 0:
            logger.info(f"[ROLLBACK SUMMARY] Session {session.session_id}: "
                        f"{rollback_count} rollback(s).")
        if feedback_count > 0:
            logger.info(f"[VISUAL FB SUMMARY] Session {session.session_id}: "
                        f"{feedback_count} concept(s) adjusted.")

        return SessionOutput(session_id=session.session_id, turns=turn_outputs)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _encoder_fn(self):
        if self.text_encoder is None:
            return None
        return lambda texts: self.text_encoder.encode(texts)

    def _resolve_query_vector(self, turn) -> torch.Tensor:
        if turn.query_vector is not None:
            return turn.query_vector.to(self.corpus_vectors.device)
        if self.text_encoder is None:
            raise ValueError(
                "DialogTurn không có query_vector và Pipeline không được cấp text_encoder."
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
