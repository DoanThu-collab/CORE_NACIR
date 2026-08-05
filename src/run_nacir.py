"""
NACIR++ Bench — Runner V2 (tích hợp DCG + Visual Feedback)
===========================================================
File chạy chính, thêm các cờ CLI cho:
  - Dynamic Concept Graph (DCG)
  - Visual-Grounded Belief Refinement
"""
import logging
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import sys
import json
import argparse
import numpy as np
import torch

torch.set_num_threads(4)

from tqdm import tqdm

from nacir import NACIRPlusPlusPipeline
from nacir.config import OPTIMAL_CONFIG, NACIRPlusPlusConfig
from nacir.metrics import compute_metrics, format_metrics_report
from nacir.schema import DialogTurn, RetrievalSession

from nacir.adapters.belief_sources import PrecomputedBeliefSource
from nacir.adapters.plugir_backbone import build_plugir_backbone
from nacir.adapters.visdial_corpus import Corpus, Queries, load_corpus_paths

# Novelty Adapters (plug-in bên ngoài)
from adapters.counterfactual import CounterfactualBeliefSource
from adapters.learned_scheduler import LearnedScheduler

# ============================================================
# DEFAULT PATHS
# ============================================================
DATA_DIR = "/mlcv1/WorkingSpace/Personal/core_baotg/thuy/Dataset/PlugIR"
QUERIES_PATH = "/mlcv1/WorkingSpace/Personal/core_baotg/thuy/PlugIR/dialogues/ours_final_q_n_5_recall_hitting_10_thres_low_500_recon_true_referring_true_filtering_true_select_true_reconed.json"
CACHE_CORPUS_PATH = "/mlcv1/WorkingSpace/Personal/core_baotg/thuy/PlugIR_Workspace/ChatIR/temp/corpus_blip_large.pth"
CORPUS_PATH = "/mlcv1/WorkingSpace/Personal/core_baotg/thuy/PlugIR/Protocol/Search_Space_val_50k.json"
DEFAULT_BELIEFS_PATH = "/mlcv1/WorkingSpace/Personal/core_baotg/thuy/CORE_NACIR_sub/data/beliefs_llama3_1_8b.json"

BATCH_SIZE = 64
RERANK_K = 50
NUM_ROUNDS = 11
TOP_K = 10


def setup_logging(output_dir, name="run"):
    run_dir = os.path.join(output_dir, name)
    os.makedirs(run_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(os.path.join(run_dir, f"{name}.log"))
        sh = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s | %(message)s")
        fh.setFormatter(formatter)
        sh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger


def main():
    parser = argparse.ArgumentParser(description="NACIR++ Bench Runner V2")
    parser.add_argument("--beliefs_path", type=str, default=DEFAULT_BELIEFS_PATH)
    parser.add_argument("--output_dir", type=str, default="logs/plugir_eval")
    parser.add_argument("--run_name", type=str, default="nacir_base")
    parser.add_argument("--debug_limit", type=int, default=None)

    # ── Đề xuất cũ (Plug-in Adapters) ──
    parser.add_argument("--use_counterfactual", action="store_true",
                        help="Bật Counterfactual Purging (Adapter bên ngoài)")
    parser.add_argument("--cf_template", type=str, default="a photo showing {}",
                        help="Template cho Counterfactual Purging")
    parser.add_argument("--use_learned_scheduler", action="store_true",
                        help="Bật Learned Scheduler (Adapter bên ngoài)")
    parser.add_argument("--scheduler_type", type=str, default="mlp",
                        choices=["linear", "mlp", "gru", "attention"])
    parser.add_argument("--scheduler_model_path", type=str,
                        default="/mlcv1/WorkingSpace/Personal/core_baotg/thuy/NACIR_Bench/checkpoints/scheduler_mlp.pt")

    # ── Semantic-Aware & Roll-back (từ V1) ──
    parser.add_argument("--use_semantic_scheduler", action="store_true")
    parser.add_argument("--use_memory_rollback", action="store_true")
    parser.add_argument("--rollback_score_drop", type=float, default=0.05)
    parser.add_argument("--rollback_top_k", type=int, default=50)

    # ══════════════════════════════════════════════════════
    # ĐỀ XUẤT MỚI 1: Dynamic Concept Graph (DCG)
    # ══════════════════════════════════════════════════════
    parser.add_argument("--use_concept_graph", action="store_true",
                        help="Bật Dynamic Concept Graph: lan truyền tín hiệu neg "
                             "qua đồ thị ngữ nghĩa để chống Semantic Leakage")
    parser.add_argument("--graph_alpha", type=float, default=0.3,
                        help="Cường độ lan truyền (0=tắt, 1=chỉ neighbor)")
    parser.add_argument("--graph_threshold", type=float, default=0.50,
                        help="Cosine sim tối thiểu để tạo cạnh trong graph")
    parser.add_argument("--graph_hops", type=int, default=1,
                        help="Số bước lan truyền (1=neighbor trực tiếp)")
    # Turn-Evolving Graph
    parser.add_argument("--graph_evolving", action="store_true",
                        help="Bật Turn-Evolving Graph (temporal smoothing)")
    parser.add_argument("--graph_temporal_gamma", type=float, default=0.3,
                        help="Tỷ lệ pha trộn đồ thị mới vào cũ (0=nhớ lâu, 1=đổi nhanh)")
    # Bimodal Concept Node
    parser.add_argument("--graph_bimodal", action="store_true",
                        help="Bật Bimodal Concept Node (visual grounding)")
    parser.add_argument("--graph_bimodal_lambda", type=float, default=0.2,
                        help="Tỷ lệ pha trộn visual vào text (0=text only, 1=visual only)")
    parser.add_argument("--graph_bimodal_top_k", type=int, default=10,
                        help="Số ảnh top-K dùng cho visual grounding")

    # ══════════════════════════════════════════════════════
    # ĐỀ XUẤT MỚI 2: Visual-Grounded Belief Refinement
    # ══════════════════════════════════════════════════════
    parser.add_argument("--use_visual_feedback", action="store_true",
                        help="Bật Visual Feedback: dùng top-K ảnh để refine beliefs")
    parser.add_argument("--vf_top_k", type=int, default=50,
                        help="Số ảnh top-K để phân tích")
    parser.add_argument("--vf_suppress_threshold", type=float, default=0.15,
                        help="Positive relevance < ngưỡng này → suppress")
    parser.add_argument("--vf_boost_threshold", type=float, default=0.25,
                        help="Negative relevance > ngưỡng này → boost")
    parser.add_argument("--vf_suppress_factor", type=float, default=0.3,
                        help="Mức giảm confidence khi suppress positive")
    parser.add_argument("--vf_boost_factor", type=float, default=0.2,
                        help="Mức tăng confidence khi boost negative")

    # ── Ablation flags gốc ──
    parser.add_argument("--no_memory", action="store_true")
    parser.add_argument("--no_ortho", action="store_true")
    parser.add_argument("--no_mask", action="store_true")

    args = parser.parse_args()

    logger = setup_logging(args.output_dir, args.run_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("=" * 60)
    logger.info(f"RUNNING NACIR++ Bench V2: {args.run_name}")
    logger.info(f"Adapters: Counterfactual={args.use_counterfactual}, "
                f"LearnedScheduler={args.use_learned_scheduler}")
    logger.info(f"V1 Improvements: SemanticSched={args.use_semantic_scheduler}, "
                f"Rollback={args.use_memory_rollback}")
    logger.info(f"V2 NOVELTY: DCG={args.use_concept_graph}, "
                f"VisualFB={args.use_visual_feedback}")
    if args.use_concept_graph:
        logger.info(f"  DCG config: α={args.graph_alpha}, "
                    f"τ={args.graph_threshold}, hops={args.graph_hops}")
        if args.graph_evolving:
            logger.info(f"  DCG Turn-Evolving: γ={args.graph_temporal_gamma}")
        if args.graph_bimodal:
            logger.info(f"  DCG Bimodal: λ={args.graph_bimodal_lambda}, "
                        f"top_k={args.graph_bimodal_top_k}")
    if args.use_visual_feedback:
        logger.info(f"  VF config: top_k={args.vf_top_k}, "
                    f"suppress_τ={args.vf_suppress_threshold}, "
                    f"boost_τ={args.vf_boost_threshold}")
    logger.info("=" * 60)

    # 1. Config
    mode = "full"
    if args.no_memory: mode = "no_memory"
    elif args.no_ortho: mode = "no_ortho"
    elif args.no_mask: mode = "no_mask"

    config = NACIRPlusPlusConfig(**vars(OPTIMAL_CONFIG))
    config.mode = mode

    # V1 flags
    config.use_semantic_scheduler = args.use_semantic_scheduler
    config.use_memory_rollback = args.use_memory_rollback
    config.rollback_score_drop = args.rollback_score_drop
    config.rollback_top_k = args.rollback_top_k

    # V2 flags — DCG
    config.use_concept_graph = args.use_concept_graph
    config.graph_propagation_alpha = args.graph_alpha
    config.graph_similarity_threshold = args.graph_threshold
    config.graph_num_hops = args.graph_hops
    config.graph_evolving = args.graph_evolving
    config.graph_temporal_gamma = args.graph_temporal_gamma
    config.graph_bimodal = args.graph_bimodal
    config.graph_bimodal_lambda = args.graph_bimodal_lambda
    config.graph_bimodal_top_k = args.graph_bimodal_top_k

    # V2 flags — Visual Feedback
    config.use_visual_feedback = args.use_visual_feedback
    config.vf_top_k = args.vf_top_k
    config.vf_suppress_threshold = args.vf_suppress_threshold
    config.vf_boost_threshold = args.vf_boost_threshold
    config.vf_suppress_factor = args.vf_suppress_factor
    config.vf_boost_factor = args.vf_boost_factor

    # 2. Backbone
    text_encoder, image_encoder, itm_scorer = build_plugir_backbone(device=device)

    # 3. Corpus
    corpus_dataset = Corpus(DATA_DIR, CORPUS_PATH, image_encoder.preprocess)
    corpus_ids, corpus_vectors = torch.load(CACHE_CORPUS_PATH, map_location=device)
    corpus_paths = load_corpus_paths(DATA_DIR, CORPUS_PATH)
    def corpus_ref_lookup(idx: int): return corpus_paths[idx]

    # 4. Dataset
    dataset = Queries(QUERIES_PATH, DATA_DIR, split=True)
    with open(QUERIES_PATH) as f:
        original_queries = json.load(f)

    if args.debug_limit:
        dataset.queries = dataset.queries[:args.debug_limit]
        original_queries = original_queries[:args.debug_limit]
        logger.info(f"DEBUG MODE: Running on {args.debug_limit} samples")

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    # 5. Belief Source
    base_belief_source = PrecomputedBeliefSource.from_json(
        args.beliefs_path, turn_offset=-1
    )
    if args.use_counterfactual:
        logger.info(f"Enabled Counterfactual Purging with template: '{args.cf_template}'")
        belief_source = CounterfactualBeliefSource(
            base_belief_source, template=args.cf_template, enabled=True
        )
    else:
        belief_source = base_belief_source

    # 6. Schedule Function
    if args.use_learned_scheduler:
        logger.info(f"Enabled Learned Scheduler ({args.scheduler_type.upper()}) "
                    f"from {args.scheduler_model_path}")
        scheduler = LearnedScheduler(
            model_type=args.scheduler_type,
            model_path=args.scheduler_model_path,
            enabled=True, device=device,
        )
        schedule_fn = scheduler.as_schedule_fn()
    else:
        schedule_fn = None

    # 7. Precompute queries
    all_text_embs = []
    for t in range(NUM_ROUNDS):
        dataset.dialog_length = t
        embs = []
        for batch in tqdm(dataloader, desc=f"Encoding Round {t}"):
            embs.append(text_encoder.encode(batch["text"]))
        all_text_embs.append(torch.cat(embs))

    # 8. Pipeline
    pipeline = NACIRPlusPlusPipeline(
        config=config,
        corpus_vectors=corpus_vectors,
        text_encoder=text_encoder,
        belief_source=belief_source,
        image_scorer=itm_scorer,
        corpus_ref_lookup=corpus_ref_lookup,
        rerank_k=RERANK_K,
        top_k=TOP_K,
        device=device,
        schedule_fn=schedule_fn,
    )

    # 9. Evaluation with Checkpointing
    run_dir = os.path.join(args.output_dir, args.run_name)
    save_path = os.path.join(run_dir, "ranks.npz")
    ckpt_path = os.path.join(run_dir, "checkpoint.npz")

    ranks_per_round = [[] for _ in range(NUM_ROUNDS)]
    start_idx = 0

    if os.path.exists(ckpt_path):
        logger.info(f"Resuming from checkpoint: {ckpt_path}")
        ckpt_data = np.load(ckpt_path, allow_pickle=True)["ranks_per_round"]
        ranks_per_round = [list(r) for r in ckpt_data]
        start_idx = len(ranks_per_round[0])
        logger.info(f"Resuming at session index {start_idx}")

    for i in tqdm(range(start_idx, len(original_queries)), desc="Evaluating Sessions"):
        target_index = corpus_dataset.path_to_index(
            os.path.join(DATA_DIR, original_queries[i]["img"])
        )
        turns = [
            DialogTurn(
                turn_index=t,
                query_text=dataset.queries[i]["dialog"][t],
                query_vector=all_text_embs[t][i],
            )
            for t in range(NUM_ROUNDS)
        ]
        session = RetrievalSession(
            session_id=i, turns=turns, target_index=target_index
        )

        result = pipeline.run_session(session)
        for out in result.turns:
            ranks_per_round[out.turn_index].append(out.target_rank)

        if (i + 1) % 100 == 0:
            temp_ckpt = ckpt_path.replace(".npz", "_tmp.npz")
            np.savez_compressed(
                temp_ckpt,
                ranks_per_round=np.array(ranks_per_round, dtype=object),
            )
            os.replace(temp_ckpt, ckpt_path)

    # 10. Metrics
    metrics = compute_metrics(ranks_per_round, k=TOP_K)
    logger.info("\n" + format_metrics_report(metrics, k=TOP_K))

    np.savez_compressed(
        save_path,
        ranks_per_round=np.array(ranks_per_round, dtype=object),
    )
    logger.info(f"Saved to {save_path}")

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    
    # Ép tiến trình Python thoát ngay lập tức (tránh kẹt Dataloader/Threads)
    os._exit(0)

if __name__ == "__main__":
    main()
