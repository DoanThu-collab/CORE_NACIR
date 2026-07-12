"""
NACIR++ Plug-and-Play — Ví dụ 1: Tái lập kết quả PlugIR/VisDial gốc
========================================================================
Đây là bản thay thế cho `main.py` gốc, nhưng giờ toàn bộ phần "đặc thù
PlugIR/VisDial/BLIP" chỉ còn là LỚP CẮM (adapter) — core logic NACIR++
(Concept Memory / Orthogonal Projection / Attention Masking / ITM rerank)
không hề bị đụng tới, chỉ được gọi qua `NACIRPlusPlusPipeline`.

Chạy:
    CUDA_VISIBLE_DEVICES=0 python examples/run_plugir_visdial.py
"""

import logging
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nacir_plusplus import NACIRPlusPlusPipeline
from nacir_plusplus.config import OPTIMAL_CONFIG
from nacir_plusplus.metrics import compute_metrics, format_metrics_report
from nacir_plusplus.schema import DialogTurn, RetrievalSession

from nacir_plusplus.adapters.belief_sources import PrecomputedBeliefSource
from nacir_plusplus.adapters.blip_backbone import build_blip_backbone
from nacir_plusplus.adapters.visdial_corpus import Corpus, Queries, load_corpus_paths

import json

# ============================================================
# Cấu hình đường dẫn — Y HỆT main.py gốc (đảm bảo tái lập được kết quả)
# ============================================================

DATA_DIR = "/AIClub_NAS/core_baotg/thuyntn/Datasets/PlugIR/"
QUERIES_PATH = "/AIClub_NAS/core_baotg/thuyntn/PlugIR_Workspace/PlugIR/dialogues/VisDial_v1.0_queries_val.json"
PLUGIR_QUERIES_PATH = "/AIClub_NAS/core_baotg/thuyntn/PlugIR/dialogues/ours_final_q_n_5_recall_hitting_10_thres_low_500_recon_true_referring_true_filtering_true_select_true_reconed.json"
CACHE_CORPUS_PATH = "/AIClub_NAS/core_baotg/thuyntn/PlugIR_Workspace/ChatIR/temp/corpus_blip_large.pth"
CORPUS_PATH = "/AIClub_NAS/core_baotg/thuyntn/PlugIR_Workspace/PlugIR/Protocol/Search_Space_val_50k.json"
BELIEFS_PATH = "/AIClub_NAS/core_baotg/thuyntn/NACIR/data/semantic_beliefs.json"
OUTPUT_DIR = "logs"

BATCH_SIZE = 64
ITM_BATCH_SIZE = 16
RERANK_K = 50
NUM_ROUNDS = 11
TOP_K = 10

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def setup_logging(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "run_nacir_plus_plugplay.log")),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def main():
    logger = setup_logging(OUTPUT_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("=" * 60)
    logger.info("NACIR++ Plug-and-Play — reproducing PlugIR/VisDial run")
    logger.info(f"Config: {OPTIMAL_CONFIG}")
    logger.info("=" * 60)

    # ---- 1. Backbone (đây là "lớp cắm" — có thể thay bằng CLIP/SigLIP khác) ----
    text_encoder, image_encoder, itm_scorer = build_blip_backbone(device)

    corpus_dataset = Corpus(DATA_DIR, CORPUS_PATH, image_encoder.preprocess)
    corpus_ids, corpus_vectors = torch.load(CACHE_CORPUS_PATH, map_location=device)
    corpus_paths = load_corpus_paths(DATA_DIR, CORPUS_PATH)

    def corpus_ref_lookup(idx: int) -> str:
        return corpus_paths[idx]

    # ---- 2. Dữ liệu hội thoại (đặc thù PlugIR) ----
    dataset = Queries(PLUGIR_QUERIES_PATH, DATA_DIR, split=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    with open(QUERIES_PATH) as f:
        original_queries = json.load(f)
    num_queries = len(original_queries)

    belief_source = PrecomputedBeliefSource.from_json(BELIEFS_PATH, turn_offset=-1)

    # ---- 3. Precompute text embeddings mỗi round (y hệt bản gốc) ----
    all_text_embs = []
    for dl in range(NUM_ROUNDS):
        dataset.dialog_length = dl
        round_embs = []
        for batch in tqdm(dataloader, desc=f"Text Round {dl}"):
            round_embs.append(text_encoder.encode(batch["text"]))
        all_text_embs.append(torch.cat(round_embs))

    # ---- 4. Xây dựng NACIR++ Pipeline (điểm cắm chuẩn hoá) ----
    pipeline = NACIRPlusPlusPipeline(
        config=OPTIMAL_CONFIG,
        corpus_vectors=corpus_vectors,
        text_encoder=None,  # query_vector đã cấp sẵn cho từng turn -> khỏi cần encoder
        belief_source=belief_source,
        image_scorer=itm_scorer,
        corpus_ref_lookup=corpus_ref_lookup,
        rerank_k=RERANK_K,
        top_k=TOP_K,
        device=device,
    )

    # ---- 5. Build RetrievalSession chuẩn hoá cho từng dialog ----
    ranks_per_round = [[] for _ in range(NUM_ROUNDS)]

    for i in tqdm(range(num_queries), desc="NACIR++ Plug-and-Play sessions"):
        target_index = corpus_dataset.path_to_index(os.path.join(DATA_DIR, original_queries[i]["img"]))

        turns = [
            DialogTurn(
                turn_index=t,
                query_text=dataset.queries[i]["dialog"][t],
                query_vector=all_text_embs[t][i],
            )
            for t in range(NUM_ROUNDS)
        ]
        session = RetrievalSession(session_id=i, turns=turns, target_index=target_index)

        result = pipeline.run_session(session)
        for turn_out in result.turns:
            ranks_per_round[turn_out.turn_index].append(turn_out.target_rank)

    # ---- 6. Metrics (giữ nguyên công thức Hits@K / Recall@K / BRI gốc) ----
    metrics = compute_metrics(ranks_per_round, k=TOP_K)
    logger.info("\n" + format_metrics_report(metrics, k=TOP_K))

    save_path = os.path.join(OUTPUT_DIR, "nacir_plus_plugplay_ranks.npz")
    np.savez_compressed(save_path, ranks_per_round=np.array(ranks_per_round, dtype=object))
    logger.info(f"Results saved to {save_path}")


if __name__ == "__main__":
    main()
