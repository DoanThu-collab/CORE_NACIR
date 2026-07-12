"""
NACIR++ Plug-and-Play — Cắm vào ChatIR baseline
===============================================
Thay vì BLIP/PlugIR, dùng ChatIR backbone nhưng giữ nguyên NACIR++ core.

🔥 BẢN DEBUG: chỉ chạy DEBUG_LIMIT sample đầu tiên để test nhanh pipeline
trước khi chạy full 2064 dialog. Set DEBUG_LIMIT = None để chạy full.
"""

import logging
import os
import sys
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm

# ============================================================
# PATH NACIR++
# ============================================================
sys.path.insert(0, "/AIClub_NAS/core_baotg/thudnm/NACIR")

from nacir_plusplus import NACIRPlusPlusPipeline
from nacir_plusplus.config import OPTIMAL_CONFIG
from nacir_plusplus.metrics import compute_metrics, format_metrics_report
from nacir_plusplus.schema import DialogTurn, RetrievalSession

from nacir_plusplus.adapters.belief_sources import PrecomputedBeliefSource
from nacir_plusplus.adapters.chatir_backbone import build_chatir_backbone
from nacir_plusplus.adapters.visdial_corpus import Corpus, Queries, load_corpus_paths


# ============================================================
# CONFIG
# ============================================================
DATA_DIR = "/AIClub_NAS/core_baotg/thuyntn/Datasets/PlugIR"
QUERIES_PATH = "/AIClub_NAS/core_baotg/thudnm/ChatIR/dialogues/VisDial_v1.0_queries_val.json"
CHATIR_QUERIES_PATH = QUERIES_PATH
CACHE_CORPUS_PATH = "/AIClub_NAS/core_baotg/thudnm/ChatIR/temp/corpus_blip_small.pth"
CORPUS_PATH = "/AIClub_NAS/core_baotg/thudnm/ChatIR/ChatIR_Protocol/Search_Space_val_50k.json"
BELIEFS_PATH = "/AIClub_NAS/core_baotg/thuyntn/NACIR/data/semantic_beliefs.json"
CHATIR_FINETUNED_PATH = "/tmp/chatir_weights.ckpt"  # bản local đã rsync về

OUTPUT_DIR = "logs/chatir_debug"

BATCH_SIZE = 64
RERANK_K = 50
NUM_ROUNDS = 11
TOP_K = 10

# ============================================================
# 🔥 DEBUG LIMIT — đổi thành None để chạy full dataset
# ============================================================
DEBUG_LIMIT = None

os.environ["TOKENIZERS_PARALLELISM"] = "true"


# ============================================================
def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "run_chatir_debug.log")),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--beliefs_path", type=str, default=BELIEFS_PATH)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    logger = setup_logging(args.output_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("=" * 60)
    logger.info("NACIR++ + ChatIR RUN (DEBUG MODE)" if DEBUG_LIMIT else "NACIR++ + ChatIR RUN (FULL)")
    logger.info(f"Device: {device}")
    logger.info(f"DEBUG_LIMIT: {DEBUG_LIMIT}")
    logger.info("=" * 60)

    # --------------------------------------------------------
    # 1. Backbone (ChatIR)
    # --------------------------------------------------------
    text_encoder, image_encoder, itm_scorer = build_chatir_backbone(
        device=device,
        chatir_repo_dir="/AIClub_NAS/core_baotg/thudnm/ChatIR",
        finetuned_ckpt_path=CHATIR_FINETUNED_PATH,
    )

    # --------------------------------------------------------
    # 2. Corpus
    # --------------------------------------------------------
    corpus_dataset = Corpus(DATA_DIR, CORPUS_PATH, image_encoder.preprocess)
    corpus_ids, corpus_vectors = torch.load(CACHE_CORPUS_PATH, map_location=device)
    corpus_paths = load_corpus_paths(DATA_DIR, CORPUS_PATH)

    def corpus_ref_lookup(idx: int):
        return corpus_paths[idx]

    # --------------------------------------------------------
    # 3. Dataset
    # --------------------------------------------------------
    dataset = Queries(CHATIR_QUERIES_PATH, DATA_DIR, split=True)

    with open(QUERIES_PATH) as f:
        original_queries = json.load(f)

    # ------------------------------------------------------------------
    # 🔥 Sanity check TRƯỚC khi cắt: xác nhận dataset.queries và
    # original_queries cùng thứ tự (cùng trỏ tới cùng 1 ảnh/dialog)
    # ------------------------------------------------------------------
    logger.info(f"dataset.queries[0] keys: {list(dataset.queries[0].keys())}")
    logger.info(f"dataset.queries[0] img : {dataset.queries[0].get('img')}")
    logger.info(f"original_queries[0] img: {original_queries[0].get('img')}")

    if dataset.queries[0].get("img") != original_queries[0].get("img"):
        logger.warning(
            "⚠️  CẢNH BÁO: dataset.queries và original_queries KHÔNG cùng "
            "thứ tự! Việc cắt DEBUG_LIMIT theo vị trí có thể tạo cặp "
            "(target_index, dialog) sai lệch. Kiểm tra lại trước khi tin "
            "kết quả bên dưới."
        )

    # ------------------------------------------------------------------
    # 🔥 Áp dụng DEBUG_LIMIT (nếu có) — cắt CẢ HAI list cùng lúc
    # ------------------------------------------------------------------
    if DEBUG_LIMIT is not None:
        dataset.queries = dataset.queries[:DEBUG_LIMIT]
        original_queries = original_queries[:DEBUG_LIMIT]
        logger.info(f"🔥 DEBUG MODE: chỉ chạy {len(original_queries)} sample")

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    belief_source = PrecomputedBeliefSource.from_json(
        args.beliefs_path, turn_offset=-1
    )

    # --------------------------------------------------------
    # 4. Precompute text embeddings
    # --------------------------------------------------------
    all_text_embs = []
    for t in range(NUM_ROUNDS):
        dataset.dialog_length = t
        embs = []
        for batch in tqdm(dataloader, desc=f"Round {t}"):
            embs.append(text_encoder.encode(batch["text"]))
        all_text_embs.append(torch.cat(embs))

    # --------------------------------------------------------
    # 5. NACIR++ pipeline
    # --------------------------------------------------------
    pipeline = NACIRPlusPlusPipeline(
        config=OPTIMAL_CONFIG,
        corpus_vectors=corpus_vectors,
        text_encoder=text_encoder,
        belief_source=belief_source,
        image_scorer=itm_scorer,
        corpus_ref_lookup=corpus_ref_lookup,
        rerank_k=RERANK_K,
        top_k=TOP_K,
        device=device,
    )

    # --------------------------------------------------------
    # 6. Run sessions
    # --------------------------------------------------------
    ranks_per_round = [[] for _ in range(NUM_ROUNDS)]

    for i in tqdm(range(len(original_queries)), desc="Sessions"):
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
            session_id=i,
            turns=turns,
            target_index=target_index
        )

        result = pipeline.run_session(session)

        for out in result.turns:
            ranks_per_round[out.turn_index].append(out.target_rank)

    # --------------------------------------------------------
    # 7. Metrics
    # --------------------------------------------------------
    metrics = compute_metrics(ranks_per_round, k=TOP_K)
    logger.info("\n" + format_metrics_report(metrics, k=TOP_K))

    save_path = os.path.join(args.output_dir, "chatir_debug_ranks.npz")
    np.savez_compressed(
        save_path,
        ranks_per_round=np.array(ranks_per_round, dtype=object)
    )

    logger.info(f"Saved to {save_path}")


if __name__ == "__main__":
    main()