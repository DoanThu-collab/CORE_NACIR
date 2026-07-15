"""Example evaluation script for NACIR++ using the PlugIR backbone."""

import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

CORE_RETRIEVAL_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = CORE_RETRIEVAL_ROOT.parent

if str(CORE_RETRIEVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_RETRIEVAL_ROOT))

from nacir_plusplus.config import OPTIMAL_CONFIG
from nacir_plusplus.adapters.belief_sources import PrecomputedBeliefSource
from nacir_plusplus.adapters.plugir_backbone import build_backbone
from nacir_plusplus.adapters.visdial_corpus import Corpus, Queries
from nacir_plusplus.core.query_update import NACIRPlusPlusBatchUpdater
from nacir_plusplus.core.reranker import rerank_topk_with_lookup
from nacir_plusplus.metrics import compute_metrics, format_metrics_report

DATA_DIR = Path(os.environ.get("NACIR_DATA_DIR", "/path/to/dataset"))
QUERIES_PATH = Path(os.environ.get("NACIR_QUERIES_PATH", "/path/to/queries.json"))
PLUGIR_QUERIES_PATH = Path(os.environ.get("NACIR_PLUGIR_QUERIES_PATH", "/path/to/plugir_queries.json"))
CACHE_CORPUS_PATH = Path(os.environ.get("NACIR_CACHE_CORPUS_PATH", "/path/to/corpus_cache.pth"))
CORPUS_PATH = Path(os.environ.get("NACIR_CORPUS_PATH", "/path/to/corpus.json"))
BELIEFS_PATH = Path(os.environ.get("NACIR_BELIEFS_PATH", "/path/to/beliefs.json"))
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "plugir"

BATCH_SIZE = 64
ITM_BATCH_SIZE = 16
RERANK_K = 50
NUM_ROUNDS = 11
ITM_WEIGHT = 0.7
USE_DYNAMIC_SCHEDULE = True

os.environ["TOKENIZERS_PARALLELISM"] = "true"

def main():
    global OUTPUT_DIR, RERANK_K
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--beliefs_path", type=str, nargs='+', required=True, help="List of belief JSON files")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR), help="Output directory for logs and results")
    parser.add_argument("--rerank_k", type=int, default=RERANK_K, help="Number of top candidates to re-rank")
    args = parser.parse_args()
    
    OUTPUT_DIR = Path(args.output_dir)
    RERANK_K = args.rerank_k

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Starting NACIR++ Plug-and-Play with config: {OPTIMAL_CONFIG}")

    text_encoder, image_encoder, itm_scorer = build_backbone(device)

    corpus_dataset = Corpus(str(DATA_DIR), str(CORPUS_PATH), image_encoder.preprocess)
    corpus_ids, corpus_vectors = torch.load(CACHE_CORPUS_PATH, map_location=device)
    with open(CORPUS_PATH) as f:
        corpus_paths = [os.path.join(DATA_DIR, p) for p in json.load(f)]

    # Align corpus vectors if cache order doesn't match
    corpus_ids_list = corpus_ids.tolist() if torch.is_tensor(corpus_ids) else list(corpus_ids)
    if corpus_ids_list != list(range(len(corpus_paths))):
        reordered = torch.empty_like(corpus_vectors)
        reordered[torch.as_tensor(corpus_ids_list, device=device)] = corpus_vectors
        corpus_vectors = reordered

    def corpus_ref_lookup(idx: int) -> str:
        return corpus_paths[idx]

    dataset = Queries(str(PLUGIR_QUERIES_PATH), str(DATA_DIR), split=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    with open(QUERIES_PATH) as f:
        original_queries = json.load(f)
    num_queries = len(original_queries)

    print("Extracting text embeddings...")
    all_text_embs = []
    for dl in range(NUM_ROUNDS):
        dataset.dialog_length = dl
        round_embs = []
        for batch in tqdm(dataloader, desc=f"Text Round {dl}"):
            round_embs.append(text_encoder.encode(batch["text"]))
        all_text_embs.append(torch.cat(round_embs))

    for belief_file in args.beliefs_path:
        model_name = os.path.splitext(os.path.basename(belief_file))[0]
        output_dir = OUTPUT_DIR / model_name
        output_dir.mkdir(parents=True, exist_ok=True)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(message)s",
            handlers=[
                logging.FileHandler(str(output_dir / "run_log.txt")),
                logging.StreamHandler(),
            ],
        )
        logger = logging.getLogger(__name__)
        logger.info("Running model: %s (file: %s)", model_name, belief_file)

        belief_source = PrecomputedBeliefSource.from_json(belief_file, turn_offset=-1)

        checkpoint_path = output_dir / "checkpoint.pkl"
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, "rb") as f:
                checkpoint = pickle.load(f)
                ranks_per_round = checkpoint["ranks_per_round"]
                top_k_per_round = checkpoint.get("top_k_per_round", [[] for _ in range(NUM_ROUNDS)])
            start_batch_idx = len(ranks_per_round[0])
            logger.info("Resuming from checkpoint: completed %s/%s queries.", start_batch_idx, num_queries)
        else:
            ranks_per_round = [[] for _ in range(NUM_ROUNDS)]
            top_k_per_round = [[] for _ in range(NUM_ROUNDS)]
            start_batch_idx = 0

        total_overrides = 0
        total_batches = num_queries // BATCH_SIZE + (1 if num_queries % BATCH_SIZE != 0 else 0)
        initial_batch = start_batch_idx // BATCH_SIZE

        for i in tqdm(range(start_batch_idx, num_queries, BATCH_SIZE), desc=f"NACIR++ Batches ({model_name})", initial=initial_batch, total=total_batches):
            end_idx = min(i + BATCH_SIZE, num_queries)
            batch_size_actual = end_idx - i
            target_indices = [corpus_dataset.path_to_index(os.path.join(DATA_DIR, original_queries[j]["img"])) for j in range(i, end_idx)]

            updater = NACIRPlusPlusBatchUpdater(config=OPTIMAL_CONFIG, batch_size=batch_size_actual, encoder=text_encoder.encode, device=device)

            for t in tqdm(range(NUM_ROUNDS), desc=f"Batch {i//BATCH_SIZE + 1} Turns", leave=False):
                q_t = all_text_embs[t][i:end_idx].to(device)
                
                progress_t = min(t / 10.0, 1.0)
                dyn_itm = 0.2 + (ITM_WEIGHT - 0.2) * progress_t

                if t > 0:
                    if USE_DYNAMIC_SCHEDULE:
                        scale = min((t - 1) / 9.0, 1.0)
                        updater.config.ortho_strength = 0.05 + (0.25 - 0.05) * scale
                        updater.config.masking_penalty_weight = 0.05 + (0.20 - 0.05) * scale
                        for b_idx in range(batch_size_actual):
                            updater.boards[b_idx].config.alpha = 0.20 + (0.60 - 0.20) * scale
                            updater.boards[b_idx].config.beta = updater.boards[b_idx].config.alpha * 0.5

                    beliefs_batch = [belief_source.get_beliefs(j, t, "", "") for j in range(i, end_idx)]
                    mapped_beliefs = []
                    for b_obj in beliefs_batch:
                        mapped_beliefs.append({
                            "positive_beliefs": [{"attribute": pb.attribute, "confidence": pb.confidence} for pb in b_obj.positive_beliefs],
                            "negative_beliefs": [{"attribute": nb.attribute, "confidence": nb.confidence} for nb in b_obj.negative_beliefs]
                        })
                    q_t = updater.update_query(q_text_batch=q_t, beliefs_batch=mapped_beliefs, turn=t)

                scores = q_t @ corpus_vectors.T
                if t > 0: scores = updater.apply_masking(scores, corpus_vectors)

                ranked = torch.argsort(scores, descending=True)

                for b in range(batch_size_actual):
                    top_rerank = ranked[b, :RERANK_K]
                    q_text_str = dataset.queries[i+b]["dialog"][t]
                    
                    reranked_indices, _ = rerank_topk_with_lookup(
                        query_text=q_text_str,
                        top_k_corpus_indices=top_rerank.cpu(),
                        corpus_ref_lookup=corpus_ref_lookup,
                        image_scorer=itm_scorer,
                        cosine_scores=scores[b, top_rerank].cpu(),
                        itm_weight=dyn_itm,
                        device=device
                    )
                    
                    full_reranked = torch.cat([reranked_indices.to(device), ranked[b, RERANK_K:]])
                    ranks_per_round[t].append((full_reranked == target_indices[b]).nonzero(as_tuple=True)[0].item())
                    top_k_per_round[t].append(full_reranked[:10].cpu().tolist())

            total_overrides += updater.get_batch_stats()["total_overrides"]

            with open(checkpoint_path, "wb") as f:
                pickle.dump({'ranks_per_round': ranks_per_round, 'top_k_per_round': top_k_per_round}, f)
            logger.info("Checkpoint saved to %s (completed batch %s/%s)", checkpoint_path, i // BATCH_SIZE + 1, total_batches)

        metrics = compute_metrics(ranks_per_round, k=10)
        logger.info("\n" + format_metrics_report(metrics, k=10))

        save_path = output_dir / "nacir_plus_plugplay_ranks.npz"
        np.savez_compressed(save_path, ranks_per_round=np.array(ranks_per_round, dtype=object))
        logger.info("Results saved to %s", save_path)

if __name__ == "__main__":
    main()
