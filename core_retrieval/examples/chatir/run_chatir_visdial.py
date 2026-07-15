"""Example evaluation script for NACIR using the ChatIR backbone."""

import sys
import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import copy
import logging
from contextlib import nullcontext

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CORE_RETRIEVAL_ROOT = PROJECT_ROOT / "core_retrieval"

if str(CORE_RETRIEVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_RETRIEVAL_ROOT))

torch.set_num_threads(4)
torch.set_num_interop_threads(2)
torch.backends.cudnn.benchmark = True

from nacir_plusplus import NACIRPlusPlusPipeline
from nacir_plusplus.config import OPTIMAL_CONFIG, DynamicScheduleConfig, default_dynamic_schedule
from nacir_plusplus.schema import DialogTurn, RetrievalSession
from nacir_plusplus.metrics import compute_metrics, format_metrics_report
from nacir_plusplus.adapters.belief_sources import PrecomputedBeliefSource
from nacir_plusplus.adapters.chatir_backbone import build_backbone
from nacir_plusplus.adapters.visdial_corpus import Corpus, Queries, load_corpus_paths

DATA_DIR = Path(os.environ.get("NACIR_DATA_DIR", "/path/to/dataset"))
CHATIR_ROOT = Path(os.environ.get("CHATIR_ROOT", "/path/to/chatir"))
BELIEFS_PATH = Path(os.environ.get("NACIR_BELIEFS_PATH", "/path/to/beliefs.json"))
CKPT_PATH = Path(os.environ.get("CHATIR_CKPT_PATH", "/path/to/model.ckpt"))

QUERIES_PATH = CHATIR_ROOT / "dialogues" / "VisDial_v1.0_queries_val.json"
CACHE_CORPUS_PATH = CHATIR_ROOT / "temp" / "corpus_blip_small.pth"
CORPUS_PATH = CHATIR_ROOT / "ChatIR_Protocol" / "Search_Space_val_50k.json"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "chatir"
RESULTS_PATH = OUTPUT_DIR / "nacir_results.json"
LOG_PATH = OUTPUT_DIR / "nacir_run.log"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(str(LOG_PATH)),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

NUM_ROUNDS = 11
BATCH_SIZE = 64
NUM_WORKERS = 4

SCHEDULE_ALPHA_START = 0.02
SCHEDULE_ALPHA_END = 0.12
SCHEDULE_ORTHO_START = 0.00
SCHEDULE_ORTHO_END = 0.02
SCHEDULE_PENALTY_START = 0.00
SCHEDULE_PENALTY_END = 0.01
SCHEDULE_WARMUP_TURNS = 12.0
SCHEDULE_ITM_START = 0.02
SCHEDULE_ITM_END = 0.20
SCHEDULE_ITM_WARMUP_TURNS = 12.0

CHATIR_BASELINE = [63.42, 67.54, 70.54, 72.97, 74.85, 
                   76.94, 78.34, 79.6, 80.57, 81.01, 81.93]


def save_results(results):
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def validate_against_baseline(hits_per_turn):
    """Validate retrieval performance against the reference baseline."""
    logger.info("\n" + "=" * 60)
    logger.info("Validation checks:")
    logger.info("=" * 60)
    
    errors = []
    warnings = []
    
    if abs(hits_per_turn[0] - CHATIR_BASELINE[0]) > 1.0:
        errors.append(f"Turn 0 mismatch: {hits_per_turn[0]:.2f}% vs {CHATIR_BASELINE[0]:.2f}%")
    else:
        logger.info(f"Turn 0 matches baseline: {hits_per_turn[0]:.2f}%")
    
    if hits_per_turn[10] > CHATIR_BASELINE[10]:
        logger.info(f"Turn 10 exceeds baseline: {hits_per_turn[10]:.2f}%")
        logger.info(f"Improvement: +{hits_per_turn[10] - CHATIR_BASELINE[10]:.2f}%")
    else:
        errors.append(f"Turn 10 did not exceed baseline: {hits_per_turn[10]:.2f}% <= {CHATIR_BASELINE[10]:.2f}%")
    
    for t in range(1, len(hits_per_turn)):
        drop = hits_per_turn[t-1] - hits_per_turn[t]
        if drop > 3:
            warnings.append(f"Large drop at turn {t}: {hits_per_turn[t-1]:.2f}% -> {hits_per_turn[t]:.2f}% (-{drop:.2f}%)")
    
    early = sum(hits_per_turn[1:6]) / 5  # Turn 1-5 average
    late = sum(hits_per_turn[6:11]) / 5  # Turn 6-10 average
    trend = late - early
    
    logger.info("\nTrend analysis:")
    logger.info(f"   Turn 1-5 avg: {early:.2f}%")
    logger.info(f"   Turn 6-10 avg: {late:.2f}%")
    logger.info(f"   Trend: {trend:+.2f}% {'up' if trend > 0 else 'down'}")
    
    if warnings:
        logger.warning("\nWarnings:")
        for w in warnings:
            logger.warning(f"   {w}")
    
    if errors:
        logger.error("\nValidation errors:")
        for e in errors:
            logger.error(f"   {e}")
        logger.error("\nValidation failed.")
        return False
    
    logger.info("\nEvaluation completed successfully.")
    return True


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = "cuda" if "cuda" in device else "cpu"

    logger.info("=" * 60)
    logger.info("Starting evaluation...")
    logger.info("Two-phase hybrid schedule with safeguards.")
    logger.info("Reference baseline: Turn 10 = 81.93%")
    logger.info("=" * 60)

    text_encoder, image_encoder, itm_scorer = build_backbone(
        device=device, chatir_repo_dir=str(CHATIR_ROOT), finetuned_ckpt_path=str(CKPT_PATH)
    )

    corpus_dataset = Corpus(str(DATA_DIR), str(CORPUS_PATH), image_encoder.preprocess)
    corpus_ids, corpus_vectors = torch.load(CACHE_CORPUS_PATH, map_location=device)
    corpus_paths = load_corpus_paths(str(DATA_DIR), str(CORPUS_PATH))

    def corpus_ref_lookup(idx):
        return corpus_paths[idx]

    dataset = Queries(str(QUERIES_PATH), str(DATA_DIR), split=False)
    with open(QUERIES_PATH) as f:
        original_queries = json.load(f)

    total_queries = len(original_queries)
    logger.info(f"Total queries: {total_queries}")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    belief_source = PrecomputedBeliefSource.from_json(str(BELIEFS_PATH), turn_offset=-1)

    logger.info("Precomputing text embeddings...")
    all_text_embs = []
    with torch.inference_mode():
        for t in range(NUM_ROUNDS):
            dataset.dialog_length = t
            embs = []
            for batch in tqdm(dataloader, desc=f"Encoding round {t}"):
                embs.append(text_encoder.encode(batch["text"]))
            all_text_embs.append(torch.cat(embs))

    def make_query(dialog, turn):
        # Must match Queries(..., split=False), which made query_vector.
        return ", ".join(dialog[: turn + 1])

    sessions = []
    for i in range(total_queries):
        target_index = corpus_dataset.path_to_index(os.path.join(DATA_DIR, original_queries[i]["img"]))
        turns = [
            DialogTurn(
                turn_index=t,
                query_text=make_query(dataset.queries[i]["dialog"], t),
                query_vector=all_text_embs[t][i],
            )
            for t in range(NUM_ROUNDS)
        ]
        sessions.append(RetrievalSession(session_id=i, turns=turns, target_index=target_index))

    # ============================================================
    cfg = copy.deepcopy(OPTIMAL_CONFIG)
    cfg.mode = "full"
    
    soft_sched = DynamicScheduleConfig(
        alpha_start=SCHEDULE_ALPHA_START, alpha_end=SCHEDULE_ALPHA_END,
        ortho_start=SCHEDULE_ORTHO_START, ortho_end=SCHEDULE_ORTHO_END,
        penalty_start=SCHEDULE_PENALTY_START, penalty_end=SCHEDULE_PENALTY_END,
        warmup_turns=SCHEDULE_WARMUP_TURNS,
        itm_start=SCHEDULE_ITM_START, itm_end=SCHEDULE_ITM_END,
        itm_warmup_turns=SCHEDULE_ITM_WARMUP_TURNS,
    )

    def piecewise_schedule_fn(t):
        base_sched = default_dynamic_schedule(t, soft_sched)
        
        if t == 0:
            return {"itm_weight": 0.0}
        
        elif t <= 5:
            sched = dict(base_sched)
            
            sched["itm_weight"] *= 0.5
            sched["memory_alpha"] *= 0.6
            sched["memory_beta"] *= 0.6
            
            return sched
        
        else:
            sched = dict(base_sched)
            
            sched["memory_alpha"] = min(sched["memory_alpha"], 0.12)
            sched["memory_beta"] = min(sched["memory_beta"], 0.06)
            sched["itm_weight"] = min(sched["itm_weight"], 0.20)
            
            if t >= 9:
                sched["masking_penalty_weight"] = 0.0
                sched["ortho_strength"] = 0.0
            
            return sched

    pipeline = NACIRPlusPlusPipeline(
        config=cfg,
        corpus_vectors=corpus_vectors,
        text_encoder=text_encoder,
        belief_source=belief_source,
        image_scorer=itm_scorer,
        corpus_ref_lookup=corpus_ref_lookup,
        rerank_k=50,
        top_k=10,
        device=device,
        schedule_fn=piecewise_schedule_fn,
    )

    ranks_per_round = [[] for _ in range(NUM_ROUNDS)]
    start_idx = 0
    
    if os.path.exists(RESULTS_PATH):
        try:
            with open(RESULTS_PATH, "r") as f:
                old_data = json.load(f)
            if "nacir_chatir_results" in old_data:
                old_ranks = old_data["nacir_chatir_results"].get("raw_ranks_per_round", [])
                if old_ranks and len(old_ranks[0]) > 0:
                    ranks_per_round = old_ranks
                    start_idx = len(old_ranks[0])
                    logger.info(f"Resuming from query {start_idx} / {total_queries}...")
        except Exception as e:
            logger.warning(f"Could not read previous results: {e}")

    with torch.inference_mode(), torch.autocast(device_type=device_type, dtype=torch.float16):
        num_batches = (total_queries - start_idx + BATCH_SIZE - 1) // BATCH_SIZE
        
        for batch_idx, i in enumerate(tqdm(range(start_idx, total_queries, BATCH_SIZE), 
                                            desc="Progress",
                                            total=num_batches,
                                            unit="batch")):
            batch_results = pipeline.run_batch(sessions[i:i + BATCH_SIZE])
            
            for result in batch_results:
                for out in result.turns:
                    ranks_per_round[out.turn_index].append(out.target_rank)
                    
            hits10_per_turn = []
            for t in range(NUM_ROUNDS):
                r = ranks_per_round[t]
                valid = [x for x in r if x is not None]
                hits10 = (sum(1 for x in valid if x < 10) / len(valid) * 100) if valid else 0.0
                hits10_per_turn.append(hits10)
            
            queries_done = min(i + BATCH_SIZE, total_queries)
            progress_pct = (queries_done / total_queries) * 100
            logger.info(
                f"Processed {queries_done}/{total_queries} queries ({progress_pct:.1f}%) | "
                f"Current Hit@10: {hits10_per_turn[10]:.2f}"
            )
                
            save_results({
                "nacir_chatir_results": {
                    "hits10_per_turn": hits10_per_turn,
                    "raw_ranks_per_round": ranks_per_round,
                }
            })

    logger.info("\n" + "=" * 60)
    logger.info("Evaluation completed.")
    logger.info("=" * 60)
    
    formatted = ", ".join(f"{v:.1f}" for v in hits10_per_turn)
    logger.info(f"Hit@10 per turn: {formatted}")
    logger.info(f"Results saved to: {RESULTS_PATH}")
    
    success = validate_against_baseline(hits10_per_turn)
    
    if success:
        logger.info("\nEvaluation passed.")
    else:
        logger.error("\nValidation failed.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Unhandled exception:")
        raise