"""Example evaluation script for NACIR++ using the BLIP backbone."""

import copy
import json
import logging
import os
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

CORE_RETRIEVAL_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = CORE_RETRIEVAL_ROOT.parent

if str(CORE_RETRIEVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_RETRIEVAL_ROOT))

import torch
from tqdm import tqdm

torch.set_num_threads(8)
torch.set_num_interop_threads(2)
torch.backends.cudnn.benchmark = True

from nacir_plusplus import NACIRPlusPlusPipeline
from nacir_plusplus.config import OPTIMAL_CONFIG, DynamicScheduleConfig, default_dynamic_schedule
from nacir_plusplus.schema import DialogTurn, RetrievalSession
from nacir_plusplus.adapters.belief_sources import PrecomputedBeliefSource
from nacir_plusplus.adapters.blip_backbone import build_backbone
from nacir_plusplus.adapters.visdial_corpus import Corpus, Queries, load_corpus_paths

DATA_DIR = Path(os.environ.get("NACIR_DATA_DIR", "/path/to/dataset"))
CHATIR_DIR = Path(os.environ.get("CHATIR_ROOT", "/path/to/chatir"))
QUERIES_PATH = CHATIR_DIR / "dialogues" / "VisDial_v1.0_queries_val.json"
CACHE_CORPUS_PATH = CHATIR_DIR / "temp" / "corpus_blip_small.pth"
CORPUS_PATH = CHATIR_DIR / "ChatIR_Protocol" / "Search_Space_val_50k.json"
BELIEFS_PATH = Path(os.environ.get("NACIR_BELIEFS_PATH", "/path/to/beliefs.json"))
CKPT_PATH = Path(os.environ.get("CHATIR_CKPT_PATH", "/path/to/model.ckpt"))

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "blip"
RESULTS_PATH = OUTPUT_DIR / "blip_nacir_full_11_turns.json"
LOG_PATH = OUTPUT_DIR / "blip_nacir_full_11_turns.log"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[logging.FileHandler(str(LOG_PATH)), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

NUM_ROUNDS = 11
BATCH_SIZE = 128

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

def save_results(results):
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = "cuda" if "cuda" in device else "cpu"

    logger.info("=" * 70)
    logger.info("Starting evaluation...")
    logger.info("BLIP + NACIR++ full 11 turns.")
    logger.info("Running on device: %s", device)
    logger.info("=" * 70)

    text_encoder, image_encoder, itm_scorer = build_backbone(
        device=device, blip_repo_dir=str(CHATIR_DIR), finetuned_ckpt_path=str(CKPT_PATH)
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

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    belief_source = PrecomputedBeliefSource.from_json(str(BELIEFS_PATH), turn_offset=-1)

    logger.info("Precomputing text embeddings...")
    all_text_embs = []
    with torch.inference_mode():
        for t in range(NUM_ROUNDS):
            dataset.dialog_length = t
            embs = []
            for batch in tqdm(dataloader, desc=f"Encoding turn {t}", leave=False):
                embs.append(text_encoder.encode(batch["text"]))
            all_text_embs.append(torch.cat(embs))

    def make_query(dialog, turn):
        return ", ".join(dialog[: turn + 1])

    sessions = []
    for i in range(total_queries):
        target_index = corpus_dataset.path_to_index(os.path.join(str(DATA_DIR), original_queries[i]["img"]))
        turns = [
            DialogTurn(
                turn_index=t,
                query_text=make_query(dataset.queries[i]["dialog"], t),
                query_vector=all_text_embs[t][i]
            )
            for t in range(NUM_ROUNDS)
        ]
        sessions.append(RetrievalSession(session_id=i, turns=turns, target_index=target_index))

    cfg = copy.deepcopy(OPTIMAL_CONFIG)
    cfg.mode = "full"
    
    soft_sched = DynamicScheduleConfig(
        alpha_start=SCHEDULE_ALPHA_START,
        alpha_end=SCHEDULE_ALPHA_END,
        ortho_start=SCHEDULE_ORTHO_START,
        ortho_end=SCHEDULE_ORTHO_END,
        penalty_start=SCHEDULE_PENALTY_START,
        penalty_end=SCHEDULE_PENALTY_END,
        warmup_turns=SCHEDULE_WARMUP_TURNS,
        itm_start=SCHEDULE_ITM_START,
        itm_end=SCHEDULE_ITM_END,
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
        device=device, schedule_fn=piecewise_schedule_fn,
    )

    ranks_per_round = [[] for _ in range(NUM_ROUNDS)]
    start_idx = 0
    
    if os.path.exists(RESULTS_PATH):
        try:
            with open(RESULTS_PATH, "r") as f:
                old_data = json.load(f)
            if "blip_nacir_full" in old_data:
                old_ranks = old_data["blip_nacir_full"].get("raw_ranks_per_round", [])
                if old_ranks and len(old_ranks[0]) > 0:
                    ranks_per_round = old_ranks
                    start_idx = len(old_ranks[0])
                    logger.info("Resuming from query %s / %s...", start_idx, total_queries)
        except Exception as e:
            logger.warning("Could not read previous results, restarting: %s", e)
            
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
                    
            queries_done = min(i + BATCH_SIZE, total_queries)
            
            valid_t10 = [x for x in ranks_per_round[10] if x is not None]
            hits10_t10 = (sum(1 for x in valid_t10 if x < 10) / len(valid_t10) * 100) if valid_t10 else 0.0
            
            logger.info("Processed %4d/%d queries | Current Hit@10 Turn 10: %.2f%%", queries_done, total_queries, hits10_t10)
            
            save_results({
                "blip_nacir_full": {
                    "raw_ranks_per_round": ranks_per_round,
                }
            })

    logger.info("Evaluation completed.")
    logger.info("Results saved to: %s", RESULTS_PATH)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Unhandled exception:")
        raise