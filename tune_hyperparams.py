"""
NACIR++ — Hyperparameter Tuning
===============================
Kịch bản chạy Optuna Bayesian Optimization để tìm ra bộ tham số cấu hình cứng (Optimal Params)
tạo ra SOTA result (như đã thiết lập trong main.py).

Usage:
    CUDA_VISIBLE_DEVICES=0 python tune_hyperparams.py
"""

import os
import sys
import json
import torch
import numpy as np
from tqdm import tqdm
import logging
from typing import List, Dict, Optional
from PIL import Image
import time
import optuna
from optuna.trial import Trial

from core.semantic_parser import load_precomputed_beliefs
from core.query_update import NACIRPlusPlusConfig, NACIRPlusPlusBatchUpdater
from transformers import AutoProcessor, BlipForImageTextRetrieval

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

# ============================================================
# Cấu hình "cứng" (Hardcoded Configurations)
# ============================================================

DATA_DIR = "/AIClub_NAS/core_baotg/thuyntn/Datasets/PlugIR/"
QUERIES_PATH = "/AIClub_NAS/core_baotg/thuyntn/PlugIR_Workspace/PlugIR/dialogues/VisDial_v1.0_queries_val.json"
PLUGIR_QUERIES_PATH = "/AIClub_NAS/core_baotg/thuyntn/PlugIR/dialogues/ours_final_q_n_5_recall_hitting_10_thres_low_500_recon_true_referring_true_filtering_true_select_true_reconed.json"
CACHE_CORPUS_PATH = "/AIClub_NAS/core_baotg/thuyntn/PlugIR_Workspace/ChatIR/temp/corpus_blip_large.pth"
CORPUS_PATH = "/AIClub_NAS/core_baotg/thuyntn/PlugIR_Workspace/PlugIR/Protocol/Search_Space_val_50k.json"
BELIEFS_PATH = "/AIClub_NAS/core_baotg/thuyntn/NACIR/data/semantic_beliefs.json"
OUTPUT_DIR = "logs/optuna_search"

BATCH_SIZE = 64
N_TRIALS = 80
BLIP_MODEL = "Salesforce/blip-itm-large-coco"

# ============================================================
# Model definitions
# ============================================================

class BlipForRetrieval(BlipForImageTextRetrieval):
    def get_text_features(self, input_ids, attention_mask=None, return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        question_embeds = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=return_dict)
        question_embeds = question_embeds[0] if not return_dict else question_embeds.last_hidden_state
        return torch.nn.functional.normalize(self.text_proj(question_embeds[:, 0, :]), dim=-1)

class SimpleCorpus:
    def __init__(self, data_dir, corpus_path):
        with open(corpus_path) as f:
            corpus = json.load(f)
        self.corpus = [os.path.join(data_dir, path) for path in corpus]
        self.data_dir = data_dir
        self.path2id = {self.corpus[i]: i for i in range(len(self.corpus))}
    def path_to_index(self, path): return self.path2id[path]

class Queries(torch.utils.data.Dataset):
    def __init__(self, queries_path, data_dir, sep_token=', ', split=True):
        with open(queries_path) as f:
            self.queries = json.load(f)
        self.dialog_length = None
        self.data_dir = data_dir
        self.sep_token = sep_token
        self.split = split
    def __len__(self): return len(self.queries)
    def __getitem__(self, i):
        target_path = os.path.join(self.data_dir, self.queries[i]['img'])
        text = self.queries[i]['dialog'][self.dialog_length] if self.split else self.sep_token.join(self.queries[i]['dialog'][:self.dialog_length + 1])
        return {'text': text, 'target_path': target_path}

def run_single_config(config, all_text_embs, corpus_vectors, corpus_dataset, original_queries, beliefs, dialog_encoder, device, batch_size=64, num_rounds=11):
    num_queries = len(original_queries)
    ranks_per_round = [[] for _ in range(num_rounds)]

    for i in range(0, num_queries, batch_size):
        end_idx = min(i + batch_size, num_queries)
        batch_size_actual = end_idx - i
        target_indices = [corpus_dataset.path_to_index(os.path.join(corpus_dataset.data_dir, original_queries[j]["img"])) for j in range(i, end_idx)]

        updater = NACIRPlusPlusBatchUpdater(config=config, batch_size=batch_size_actual, encoder=dialog_encoder, device=device)

        for t in range(num_rounds):
            q_t = all_text_embs[t][i:end_idx].to(device)
            if t > 0:
                beliefs_batch = [{"positive_beliefs": beliefs.get(i+b, {}).get(t-1, {}).get("positive_beliefs", []),
                                  "negative_beliefs": beliefs.get(i+b, {}).get(t-1, {}).get("negative_beliefs", [])} for b in range(batch_size_actual)]
                q_t = updater.update_query(q_text_batch=q_t, beliefs_batch=beliefs_batch, turn=t)
            
            scores = q_t @ corpus_vectors.T
            if t > 0: scores = updater.apply_masking(scores, corpus_vectors)

            ranked = torch.argsort(scores, descending=True)
            for b in range(batch_size_actual):
                ranks_per_round[t].append((ranked[b] == target_indices[b]).nonzero(as_tuple=True)[0].item())

    dialog_recalls_list = [torch.tensor(r, dtype=torch.long) for r in ranks_per_round]
    min_ranks = [dialog_recalls_list[0].float()]
    for t in range(1, num_rounds): min_ranks.append(torch.minimum(min_ranks[t-1], dialog_recalls_list[t].float()))
    bri = sum(((torch.log(min_ranks[t] + 1.0) + torch.log(min_ranks[t+1] + 1.0)) / 2).mean() for t in range(num_rounds - 1)) / (num_rounds - 1)
    return bri.item()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                        handlers=[logging.FileHandler(os.path.join(OUTPUT_DIR, "optuna_search.log")), logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading BLIP model...")
    model = BlipForRetrieval.from_pretrained(BLIP_MODEL).to(device)
    processor = AutoProcessor.from_pretrained(BLIP_MODEL)
    dialog_encoder = lambda text: model.get_text_features(**{k: v.to(device) for k, v in processor(text=text, padding=True, truncation=True, return_tensors="pt").items()})

    logger.info("Loading corpus cache...")
    corpus_ids, corpus_vectors = torch.load(CACHE_CORPUS_PATH, map_location=device)
    corpus_dataset = SimpleCorpus(DATA_DIR, CORPUS_PATH)

    with open(QUERIES_PATH) as f: original_queries = json.load(f)
    dataset = Queries(PLUGIR_QUERIES_PATH, DATA_DIR, split=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    beliefs = load_precomputed_beliefs(BELIEFS_PATH)

    logger.info("Pre-computing text features...")
    all_text_embs = []
    for dl in range(11):
        dataset.dialog_length = dl
        round_embs = []
        for batch in tqdm(dataloader, desc=f"Text Round {dl}"):
            with torch.no_grad(): round_embs.append(dialog_encoder(batch['text']))
        all_text_embs.append(torch.cat(round_embs))

    best_bri_ever = float('inf')
    trial_results = []

    def objective(trial: Trial) -> float:
        nonlocal best_bri_ever
        alpha = trial.suggest_float("alpha", 0.05, 0.6, step=0.05)
        beta_ratio = trial.suggest_float("beta_ratio", 0.3, 0.7, step=0.1)
        ortho_strength = trial.suggest_float("ortho_strength", 0.1, 1.0, step=0.1)
        mask_penalty = trial.suggest_float("mask_penalty", 0.02, 0.35, step=0.02)
        mask_threshold = trial.suggest_float("mask_threshold", 0.10, 0.35, step=0.05)
        
        config = NACIRPlusPlusConfig(
            memory_alpha=alpha, memory_beta=alpha * beta_ratio, positive_blend_alpha=alpha,
            ortho_strength=ortho_strength, masking_penalty_weight=mask_penalty,
            masking_threshold=mask_threshold, recency_decay=0.1, concept_match_threshold=0.85, mode="full"
        )

        start_time = time.time()
        bri = run_single_config(config, all_text_embs, corpus_vectors, corpus_dataset, original_queries, beliefs, dialog_encoder, device, BATCH_SIZE)
        elapsed = time.time() - start_time

        is_best = bri < best_bri_ever
        if is_best: best_bri_ever = bri
        logger.info(f"[Trial {trial.number+1}/{N_TRIALS}] α={alpha:.2f}, ortho={ortho_strength:.1f}, mask_p={mask_penalty:.2f} → BRI={bri:.4f} ({elapsed:.1f}s){' ★ NEW BEST!' if is_best else ''}")
        
        trial_results.append({"trial": trial.number, "alpha": alpha, "ortho_strength": ortho_strength, "mask_penalty": mask_penalty, "bri": bri})
        return bri

    logger.info("Starting Optuna Bayesian Optimization...")
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS)

    logger.info("\n★ BEST TRIAL:")
    logger.info(f"  BRI = {study.best_trial.value:.4f}")
    for k, v in study.best_trial.params.items(): logger.info(f"  {k} = {v}")

    results_path = os.path.join(OUTPUT_DIR, "optuna_results.json")
    with open(results_path, 'w') as f: json.dump(sorted(trial_results, key=lambda x: x["bri"]), f, indent=2)
    logger.info(f"Results saved to {results_path}")

if __name__ == '__main__':
    main()
