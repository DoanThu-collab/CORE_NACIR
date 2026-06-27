"""
NACIR++ — Final Code Release (End-to-End Evaluation)
=====================================================
Chạy evaluation loop sử dụng kiến trúc NACIR++ đã được tối ưu hóa.
File này đã được cấu hình cứng (hardcode) bộ siêu tham số tạo ra kết quả tốt nhất (BRI = 0.6861).

Usage:
    CUDA_VISIBLE_DEVICES=0 python main.py
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

# Import local modules
from core.semantic_parser import load_precomputed_beliefs
from core.query_update import NACIRPlusPlusConfig, NACIRPlusPlusBatchUpdater
from core.reranker import ITMReranker
from utils.negative_detector import parse_visdial_dialog, label_dialog

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
OUTPUT_DIR = "logs"

BATCH_SIZE = 64
ITM_BATCH_SIZE = 16
RERANK_K = 50
ITM_WEIGHT = 0.7

# Tham số tối ưu của NACIR++ (Đã tune qua Optuna)
OPTIMAL_CONFIG = NACIRPlusPlusConfig(
    memory_alpha=0.55,
    memory_beta=0.275,        # Thường bằng alpha * 0.5
    positive_blend_alpha=0.55,
    ortho_strength=0.2,
    masking_penalty_weight=0.18,
    masking_threshold=0.25,
    recency_decay=0.1,
    concept_match_threshold=0.75,
    mode="full"
)

# ============================================================
# Model definitions
# ============================================================

from transformers import AutoProcessor, BlipForImageTextRetrieval

class BlipForRetrieval(BlipForImageTextRetrieval):
    def get_text_features(self, input_ids: torch.LongTensor, attention_mask: Optional[torch.LongTensor] = None, return_dict: Optional[bool] = None) -> torch.FloatTensor:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        question_embeds = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=return_dict)
        question_embeds = question_embeds[0] if not return_dict else question_embeds.last_hidden_state
        return torch.nn.functional.normalize(self.text_proj(question_embeds[:, 0, :]), dim=-1)

    def get_image_features(self, pixel_values: torch.FloatTensor, output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None) -> torch.FloatTensor:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_outputs = self.vision_model(pixel_values=pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        return torch.nn.functional.normalize(self.vision_proj(vision_outputs[0][:, 0, :]), dim=-1)

class ImageEmbedder:
    def __init__(self, model, preprocessor, expected_dim=None):
        self.model = model
        self.processor = preprocessor
        self.expected_dim = expected_dim

class Corpus(torch.utils.data.Dataset):
    def __init__(self, data_dir, corpus_path, preprocessor):
        with open(corpus_path) as f:
            self.corpus = [os.path.join(data_dir, p) for p in json.load(f)]
        self.preprocessor = preprocessor
        self.path2id = {p: i for i, p in enumerate(self.corpus)}

    def __len__(self): return len(self.corpus)
    def path_to_index(self, path): return self.path2id[path]
    def __getitem__(self, i):
        return {'id': i, 'image': self.preprocessor(self.corpus[i])['pixel_values'][0]}

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

def setup_logging(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                        handlers=[logging.FileHandler(os.path.join(output_dir, "run_nacir_plus.log")), logging.StreamHandler()])
    return logging.getLogger(__name__)

def main():
    logger = setup_logging(OUTPUT_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("=" * 60)
    logger.info("NACIR++ Final Code Release Evaluation")
    logger.info(f"Config: {OPTIMAL_CONFIG}")
    logger.info("=" * 60)

    # 1. Load Model
    blip_model_id = "Salesforce/blip-itm-large-coco"
    model = BlipForRetrieval.from_pretrained(blip_model_id).to(device)
    processor = AutoProcessor.from_pretrained(blip_model_id)
    image_embedder = ImageEmbedder(lambda img: model.get_image_features(img),
                                   lambda path: processor(images=Image.open(path), return_tensors='pt'),
                                   int(model.text_proj.out_features))
    dialog_encoder = lambda text: model.get_text_features(**{k: v.to(device) for k, v in processor(text=text, padding=True, truncation=True, return_tensors="pt").items()})

    corpus_dataset = Corpus(DATA_DIR, CORPUS_PATH, image_embedder.processor)
    corpus_ids, corpus_vectors = torch.load(CACHE_CORPUS_PATH, map_location=device)

    with open(CORPUS_PATH) as f:
        corpus_paths = [os.path.join(DATA_DIR, p) for p in json.load(f)]

    itm_reranker = ITMReranker(model=model, processor=processor, device=device, rerank_k=RERANK_K)

    dataset = Queries(PLUGIR_QUERIES_PATH, DATA_DIR, split=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    with open(QUERIES_PATH) as f:
        original_queries = json.load(f)

    beliefs = load_precomputed_beliefs(BELIEFS_PATH)
    num_queries = len(original_queries)
    num_rounds = 11

    ranks_per_round = [[] for _ in range(num_rounds)]
    top_k_per_round = [[] for _ in range(num_rounds)]
    K = 10

    # Precompute text embeddings
    all_text_embs = []
    for dl in range(num_rounds):
        dataset.dialog_length = dl
        round_embs = []
        for batch in tqdm(dataloader, desc=f"Text Round {dl}"):
            with torch.no_grad(): round_embs.append(dialog_encoder(batch['text']))
        all_text_embs.append(torch.cat(round_embs))

    # Retrieval Loop
    total_overrides = 0
    for i in tqdm(range(0, num_queries, BATCH_SIZE), desc="NACIR++ Batches"):
        end_idx = min(i + BATCH_SIZE, num_queries)
        batch_size_actual = end_idx - i
        target_indices = [corpus_dataset.path_to_index(os.path.join(DATA_DIR, original_queries[j]["img"])) for j in range(i, end_idx)]

        updater = NACIRPlusPlusBatchUpdater(config=OPTIMAL_CONFIG, batch_size=batch_size_actual, encoder=dialog_encoder, device=device)

        for t in range(num_rounds):
            q_t = all_text_embs[t][i:end_idx].to(device)
            # ── Dynamic ITM Weight (Avoid cold start at Turn 0) ──
            progress_t = min(t / 10.0, 1.0)
            dyn_itm = 0.2 + (ITM_WEIGHT - 0.2) * progress_t  # Starts at 0.2, ramps up to 0.7

            if t > 0:
                # ── Dynamic Parameter Scheduling ──
                # Use linear scale instead of quadratic to avoid abrupt penalty spikes (e.g. at Turn 8)
                progress = min((t - 1) / 9.0, 1.0)
                scale = progress  # Linear scale
                
                dyn_alpha = 0.20 + (0.60 - 0.20) * scale
                dyn_beta = dyn_alpha * 0.5
                dyn_ortho = 0.05 + (0.25 - 0.05) * scale    # Capped at 0.25 (was 0.35)
                dyn_penalty = 0.05 + (0.20 - 0.05) * scale  # Capped at 0.20 (was 0.30)

                updater.config.ortho_strength = dyn_ortho
                updater.config.masking_penalty_weight = dyn_penalty
                for b_idx in range(batch_size_actual):
                    updater.boards[b_idx].config.alpha = dyn_alpha
                    updater.boards[b_idx].config.beta = dyn_beta
                
                beliefs_batch = [{"positive_beliefs": beliefs.get(i+b, {}).get(t-1, {}).get("positive_beliefs", []),
                                  "negative_beliefs": beliefs.get(i+b, {}).get(t-1, {}).get("negative_beliefs", [])} for b in range(batch_size_actual)]
                q_t = updater.update_query(q_text_batch=q_t, beliefs_batch=beliefs_batch, turn=t)

            scores = q_t @ corpus_vectors.T
            if t > 0: scores = updater.apply_masking(scores, corpus_vectors)

            ranked = torch.argsort(scores, descending=True)
            
            for b in range(batch_size_actual):
                top_rerank = ranked[b, :RERANK_K]
                q_text_str = dataset.queries[i+b]['dialog'][t]
                
                reranked_indices, _ = itm_reranker.rerank_topk(
                    query_text=q_text_str, top_k_corpus_indices=top_rerank.cpu(),
                    corpus_paths=corpus_paths, cosine_scores=scores[b, top_rerank].cpu(),
                    itm_weight=dyn_itm, batch_size=ITM_BATCH_SIZE
                )
                full_reranked = torch.cat([reranked_indices.to(device), ranked[b, RERANK_K:]])
                ranks_per_round[t].append((full_reranked == target_indices[b]).nonzero(as_tuple=True)[0].item())
                top_k_per_round[t].append(full_reranked[:K].cpu().tolist())

        total_overrides += updater.get_batch_stats()["total_overrides"]

    # Metrics computation
    dialog_recalls_list = [torch.tensor(r, dtype=torch.long) for r in ranks_per_round]
    
    final_hits = torch.inf * torch.ones(num_queries)
    hitting_times, temp_hitting_times = [], []
    for ro_i in range(num_rounds):
        temp_hits = torch.inf * torch.ones(num_queries)
        rh = dialog_recalls_list[ro_i] < K
        final_hits[rh] = torch.min(final_hits[rh], torch.ones(final_hits[rh].shape) * ro_i)
        temp_hits[rh] = torch.min(temp_hits[rh], torch.ones(temp_hits[rh].shape) * ro_i)
        hitting_times.append(final_hits.clone())
        temp_hitting_times.append(temp_hits)
        
    ht_times, temp_ht_times = torch.stack(hitting_times), torch.stack(temp_hitting_times)
    cumulative_hits = (ht_times < torch.inf).sum(dim=-1).float() * 100 / num_queries
    per_round_recall = (temp_ht_times < torch.inf).sum(dim=-1).float() * 100 / num_queries

    logger.info("====== Results for Hits@10 ======")
    for t in range(num_rounds): logger.info(f"\t Dialog Length: {t}: {round(cumulative_hits[t].item(), 2)}%")
    logger.info("====== Results for Recall@10 ======")
    for t in range(num_rounds): logger.info(f"\t Dialog Length: {t}: {round(per_round_recall[t].item(), 2)}%")

    min_ranks = [dialog_recalls_list[0].float()]
    for t in range(1, num_rounds): min_ranks.append(torch.minimum(min_ranks[t-1], dialog_recalls_list[t].float()))
    bri = sum(((torch.log(min_ranks[t] + 1.0) + torch.log(min_ranks[t+1] + 1.0)) / 2).mean() for t in range(num_rounds - 1)) / (num_rounds - 1)
    
    logger.info("====== Best log Rank Integral ======")
    logger.info(f"\t BRI: {bri.item():.4f}")
    
    # Save results
    save_path = os.path.join(OUTPUT_DIR, "nacir_plus_ranks_final.npz")
    np.savez_compressed(save_path, ranks_per_round=np.array(ranks_per_round, dtype=object), top_k_per_round=np.array(top_k_per_round, dtype=object))
    logger.info(f"Results saved to {save_path}")

if __name__ == "__main__":
    main()
