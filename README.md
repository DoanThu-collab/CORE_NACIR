# NACIR: Non-Parametric Adaptive Concept Interactive Retrieval

This directory contains the final, cleaned-up source code for NACIR++, which achieves state-of-the-art results (BRI = 0.6861) on the VisDial Interactive Image Retrieval benchmark.

## Repository Structure

- `main.py`: The primary end-to-end evaluation script. It uses hardcoded optimal hyperparameters and absolute paths for seamless execution.
- `tune_hyperparams.py`: The Optuna Bayesian Optimization script used to discover the optimal hyperparameters hardcoded in `main.py`.
- `core/`: The heart of NACIR++ containing:
  - `semantic_parser.py`: Concept extraction logic from pre-computed beliefs.
  - `query_update.py`: The `NACIRPlusPlusBatchUpdater` which orchestrates Concept Memory, Orthogonal Projection, and Attention Masking.
  - `reranker.py`: The ITM Cross-Attention Re-ranking module.
- `utils/`: Auxiliary logic (e.g., VisDial dialog parsing).

## How to Run

Because this is a final release for your local server, all dataset and model paths have been pre-configured to point directly to your NAS workspace:
`/AIClub_NAS/core_baotg/thuyntn/`

### 1. Run the Evaluation
Simply run:
```bash
CUDA_VISIBLE_DEVICES=index python main.py
```
This script will:
- Load the BLIP dual-encoder.
- Load the pre-computed corpus cache.
- Extract Positive/Negative beliefs from the dialog.
- Perform Vector Surgery (Positive Blend + Orthogonal Projection).
- Re-rank the Top-50 candidates using the ITM Cross-Encoder.
- Output the `Cumulative Hits@10`, `Per-round Recall@10`, and `BRI`.
- Save the ranks to `logs/nacir_plus_ranks_final.npz`.

### 2. Run Hyperparameter Tuning
If you wish to re-run or verify the Bayesian Optimization process that found the optimal hyperparameters:
```bash
CUDA_VISIBLE_DEVICES=index python tune_hyperparams.py
```

## Core Innovations
1. **Concept Memory Board**: Memorizes extracted features across long conversations, preventing the "Long-context Forgetting" typical in standard LLM summarization.
2. **Vector Surgery (Orthogonal Projection)**: Uses Gram-Schmidt orthogonalization to mathematically purge negative concepts from the query vector space.
3. **ITM Re-ranking**: Integrates a robust cross-attention verification step at the end of the pipeline to push highly relevant candidates to the absolute top ranks.
