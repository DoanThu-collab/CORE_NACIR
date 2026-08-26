# NACIR-: Negative-Only Belief Memory for Conversational Image Retrieval

This repository contains the official, simplified implementation of **NACIR-**, a training-free layer over a frozen retriever that handles multi-turn conversational image retrieval by extracting and subtracting negative evidence.

## Why is it so simple?

Earlier iterations of our methodology explored complex, multi-route mechanisms, including:
- Dual-Route Fusion (blending positive and negative memory rankings).
- Orthogonal Projection (removing query vector projection along negative axes).
- Corpus-wise Masking (penalizing specific corpus items based on negative similarities).
- Asymmetric KL Proposal Constraints (APC).

In our final published evaluation, we demonstrated that **all these additions artificially inflate trajectory metrics (like Best-log-Rank Integral) but consistently degrade per-turn Recall@10 accuracy.** 

Therefore, our final proposed method discards all of them. The entire core of NACIR- fits in a single line (Eq. 2 of the paper):
```python
q_minus = norm(base_query - 0.275 * norm(negative_memory_vector))
```

## Legacy Code Archives
To maintain full scientific transparency and allow readers to verify our ablation studies against the flawed trajectory metrics, we have preserved the rejected components in the `src/nacir/core/legacy_variants/` directory. These files (such as `dual_route_fusion.py`, `masking.py`, `projection.py`, etc.) are isolated and **not active** in the canonical NACIR- evaluation pipeline.

The active code evaluates strictly Negative Memory and validates the single subtraction step.

## Setup & Running

Install the dependencies:
```bash
pip install -e '.[dev]'
```

Evaluate the protocol on offline tensors (requires `sessions.pt` and `corpus_vectors.pt`):
```bash
python scripts/evaluate_precomputed.py \
    --mode nacir \
    --corpus-vectors path/to/corpus_vectors.pt \
    --sessions path/to/sessions.pt \
    --beliefs path/to/beliefs.json \
    --output runs/final_test \
    --config configs/f1_frozen.json
```
