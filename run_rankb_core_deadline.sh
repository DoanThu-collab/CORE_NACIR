#!/usr/bin/env bash
set -euo pipefail

# Run from CORE_NACIR_24H repo root.
BELIEFS="${BELIEFS:-/mlcv1/WorkingSpace/Personal/core_baotg/thuy/NACIR_FIX/data/beliefs_v2/llama3_1_8b_v9_final_20260824.json}"
STRUCTURED="${STRUCTURED:-artifacts_final/typed_nacir/chatir_structured_negative_final_v1_1.json}"

echo "============================================================"
echo "0) SYNTAX"
echo "============================================================"
PYTHONPATH=src python -m py_compile \
  scripts/analysis/clean_persistence_challenge.py \
  scripts/experiments/evaluate_weight_ablation.py \
  scripts/experiments/evaluate_text_persistent.py \
  scripts/analysis/sensitivity_final_round.py \
  scripts/analysis/summarize_rankb_deadline.py

echo
echo "============================================================"
echo "1) CLEAN PERSISTENCE CHALLENGE -- CPU, FAST"
echo "============================================================"
PYTHONPATH=src python scripts/analysis/clean_persistence_challenge.py \
  --beliefs "$BELIEFS" \
  --structured "$STRUCTURED"

echo
echo "============================================================"
echo "2) BLIP WEIGHT ABLATIONS"
echo "============================================================"
PYTHONPATH=src python scripts/experiments/evaluate_weight_ablation.py \
  --corpus-vectors artifacts_final/corpus_blip_large_vectors.pt \
  --sessions artifacts_final/sessions_chatir_blip.pt \
  --beliefs "$BELIEFS" \
  --adapter-module nacir.adapters.plugir_blip \
  --adapter-func load_blip_text_encoder \
  --concept-cache artifacts_final/cache/blip_negative_concepts.pt \
  --verify-full-ranks runs_final/chatir_blip_nacir_minus/ranks.npz \
  --output runs_deadline/blip_weight_ablation

echo
echo "============================================================"
echo "3) CLIP WEIGHT ABLATIONS"
echo "============================================================"
PYTHONPATH=src python scripts/experiments/evaluate_weight_ablation.py \
  --corpus-vectors artifacts_final/corpus_openai_clip_vitl14_vectors.pt \
  --sessions artifacts_final/sessions_chatir_clip_vitl14.pt \
  --beliefs "$BELIEFS" \
  --adapter-module nacir.adapters.openai_clip_vitl14 \
  --adapter-func load_clip_text_encoder \
  --concept-cache artifacts_final/cache/clip_vitl14_negative_concepts.pt \
  --verify-full-ranks runs_final/chatir_clip_vitl14_nacir_minus/ranks.npz \
  --output runs_deadline/clip_weight_ablation

echo
echo "============================================================"
echo "4) SUMMARY -- THIS IS THE FIRST DECISION POINT"
echo "============================================================"
PYTHONPATH=src python scripts/analysis/summarize_rankb_deadline.py

echo
echo "============================================================"
echo "CORE DEADLINE CODE DONE."
echo "If elapsed time is still safe, run text-persistent and sensitivity commands"
echo "from DEADLINE_COMMANDS.md. Do NOT overwrite frozen main runs."
echo "============================================================"
