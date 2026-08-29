#!/usr/bin/env bash
set -euo pipefail
BELIEFS="${BELIEFS:-/mlcv1/WorkingSpace/Personal/core_baotg/thuy/NACIR_FIX/data/beliefs_v2/llama3_1_8b_v9_final_20260824.json}"

PYTHONPATH=src python scripts/experiments/evaluate_text_reencode_control.py \
  --corpus-vectors artifacts_final/corpus_blip_large_vectors.pt \
  --sessions artifacts_final/sessions_chatir_blip.pt \
  --beliefs "$BELIEFS" \
  --adapter-module nacir.adapters.plugir_blip \
  --adapter-func load_blip_text_encoder \
  --frozen-h0-ranks runs_final/chatir_blip_h0/ranks.npz \
  --output runs_deadline/blip_text_control

PYTHONPATH=src python scripts/experiments/evaluate_text_reencode_control.py \
  --corpus-vectors artifacts_final/corpus_openai_clip_vitl14_vectors.pt \
  --sessions artifacts_final/sessions_chatir_clip_vitl14.pt \
  --beliefs "$BELIEFS" \
  --adapter-module nacir.adapters.openai_clip_vitl14 \
  --adapter-func load_clip_text_encoder \
  --frozen-h0-ranks runs_final/chatir_clip_vitl14_h0/ranks.npz \
  --output runs_deadline/clip_text_control
