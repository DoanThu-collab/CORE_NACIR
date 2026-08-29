#!/usr/bin/env bash
set -euo pipefail

BELIEFS="${BELIEFS:?Set BELIEFS to the frozen belief artifact path}"

PYTHONPATH=src python scripts/experiments/reencode_host_sessions.py \
  --source-sessions artifacts_final/sessions_plugir_cr_blip.pt \
  --adapter-module nacir.adapters.openai_clip_vitl14 \
  --adapter-func load_clip_text_encoder \
  --output artifacts_final/sessions_plugir_cr_clip_vitl14.pt

PYTHONPATH=src python scripts/evaluate.py \
  --method h0 \
  --corpus-vectors artifacts_final/corpus_blip_large_vectors.pt \
  --sessions artifacts_final/sessions_plugir_cr_blip.pt \
  --output runs_deadline/plugir_cr_blip_h0

PYTHONPATH=src python scripts/evaluate.py \
  --method current \
  --corpus-vectors artifacts_final/corpus_blip_large_vectors.pt \
  --sessions artifacts_final/sessions_plugir_cr_blip.pt \
  --beliefs "$BELIEFS" \
  --adapter-module nacir.adapters.plugir_blip \
  --adapter-func load_blip_text_encoder \
  --output runs_deadline/plugir_cr_blip_current

PYTHONPATH=src python scripts/evaluate.py \
  --method persistent \
  --corpus-vectors artifacts_final/corpus_blip_large_vectors.pt \
  --sessions artifacts_final/sessions_plugir_cr_blip.pt \
  --beliefs "$BELIEFS" \
  --adapter-module nacir.adapters.plugir_blip \
  --adapter-func load_blip_text_encoder \
  --output runs_deadline/plugir_cr_blip_persistent

PYTHONPATH=src python scripts/evaluate.py \
  --method h0 \
  --corpus-vectors artifacts_final/corpus_openai_clip_vitl14_vectors.pt \
  --sessions artifacts_final/sessions_plugir_cr_clip_vitl14.pt \
  --output runs_deadline/plugir_cr_clip_h0

PYTHONPATH=src python scripts/evaluate.py \
  --method current \
  --corpus-vectors artifacts_final/corpus_openai_clip_vitl14_vectors.pt \
  --sessions artifacts_final/sessions_plugir_cr_clip_vitl14.pt \
  --beliefs "$BELIEFS" \
  --adapter-module nacir.adapters.openai_clip_vitl14 \
  --adapter-func load_clip_text_encoder \
  --output runs_deadline/plugir_cr_clip_current

PYTHONPATH=src python scripts/evaluate.py \
  --method persistent \
  --corpus-vectors artifacts_final/corpus_openai_clip_vitl14_vectors.pt \
  --sessions artifacts_final/sessions_plugir_cr_clip_vitl14.pt \
  --beliefs "$BELIEFS" \
  --adapter-module nacir.adapters.openai_clip_vitl14 \
  --adapter-func load_clip_text_encoder \
  --output runs_deadline/plugir_cr_clip_persistent

PYTHONPATH=src python scripts/analysis/analyze_cross_host_matrix.py
