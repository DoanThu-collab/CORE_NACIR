#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python scripts/analysis/diagnose_fast_regression.py \
  --name BLIP \
  --frozen runs_final/chatir_blip_nacir_minus/ranks.npz \
  --fast runs_deadline/blip_weight_ablation/full_ranks.npz

echo

PYTHONPATH=src python scripts/analysis/diagnose_fast_regression.py \
  --name CLIP \
  --frozen runs_final/chatir_clip_vitl14_nacir_minus/ranks.npz \
  --fast runs_deadline/clip_weight_ablation/full_ranks.npz
