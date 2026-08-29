#!/usr/bin/env bash
set -euo pipefail

BELIEFS="${BELIEFS:-/mlcv1/WorkingSpace/Personal/core_baotg/thuy/NACIR_FIX/data/beliefs_v2/llama3_1_8b_v9_final_20260824.json}"

echo "============================================================"
echo "1) BUILD SAME PlugIR HOST TEXT IN CLIP SPACE"
echo "============================================================"
PYTHONPATH=src python scripts/experiments/reencode_host_sessions.py \
  --source-sessions artifacts_final/sessions_plugir_cr_blip.pt \
  --adapter-module nacir.adapters.openai_clip_vitl14 \
  --adapter-func load_clip_text_encoder \
  --output artifacts_final/sessions_plugir_cr_clip_vitl14.pt

echo
echo "============================================================"
echo "2) PlugIR × BLIP: canonical unified evaluator"
echo "============================================================"
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

echo
echo "BLIP regression against existing stronger-host frozen ranks"
python - <<'PY'
import numpy as np
for new,old,name in [
("runs_deadline/plugir_cr_blip_h0/ranks.npz","runs_final/plugir_cr_blip_h0/ranks.npz","H0"),
("runs_deadline/plugir_cr_blip_persistent/ranks.npz","runs_final/plugir_cr_blip_nacir_minus/ranks.npz","Persistent"),
]:
    a=np.load(new)["ranks"]; b=np.load(old)["ranks"]
    print(name,"exact=",np.array_equal(a,b),"different=",int((a!=b).sum()))
PY

echo
echo "============================================================"
echo "3) PlugIR × CLIP: SAME reconstructed host text, CLIP space"
echo "============================================================"
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

echo
echo "============================================================"
echo "4) CROSS-HOST MATRIX + paired bootstrap Persistent-Current"
echo "============================================================"
PYTHONPATH=src python scripts/analysis/analyze_cross_host_matrix.py
