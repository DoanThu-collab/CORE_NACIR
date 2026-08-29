#!/usr/bin/env bash
set -euo pipefail

CLIP_COMMIT="a1d071733d7111c9c014f024669f959182114e33"

python -m pip install -e '.[clip]'
python -m pip install 'setuptools<81'
python -m pip install --no-build-isolation \
  "git+https://github.com/openai/CLIP.git@${CLIP_COMMIT}"

python - <<'PY'
import clip
from PIL import Image

print("OpenAI CLIP import: PASS")
print("Pillow import: PASS")
PY
