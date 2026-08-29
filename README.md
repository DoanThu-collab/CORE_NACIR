# NACIR: Negative-Aware Conversational Image Retrieval

Implementation of **NACIR**, a training-free retrieval adapter for preserving exclusionary evidence across multi-turn conversational image retrieval.

The canonical method is intentionally minimal: it keeps a persistent memory of negative concepts and subtracts their weighted embedding direction from the host query before retrieval.

## Canonical evaluation conditions

The unified evaluator exposes three conditions:

- `h0`: host retriever only.
- `current`: use negative evidence from the current feedback turn only.
- `persistent`: retain negative evidence across dialogue turns (NACIR).

The frozen configuration is `configs/nacir_minus_frozen.json`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

For the OpenAI CLIP ViT-L/14 adapter, install OpenAI CLIP separately:

```bash
pip install git+https://github.com/openai/CLIP.git
```

## Evaluate

The evaluator expects precomputed session/query embeddings and corpus vectors in the formats documented in `docs/INPUT_FORMAT.md`.

Host baseline:

```bash
PYTHONPATH=src python scripts/evaluate.py \
  --method h0 \
  --corpus-vectors /path/to/corpus_vectors.pt \
  --sessions /path/to/sessions.pt \
  --output outputs/h0
```

Current-turn negative adaptation:

```bash
PYTHONPATH=src python scripts/evaluate.py \
  --method current \
  --corpus-vectors /path/to/corpus_vectors.pt \
  --sessions /path/to/sessions.pt \
  --beliefs /path/to/beliefs.json \
  --output outputs/current
```

Persistent NACIR:

```bash
PYTHONPATH=src python scripts/evaluate.py \
  --method persistent \
  --corpus-vectors /path/to/corpus_vectors.pt \
  --sessions /path/to/sessions.pt \
  --beliefs /path/to/beliefs.json \
  --config configs/nacir_minus_frozen.json \
  --output outputs/persistent
```

Use an adapter that matches the retrieval embedding space. See `docs/ADAPTERS.md`.

## Repository layout

- `src/nacir/`: canonical implementation.
- `scripts/evaluate.py`: unified H0 / Current / Persistent evaluator.
- `scripts/analysis/`: paper analysis and audit utilities.
- `scripts/experiments/`: auxiliary experiment entry points.
- `scripts/prepare/`: embedding preparation utilities.
- `tests/`: protocol and regression checks.
- `docs/`: method, input-format, adapter, and data/license documentation.
- `results/`: compact paper-result summaries only; raw experiment artifacts are intentionally not tracked.

## Reproducibility

See `REPRODUCIBILITY.md` for the frozen main evaluation, exact-rank regression record, and final aggregate metrics.

## Data and licenses

The repository does not redistribute external datasets, model weights, frozen corpus embeddings, or private experiment workspaces. See `docs/DATA_AND_LICENSES.md`.
