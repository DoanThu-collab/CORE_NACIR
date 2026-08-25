# Reproduction guide

## 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

## 2. Prepare external inputs

Create the vector and session files described in [INPUT_FORMAT.md](INPUT_FORMAT.md). Audit the belief file with the artifact-generation repository before it enters this release evaluator. It must be complete and provenance-bound.

## 3. Run the frozen ablation

H0 is the raw-query baseline:

```bash
python scripts/evaluate_precomputed.py \
  --mode h0 \
  --corpus-vectors /path/to/corpus_vectors.pt \
  --sessions /path/to/sessions.pt \
  --output outputs/h0
```

H1 is the signed-memory anchor:

```bash
python scripts/evaluate_precomputed.py \
  --mode h1 \
  --corpus-vectors /path/to/corpus_vectors.pt \
  --sessions /path/to/sessions.pt \
  --beliefs /path/to/beliefs_complete.json \
  --output outputs/h1
```

F1 is the final method:

```bash
python scripts/evaluate_precomputed.py \
  --mode f1 \
  --corpus-vectors /path/to/corpus_vectors.pt \
  --sessions /path/to/sessions.pt \
  --beliefs /path/to/beliefs_complete.json \
  --output outputs/f1
```

The evaluator writes `ranks.npz`, `report.json`, and `turn_traces.jsonl`. `ranks.npz` is the only required input to the paired comparison command.

## 4. Compare aligned runs

```bash
python scripts/compare_runs.py outputs/h0 outputs/h1 --output outputs/compare_h0_h1.json
python scripts/compare_runs.py outputs/h1 outputs/f1 --output outputs/compare_h1_f1.json
python scripts/compare_runs.py outputs/h0 outputs/f1 --output outputs/compare_h0_f1.json
```

Each comparison uses paired sessions and writes BRI difference with a 95% bootstrap interval, per-turn Recall@10 differences with bootstrap intervals, exact McNemar p-values, and Holm-adjusted p-values.

## 5. Sanity checks before reporting

- Keep the same corpus order and the same session order across all runs.
- Preserve zero-indexed ranks and use `top_k=10`.
- Verify the F1 configuration equals `configs/f1_frozen.json`.
- Verify no target index is passed to the pipeline before score computation.
- Archive the generated comparison JSON files alongside the final paper tables.
