# NACIR

Reference implementation for **NACIR**, a training-free conversational image retrieval method that accumulates signed natural-language beliefs over dialogue turns. This repository is the clean paper-release package: it contains the frozen F1 method, the H0/H1/F1 evaluation protocol, small tests, and compact result summaries. It does not contain raw datasets, private paths, generated belief files, full experiment caches, or exploratory modules that are outside the final method.

## What This Code Solves

Conversational image retrieval receives a sequence of turns about the same target image. Later turns often add constraints, clarify earlier answers, or explicitly rule out visual attributes. A raw retriever that ranks with only the current query loses this dialogue state. A naive memory can also be risky: negative evidence may globally suppress good candidates or move the ranking too aggressively.

NACIR solves this by maintaining signed belief memory:

- **Positive beliefs** describe attributes that should be preserved across turns.
- **Negative beliefs** describe attributes that should be avoided.
- **H1 anchor** creates a conservative signed-memory ranking.
- **F1 dual-route trust fusion** adds a positive proposal route and uses negative evidence only as a locally constrained correction before fusing back with the H1 anchor.

The final paper method is intentionally simple: no training, no learned gate, no target-aware routing, no visual feedback, and no ITM reranking in the headline protocol.

## Current Release Status

This is an advisor-facing/public-release staging repo. The core method has been separated from the experimental workspace. Before public GitHub release, complete the items in [docs/PUBLIC_RELEASE_CHECKLIST.md](docs/PUBLIC_RELEASE_CHECKLIST.md), especially license selection, private-path scan, and final packaging tests.

Verified aggregate results are stored in [results/](results/). They are compact summaries only; they are not a substitute for the original run artifacts archived in the research workspace.

## Repository Layout

```text
configs/
  f1_frozen.json          Frozen F1 configuration used by the paper method.
  paths.example.json      Example local path map for external inputs.

docs/
  METHOD.md               Method/protocol description.
  INPUT_FORMAT.md         Required tensor and belief artifact formats.
  REPRODUCTION.md         Commands for H0/H1/F1 and paired comparisons.
  ADAPTERS.md             How to connect NACIR to PlugIR or another pipeline.
  DATA_AND_LICENSES.md    Data/model ownership notes.
  PUBLIC_RELEASE_CHECKLIST.md

scripts/
  evaluate_precomputed.py Run H0, H1, or F1 from external precomputed tensors.
  compare_runs.py         Paired BRI/Recall/McNemar/Holm comparison.

src/nacir/
  core/                   Training-free NACIR logic: memory, projection, masking, APC, fusion.
  adapters/plugir_blip.py PlugIR-compatible BLIP text adapter used by the paper setup.
  interfaces.py           Minimal protocols external adapters must satisfy.
  pipeline.py             F1 pipeline over vectors and belief bundles.
  evaluation.py           H0/H1/F1 evaluation protocol.
  metrics.py              BRI, Recall@K, Hits@K metrics.
  statistics.py           Paired bootstrap and statistical tests.

tests/                    Lightweight unit/protocol tests.
```

## Installation

Create or activate a Python environment with PyTorch available. For advisor review on the existing machine, prefer using the current `nacir` environment.

```bash
cd /mlcv1/WorkingSpace/Personal/core_baotg/thuy/NACIR_FIX/NACIR_PAPER_RELEASE
pip install -e '.[dev]'
pytest
```

If the environment already has all runtime dependencies and you only want to register the local package, use:

```bash
pip install -e . --no-deps
python -m pytest
```

If `pytest` is not installed in the active environment, run:

```bash
python -m pip install pytest
python -m pytest
```

## Inputs Required To Run Experiments

The release evaluator expects precomputed tensors. This keeps NACIR independent of any private dataset loader or feature cache.

You need three external files:

- `corpus_vectors.pt`: tensor `[num_images, dim]`, or a dictionary with key `vectors`.
- `sessions.pt`: list of retrieval sessions with `session_id`, `target_index`, and `query_vectors` `[num_turns, dim]`.
- `beliefs_complete.json`: NACIR schema-v2 complete belief artifact for H1/F1.

See [docs/INPUT_FORMAT.md](docs/INPUT_FORMAT.md) for the exact contract.

## Run H0, H1, And F1

H0 is the raw-query baseline and does not read beliefs:

```bash
python scripts/evaluate_precomputed.py   --mode h0   --config configs/f1_frozen.json   --corpus-vectors /path/to/corpus_vectors.pt   --sessions /path/to/sessions.pt   --output outputs/h0
```

H1 is the signed-memory anchor:

```bash
python scripts/evaluate_precomputed.py   --mode h1   --config configs/f1_frozen.json   --corpus-vectors /path/to/corpus_vectors.pt   --sessions /path/to/sessions.pt   --beliefs /path/to/beliefs_complete.json   --output outputs/h1
```

F1 is the final method:

```bash
python scripts/evaluate_precomputed.py   --mode f1   --config configs/f1_frozen.json   --corpus-vectors /path/to/corpus_vectors.pt   --sessions /path/to/sessions.pt   --beliefs /path/to/beliefs_complete.json   --output outputs/f1
```

Each run writes:

- `report.json`: aggregate metrics and method config.
- `ranks.npz`: rank matrix for paired comparison.
- `turn_traces.jsonl`: per-turn diagnostics.

Compare aligned runs:

```bash
python scripts/compare_runs.py outputs/h1 outputs/f1 --output outputs/compare_h1_f1.json
```

## Frozen F1 Configuration

The frozen configuration is [configs/f1_frozen.json](configs/f1_frozen.json). 

We do not claim exhaustive hyperparameter optimization. NACIR is evaluated as a frozen training-free retrieval overlay. We support the frozen configuration with: (i) component ablations for memory/projection/masking/proposal routes, (ii) replication across belief generators, and (iii) targeted sensitivity on the two APC controls most directly governing intervention strength: KL budget and negative evidence strength.

The parameters fall into three distinct categories for evaluation:

1. **APC Proposal Controls (Validated by Sensitivity)**
   * `apc_max_kl = 0.002`: The maximum KL divergence budget for the asymmetrical proposal.
   * `apc_negative_strength = 0.275`: The strength of negative evidence interpolation.
   * *Validation*: These parameters directly restrict how deeply negative evidence can perturb the anchor ranking. They are supported by a targeted scalar sensitivity sweep demonstrating stable Recall@10 and BRI around the frozen point.

2. **Anchor Weights (Validated by Component Ablation)**
   * `positive_memory_weight = 0.55`
   * `negative_memory_weight = 0.275`
   * `projection_strength = 0.20`
   * `masking_threshold` / `max_penalty` / `temperature` = `0.25` / `0.18` / `0.10`
   * *Validation*: These are fixed implementation constants of the signed-memory anchor. Their contributions are rigorously supported by module-wise ablations (e.g. positive-only, negative-only, no-projection, no-masking, current-turn-only). We validate the mechanisms, not their individual numerical optimality.

3. **Design & Protocol Constants (Not Tuned)**
   * `max_concepts = 50`
   * `recency_decay = 0.10`
   * `override_boost = 0.15`
   * `candidate_k = 500` (limits locality of the proposal routing mechanism)
   * `apc_posterior_temperature = 0.05` & `apc_min_spread = 0.01` (numerical stability/calibration)
   * `top_k = 10`
   * *Validation*: We fix these values across all experiments as implementation or protocol constraints. They are not swept per model or per run.

## Adapter Design

NACIR is plug-and-play at the vector interface. It does not require retraining the retriever. To attach NACIR to another retrieval stack, the external pipeline must provide:

- normalized or normalizable corpus vectors with shape `[N, D]`;
- one query vector per dialogue turn with shape `[D]`;
- a text encoder that maps belief strings into the same embedding dimension `D`;
- a stable corpus row order so `target_index` and rankings refer to the same images across H0/H1/F1.

The only required Python protocol is in [src/nacir/interfaces.py](src/nacir/interfaces.py):

```python
class TextEncoder(Protocol):
    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        """Return normalized embeddings with shape [len(texts), D]."""
```

The current paper adapter is [src/nacir/adapters/plugir_blip.py](src/nacir/adapters/plugir_blip.py). It loads the pinned BLIP retrieval text tower used with the PlugIR-style vector space. The legacy import path `nacir.encoder` is kept for compatibility.

By default, `evaluate_precomputed.py` uses PlugIR BLIP. To swap to a new pipeline, you simply pass your custom encoder to the script:

```bash
python scripts/evaluate_precomputed.py \
  --mode f1 \
  --config configs/f1_frozen.json \
  --corpus-vectors ... \
  --sessions ... \
  --beliefs ... \
  --output outputs/f1 \
  --adapter-module my_custom_adapter.clip_encoder \
  --adapter-func load_clip_text_encoder
```

For a new retriever or dataset, write a small adapter that satisfies `TextEncoder` and exports the external vectors into the input format above. The NACIR core does not need to know whether vectors came from PlugIR, another BLIP setup, CLIP, a dense retriever, or a dataset-specific cache, as long as query, corpus, and belief-text vectors share the same space.

See [docs/ADAPTERS.md](docs/ADAPTERS.md) for a step-by-step partner guide.

## What Is Not In The Headline Method

The following modules are intentionally excluded from this release and should not be described as part of F1:

- ITM reranking;
- learned gate;
- visual feedback;
- counterfactual rerouting;
- target-aware oracle routing;
- belief generation scripts;
- raw image data, caches, or private experiment paths.

The BLIP checkpoint name contains `itm` because it is the upstream model name. In the headline NACIR protocol, ITM reranking is disabled. The release pipeline uses the text embedding interface, not an ITM reranking step.

## Handoff Checklist For A Partner Pipeline

1. Export `corpus_vectors.pt` with the exact corpus row order used by retrieval.
2. Export `sessions.pt` with `session_id`, `target_index`, and per-turn `query_vectors` in the same vector space.
3. Provide or implement a `TextEncoder` adapter for belief strings in that vector space.
4. Validate the belief artifact against NACIR schema-v2 before evaluation.
5. Run H0, H1, and F1 with the same config and session order.
6. Run `compare_runs.py` for H0/H1, H1/F1, and H0/F1.
7. Archive `report.json`, `ranks.npz`, `turn_traces.jsonl`, and comparison JSON files.

## Citation And License

Add the final paper citation and an author-approved license before public release. Until then, see [LICENSE_PENDING.md](LICENSE_PENDING.md).
