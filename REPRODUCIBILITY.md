# NACIR Reproducibility Record

## Canonical main evaluation

The canonical main-evaluation freeze is:

    paper-main-eval-v1

This tag contains the unified evaluator used to reproduce all six main
experimental conditions exactly against the frozen rank artifacts.

| Backbone | H0 | Current-turn | Persistent NACIR |
|---|---|---|---|
| BLIP | exact match | exact match | exact match |
| OpenAI CLIP ViT-L/14 | exact match | exact match | exact match |

For all six runs:

    rank matrix shape = (11, 2064)
    exact equality = True
    number of differing ranks = 0
    maximum absolute rank difference = 0

## Main final-round Recall@10

BLIP:
- H0: 59.399223
- Current-turn: 60.562016
- Persistent NACIR: 63.032944

OpenAI CLIP ViT-L/14:
- H0: 39.341087
- Current-turn: 39.825584
- Persistent NACIR: 41.230618

## Frozen NACIR configuration

    configs/nacir_minus_frozen.json

Main parameters:

- lambda = 0.275
- rho = 0.10
- maximum memory concepts = 50
- semantic merge = disabled

The canonical method uses negative beliefs only.

## Canonical belief artifact

The main experiments use the frozen Llama-3.1-8B belief artifact:

    llama3_1_8b_v9_final_20260824.json

The original experiment workspace stored this artifact outside the repository.
Its path in historical analysis scripts is therefore environment-specific.

## Auxiliary analyses

`scripts/analysis/` contains analysis and audit scripts used for belief-state,
negative-density, stronger-host, and diagnostic analyses.

`scripts/experiments/` contains artifact-building scripts for the stronger-host
and diagnostic experiments.

Some auxiliary scripts reference historical external datasets or frozen
workspaces through environment-specific paths. These auxiliary paths are not
part of the canonical six-run evaluator and do not affect the frozen main
results identified by `paper-main-eval-v1`.

## Evaluator safety

For NACIR conditions, the unified evaluator checks that the negative-concept
text encoder and corpus embeddings have matching dimensions before evaluation.
H0 does not invoke the belief encoder.
