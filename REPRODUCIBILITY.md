# NACIR Reproducibility Record

## Canonical main-evaluation freeze

Git tag:

    paper-main-eval-v1

This tag corresponds to the unified evaluator used for the main NACIR
experiments.

The evaluator was regression-tested against the previously frozen rank
artifacts for all six main conditions:

| Backbone | H0 | Current-turn | Persistent NACIR |
|---|---|---|---|
| BLIP | exact match | exact match | exact match |
| OpenAI CLIP ViT-L/14 | exact match | exact match | exact match |

For every run:

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

## Canonical belief artifact

    /mlcv1/WorkingSpace/Personal/core_baotg/thuy/NACIR_FIX/data/beliefs_v2/llama3_1_8b_v9_final_20260824.json

## Frozen NACIR configuration

    configs/nacir_minus_frozen.json

Main parameters:
- lambda = 0.275
- rho = 0.10
- max memory concepts = 50
- semantic merge = disabled

## Notes

NACIR is a negative-only persistent belief-state retrieval adapter.
Positive beliefs are not used by the canonical main method.

The unified evaluator performs a fail-fast dimension check between the
backbone-specific negative-concept text encoder and the retrieval embedding
space.
