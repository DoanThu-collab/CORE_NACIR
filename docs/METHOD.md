# Frozen paper method

## Scope

This implementation exposes the final training-free method only. Its method name
is **F1 dual-route trust fusion**. The ablation protocol has three runs:

| Run | Scoring rule |
| --- | --- |
| H0 | cosine similarity between the current query and every corpus vector |
| H1 | persistent signed memory, negative projection, and negative masking |
| F1 | H1 anchor plus positive-only proposal routing and trust-weighted fusion |

All vectors are L2-normalized. A session target is used only after scores have
been computed, to obtain its rank for evaluation.

## Signed memory

Every dialogue turn can add positive and negative concepts with a confidence
score. The memory is persistent over the session, retains history, and stores at
most 50 concepts. The frozen weights are:

| Parameter | Value |
| --- | ---: |
| positive weight | 0.55 |
| negative weight | 0.275 |
| recency decay | 0.10 |
| override boost | 0.15 |
| semantic merge | disabled |

The actual belief artifact must be schema version 2, complete, provenance-bound,
and audited before use. `BeliefStore` rejects incomplete or structurally
inconsistent artifacts.

## H1 anchor

H1 synthesizes a signed query from the current query and memory. It removes
components aligned with negative concept vectors, then applies a bounded
similarity-based penalty to corpus candidates. Frozen values are projection
strength 0.20, masking threshold 0.25, maximum penalty 0.18, and temperature
0.10.

## F1 proposal, constraint, and fusion

F1 forms a positive-only proposal query. The proposal ranks candidates, then the
asymmetric constraint router may reorder only its top 500 candidates. Negative
evidence is used exclusively in this local constraint stage, with strength 0.275,
posterior temperature 0.05, minimum spread 0.01, and KL budget 0.002.

The router is not allowed to replace the anchor globally. It returns diagnostics
used by `TrustWeightedDualRouteFusion`, which calculates the target-free trust
weight and combines proposal scores with H1 anchor scores. No learned gate,
ground-truth dependent routing, ITM reranking, visual feedback, or
counterfactual module is used.

## Metrics and uncertainty

Ranks are zero-indexed. Recall@10 is the fraction of sessions whose rank at a
given turn is below 10. Hits@10 is cumulative over turns. BRI is the mean
trapezoidal integral of `log(best_rank_so_far + 1)` over dialogue turns; lower is
better. Run comparison uses paired bootstrap confidence intervals for BRI and
Recall@10, exact McNemar tests per turn, and Holm correction across turns.

The frozen configuration is machine-readable in
[`configs/f1_frozen.json`](../configs/f1_frozen.json).
