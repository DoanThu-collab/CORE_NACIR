# NACIR method

## Scope

NACIR is a training-free adapter for conversational image retrieval. The canonical implementation preserves **negative visual evidence** across dialogue turns and applies that persistent exclusion state in the same embedding space used by the host retriever.

The main evaluation distinguishes three conditions:

| Condition | Negative evidence used |
| --- | --- |
| `h0` | none; host query only |
| `current` | current feedback turn only |
| `persistent` | persistent negative memory across turns |

The target index is used only after retrieval scores have been computed, to obtain the evaluation rank.

## Persistent negative memory

Each retained negative concept stores text, its embedding vector, confidence, and the turn at which it was last updated. For concept `j` at retrieval turn `t`, the frozen weight is

\[
w_{j,t}=\frac{c_j}{1+\rho(t-\tau_j)}.
\]

The persistent negative memory vector is

\[
m_t^- = \sum_j w_{j,t} v_j.
\]

Repeated canonical negative concepts refresh their vector, confidence, and update turn. The frozen release disables semantic merging and stores at most 50 concepts.

## Query correction

Let `q_t` be the host query vector and `\bar q_t=q_t/\lVert q_t\rVert_2`. NACIR forms

\[
q_t^- = \operatorname{norm}\!\left(\bar q_t - \lambda\,\operatorname{norm}(m_t^-)\right).
\]

The frozen configuration uses:

| Parameter | Value |
| --- | ---: |
| `lambda` | 0.275 |
| `rho` | 0.10 |
| maximum memory concepts | 50 |
| semantic merge | disabled |

Only negative beliefs are active in the canonical method. No projection module, masking module, dual-route fusion, learned gate, ITM reranking, or counterfactual component is part of the frozen NACIR evaluator.

## Current-turn control

The `current` condition uses exactly the same query-correction form but constructs its negative state from the current feedback turn only. It therefore isolates explicit current-turn negative awareness from the additional effect of persistence.

When the current feedback contains no extracted negative belief, `current` reduces to the host baseline (up to the evaluator's normalization convention). This property underlies the clean persistence challenge used in the paper analysis.

## Retrieval spaces

The adapter requires a text encoder whose output dimension and semantic space match the precomputed corpus embeddings. The release includes adapters for the BLIP-based host space used by the main ChatIR-style evaluation and for OpenAI CLIP ViT-L/14. The evaluator validates embedding dimensionality before NACIR scoring.

## Frozen configuration

The machine-readable canonical configuration is:

`configs/nacir_minus_frozen.json`
