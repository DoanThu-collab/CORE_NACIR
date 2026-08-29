# NACIR Reproducibility Record

## Canonical main evaluation

The canonical main-evaluation freeze is:

    paper-main-eval-v1

This tag identifies the frozen six-condition main evaluation used for exact-rank regression.

| Retrieval space | H0 | Current-turn | Persistent NACIR |
|---|---|---|---|
| BLIP | exact match | exact match | exact match |
| OpenAI CLIP ViT-L/14 | exact match | exact match | exact match |

For all six frozen main runs:

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

The canonical retrieval intervention uses negative beliefs only.

## Canonical belief artifact

The main experiments use the frozen Llama-3.1-8B belief artifact:

    llama3_1_8b_v9_final_20260824.json

The artifact itself is not redistributed in this repository. Pass its local path explicitly with `--beliefs` or the `BELIEFS` environment variable used by the shell runners.

## Provenance and paired comparisons

New evaluator outputs write strict provenance into `ranks.npz`:

- `session_ids`
- `target_indices`
- `pairing_fingerprint`
- `evaluation_fingerprint`
- `provenance_status`
- `metadata_json`

The pairing fingerprint binds the session artifact, corpus vectors, embedding dimension, and aligned session/target ordering. Paired statistics are rejected unless pairing provenance and ordering agree exactly. This prevents accidental BLIP-vs-CLIP or otherwise misaligned paired comparisons.

Legacy frozen rank archives can be upgraded with `scripts/upgrade_rank_archive.py`; upgraded archives are marked as rehydrated from explicitly declared inputs rather than as historically emitted provenance.

## Paper-facing analyses

The retained analysis utilities are intentionally limited to paper-facing or release-audit tasks:

- `scripts/analysis/analyze_belief_state.py`
- `scripts/analysis/clean_persistence_challenge.py`
- `scripts/analysis/analyze_cross_host_matrix.py`
- `scripts/analysis/audit_dataset_lineage.py`
- `scripts/analysis/audit_chatir_canonical.py`
- `scripts/analysis/summarize_host_boundary.py`

Auxiliary experiment runners are under `scripts/experiments/`. Generated vectors, rank archives, belief artifacts, and external datasets remain outside Git and are passed through explicit paths or environment variables.

## Evaluator safety

For Current and Persistent conditions, the unified evaluator checks that the negative-concept text encoder and corpus embeddings have matching dimensions before evaluation. H0 does not invoke the belief encoder.
