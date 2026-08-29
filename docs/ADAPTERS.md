# Adapter Guide

NACIR is a retrieval overlay. The core method consumes query/corpus vectors and negative belief bundles; it does not own image loading, dataset indexing, feature extraction, or retriever training.

## Integration Boundary

An external pipeline is responsible for four things:

1. Build or load corpus vectors with shape `[num_items, embedding_dim]`.
2. Build one query vector per dialogue turn with shape `[embedding_dim]`.
3. Encode negative belief strings into the same vector space as the query/corpus vectors.
4. Keep corpus row order and session/target ordering stable across H0, Current, Persistent, and every paired comparison.

Persistent NACIR updates a negative-only memory, forms the weighted historical exclusion vector, subtracts it from the normalized host query, and ranks with the corrected query. No projection module, masking module, dual-route fusion, learned gate, ITM reranking, or counterfactual component is active in the canonical evaluator.

## Required Text Encoder Protocol

Implement the protocol in `src/nacir/interfaces.py`:

```python
from collections.abc import Sequence
import torch

class MyTextEncoder:
    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        # Return a finite, non-zero tensor with shape [len(texts), D].
        # The vectors must live in the same space as corpus/query vectors.
        return vectors
```

The pipeline normalizes corpus and query vectors internally. Adapter outputs should already be normalized when possible; returning non-zero finite vectors is mandatory.

## BLIP Adapter

The paper BLIP setup uses `nacir.adapters.plugir_blip`:

```python
from nacir.adapters import load_blip_text_encoder

encoder = load_blip_text_encoder(device="cuda", allow_download=False)
```

The adapter loads the pinned BLIP retrieval text tower:

```text
model: Salesforce/blip-itm-large-coco
revision: 19502f1e215844f7e48bd48473f86932486d3441
```

The checkpoint name includes `itm`, but NACIR does not perform ITM reranking. This adapter is used only to embed negative concept text into the same retrieval space as the cached query and corpus vectors.

## OpenAI CLIP ViT-L/14 Adapter

The CLIP evaluation uses `nacir.adapters.openai_clip_vitl14` and requires the OpenAI CLIP package in addition to the base dependencies.

```bash
pip install git+https://github.com/openai/CLIP.git
```

Use:

```bash
--adapter-module nacir.adapters.openai_clip_vitl14 \
--adapter-func load_clip_text_encoder
```

## Adding A New Retriever Adapter

Create a file under `src/nacir/adapters/` and expose a class or loader returning an object with `.encode(texts)`.

Minimal example:

```python
import torch
import torch.nn.functional as F

class PrecomputedVocabularyEncoder:
    def __init__(self, lookup: dict[str, torch.Tensor]) -> None:
        self.lookup = lookup

    def encode(self, texts):
        vectors = []
        for text in texts:
            if text not in self.lookup:
                raise KeyError(f"missing text vector: {text}")
            vectors.append(self.lookup[text].float())
        return F.normalize(torch.stack(vectors), dim=-1)
```

For model-backed encoders, pin the model revision and document whether downloads are allowed. Never silently change embedding spaces between paired runs.

## Adding A New Dataset

A dataset adapter should export the release input files rather than modifying NACIR core code.

`corpus_vectors.pt`:

```python
torch.save({"vectors": corpus_vectors}, "corpus_vectors.pt")
```

`sessions.pt`:

```python
torch.save([
    {
        "session_id": 0,
        "target_index": 123,
        "query_vectors": query_vectors,  # [num_turns, D]
        "query_texts": query_texts,
    }
], "sessions.pt")
```

The belief artifact must follow `docs/INPUT_FORMAT.md`.

## Validation Checklist

Before reporting results from a new adapter or dataset:

- query vectors, corpus vectors, and belief vectors have the same dimension;
- every vector is finite and non-zero;
- corpus row order is identical across paired runs;
- `session_id` aligns with belief `dialog_id`;
- `target_index` is used only after scores are computed;
- Persistent uses `configs/nacir_minus_frozen.json` unless explicitly labeled as a sensitivity analysis;
- paired statistics are computed only after `pairing_fingerprint`, `session_ids`, and `target_indices` match exactly.

## Recommended Handoff Artifacts

For each paper-facing run, archive:

- adapter name, model ID/revision, and feature-cache provenance;
- exact config file;
- `ranks.npz` with pairing/evaluation provenance;
- aggregate metrics/report JSON;
- strict paired-comparison output;
- belief audit/generation provenance.
