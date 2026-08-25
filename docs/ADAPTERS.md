# Adapter Guide

NACIR is designed as a retrieval overlay. The core method consumes vectors and signed belief bundles; it does not own image loading, dataset indexing, feature extraction, or retriever training.

## Integration Boundary

An external pipeline is responsible for four things:

1. Build or load corpus vectors with shape `[num_items, embedding_dim]`.
2. Build one query vector per dialogue turn with shape `[embedding_dim]`.
3. Encode belief strings into the same vector space as the query/corpus vectors.
4. Keep the corpus row order stable across H0, H1, F1, and all comparisons.

NACIR then performs memory update, signed projection, negative masking, asymmetric proposal/constraint routing, and trust fusion.

## Required Text Encoder Protocol

Implement the protocol in `src/nacir/interfaces.py`:

```python
from collections.abc import Sequence
import torch

class MyTextEncoder:
    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        # Return a finite, non-zero tensor with shape [len(texts), D].
        # The vectors should live in the same space as corpus/query vectors.
        return vectors
```

The pipeline normalizes corpus and query vectors internally. Adapter outputs should already be normalized when possible; returning non-zero finite vectors is mandatory.

## Current PlugIR Adapter

The current paper setup uses `nacir.adapters.plugir_blip`:

```python
from nacir.adapters import load_blip_text_encoder

encoder = load_blip_text_encoder(device="cuda", allow_download=False)
```

This adapter loads the pinned BLIP retrieval text tower:

```text
model: Salesforce/blip-itm-large-coco
revision: 19502f1e215844f7e48bd48473f86932486d3441
```

The checkpoint name includes `itm`, but the headline F1 method does not perform ITM reranking. This adapter is used to embed text into the same PlugIR-compatible space as the cached query and corpus vectors.

## Adding A New Retriever Adapter

Create a new file under `src/nacir/adapters/`, for example:

```text
src/nacir/adapters/clip_text.py
src/nacir/adapters/my_retriever.py
```

The adapter should expose a class or loader returning an object with `.encode(texts)`.

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

For model-backed encoders, keep the model revision pinned and document whether downloads are allowed. Do not silently change embedding spaces between H0, H1, and F1.

## Adding A New Dataset

A dataset adapter should export the release input files, not modify NACIR core code.

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

`beliefs_complete.json` must follow the schema described in `docs/INPUT_FORMAT.md`.

## Validation Checklist

Before reporting results from a new adapter or dataset:

- query vectors, corpus vectors, and belief vectors have the same dimension;
- every vector is finite and non-zero;
- corpus row order is identical across all runs;
- `session_id` aligns with belief `dialog_id`;
- `target_index` is used only after scores are computed;
- F1 uses `configs/f1_frozen.json` unless the experiment is explicitly labeled as sensitivity;
- comparison is paired using the same sessions in the same order.

## Recommended Handoff Artifacts

For each partner run, archive:

- adapter name, model ID, model revision, and any feature-cache provenance;
- `configs/f1_frozen.json` or the exact sensitivity config;
- H0/H1/F1 `report.json`;
- H0/H1/F1 `ranks.npz`;
- `compare_h0_h1.json`, `compare_h1_f1.json`, and `compare_h0_f1.json`;
- belief audit report and generation provenance.
