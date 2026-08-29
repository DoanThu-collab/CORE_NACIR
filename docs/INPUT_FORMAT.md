# Input contract

The public evaluator uses precomputed vectors. This keeps the paper retrieval logic separate from a particular image encoder or dataset loader.

## Corpus vectors

`--corpus-vectors` is a `torch.save` file containing either a float tensor of shape `[num_items, embedding_dim]`, or a dictionary with a `vectors` tensor of that shape. Rows must be finite and non-zero. The evaluator normalizes them.

## Sessions

`--sessions` is a `torch.save` file containing a non-empty list of dictionaries. Each dictionary has:

```python
{
    "session_id": 0,
    "target_index": 123,
    "query_vectors": torch.Tensor,  # [num_turns, embedding_dim]
    "query_texts": ["optional text", ...],
}
```

`session_id` must agree with the belief artifact dialog identifier. Query vectors must be finite and non-zero. `target_index` is a zero-indexed corpus row and is used only to compute ranks after scoring.

## Beliefs

`current` and `persistent` require `--beliefs`; `h0` may omit it. The canonical artifact uses schema version 2 and must be complete and provenance-bound according to the release belief loader.

The retrieval-turn convention is deliberate: retrieval turn zero has no prior feedback beliefs; retrieval turn `t > 0` reads the generated belief record associated with feedback turn `t - 1`. This prevents the answer to the current retrieval turn from leaking into its own query state.

Only negative beliefs are active in the canonical NACIR memory update. Positive beliefs may exist in the artifact for diagnostics or lineage but are ignored by the canonical retrieval intervention.

## Paired-run provenance

Paper-facing `ranks.npz` archives include:

- `ranks`;
- `session_ids`;
- `target_indices`;
- `pairing_fingerprint`;
- `evaluation_fingerprint`;
- `provenance_status`;
- `metadata_json`.

Paired statistics are valid only when the two archives have the same pairing fingerprint and exact session/target ordering. The pairing identity binds the session file, corpus vectors, embedding dimension, and aligned session/target lists.
