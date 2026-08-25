# Input contract

The public evaluator uses precomputed vectors. This keeps the paper retrieval
logic separate from a particular image encoder or dataset loader.

## Corpus vectors

`--corpus-vectors` is a `torch.save` file containing either a float tensor of
shape `[num_items, embedding_dim]`, or a dictionary with a `vectors` tensor of
that shape. Rows must be finite and non-zero. The evaluator normalizes them.

## Sessions

`--sessions` is a `torch.save` file containing a non-empty list of dictionaries.
Each dictionary has:

```python
{
    "session_id": 0,
    "target_index": 123,
    "query_vectors": torch.Tensor,  # [num_turns, embedding_dim]
    "query_texts": ["optional text for trace output", ...],
}
```

`session_id` must agree with the belief artifact dialog identifier. Query vectors
must be finite and non-zero. `target_index` is a zero-indexed corpus row and is
used only to compute ranks after scoring.

## Beliefs

H1 and F1 require `--beliefs`; H0 must omit it. The JSON artifact must contain:

- `schema_version: 2`;
- `status: "complete"`;
- a `provenance` mapping;
- a `quality` mapping with `status: "passed"`;
- contiguous `dialog_id` and `turn` numbering.

The expected turn convention is deliberate: retrieval turn zero has no prior
beliefs; retrieval turn `t > 0` reads the generated belief record at index
`t - 1`. This prevents the current answer from leaking into retrieval at the
same turn.
