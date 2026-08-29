#!/usr/bin/env python3
"""Re-encode an existing standardized conversational host into another retrieval space.

Input sessions must be a list of dicts with:
  session_id, target_index, query_texts[11], query_vectors[11,D_old]

Output preserves session_id/target_index/query_texts and replaces query_vectors
with vectors from the requested text encoder. This lets us test the SAME PlugIR
reconstructed host text in BLIP and CLIP retrieval spaces without regenerating
dialogues or changing the host.
"""

from __future__ import annotations
import argparse, hashlib, importlib, json
from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm import tqdm


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-sessions", type=Path, required=True)
    ap.add_argument("--adapter-module", required=True)
    ap.add_argument("--adapter-func", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--allow-download", action="store_true")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    sessions = torch.load(args.source_sessions, map_location="cpu", weights_only=False)
    if not isinstance(sessions, list) or len(sessions) != 2064:
        raise ValueError("expected standardized list of 2064 sessions")

    for i, s in enumerate(sessions):
        if int(s.get("session_id", -1)) != i:
            raise ValueError(f"session order/id mismatch at {i}")
        qt = s.get("query_texts")
        if not isinstance(qt, list) or len(qt) != 11:
            raise ValueError(f"session {i}: query_texts must have length 11")

    module = importlib.import_module(args.adapter_module)
    encoder = getattr(module, args.adapter_func)(
        args.device, allow_download=args.allow_download
    )

    all_texts = [text for s in sessions for text in s["query_texts"]]
    encoded = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(all_texts), args.batch_size), desc="Encoding host texts"):
            batch = all_texts[start:start+args.batch_size]
            vec = encoder.encode(batch)
            if not isinstance(vec, torch.Tensor) or vec.ndim != 2:
                raise ValueError(f"unexpected encoder output: {type(vec)} {getattr(vec,'shape',None)}")
            encoded.append(F.normalize(vec.float().cpu(), dim=-1))
    encoded = torch.cat(encoded, dim=0)
    if encoded.shape[0] != 2064*11:
        raise AssertionError(encoded.shape)
    dim = int(encoded.shape[1])
    encoded = encoded.reshape(2064, 11, dim)

    out = []
    for i, s in enumerate(sessions):
        out.append({
            "session_id": int(s["session_id"]),
            "target_index": int(s["target_index"]),
            "query_texts": list(s["query_texts"]),
            "query_vectors": encoded[i],
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.output)

    prov = {
        "status": "complete",
        "host_text_source": str(args.source_sessions),
        "host_text_source_sha256": sha256(args.source_sessions),
        "adapter_module": args.adapter_module,
        "adapter_func": args.adapter_func,
        "num_sessions": len(out),
        "rounds": 11,
        "embedding_dim": dim,
        "same_host_text": True,
        "description": (
            "Same PlugIR reconstructed query texts re-encoded into a different "
            "retrieval embedding space; no dialogue regeneration."
        ),
    }
    prov_path = args.output.with_suffix(".provenance.json")
    prov_path.write_text(json.dumps(prov, indent=2), encoding="utf-8")
    print("Saved:", args.output)
    print("Provenance:", prov_path)
    print("shape per session:", tuple(out[0]["query_vectors"].shape))


if __name__ == "__main__":
    main()
