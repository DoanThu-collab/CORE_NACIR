#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from nacir.beliefs import BeliefStore
from nacir.config import MemoryConfig, NACIRMinusConfig
from nacir.evaluation import evaluate_session, rank_matrix
from nacir.metrics import compute_metrics
from nacir.pipeline import NACIRMinusPipeline
from nacir.pipeline_current_turn import NACIRCurrentTurnPipeline

def _to_jsonable(obj):
    """Recursively convert common non-JSON objects to JSON-safe Python types."""
    if torch.is_tensor(obj):
        obj = obj.detach().cpu()
        if obj.numel() == 1:
            return obj.item()
        return obj.tolist()

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]

    return obj


def _load_vectors(path: Path) -> torch.Tensor:
    loaded = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    vectors = (
        loaded.get("vectors")
        if isinstance(loaded, dict)
        else loaded
    )

    if not isinstance(vectors, torch.Tensor):
        raise ValueError(
            "vector file must be a tensor "
            "or {'vectors': tensor}"
        )

    return vectors


def _load_config(path: Path | None) -> NACIRMinusConfig:
    if path is None or not path.exists():
        return NACIRMinusConfig()

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    return NACIRMinusConfig(
        memory=MemoryConfig(**raw.get("memory", {})),
        top_k=raw.get("top_k", 1000),
    )


def _load_encoder(
    module_name: str,
    function_name: str,
    device: str,
    allow_download: bool,
):
    module = importlib.import_module(module_name)
    factory = getattr(module, function_name)

    try:
        return factory(
            device,
            allow_download=allow_download,
        )
    except TypeError:
        return factory(device)


def _load_sessions(
    path: Path,
    belief_store: BeliefStore | None,
):
    loaded = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(loaded, list) or not loaded:
        raise ValueError(
            "session file must be a non-empty list"
        )

    from nacir.schema import DialogTurn, RetrievalSession

    sessions: list[RetrievalSession] = []

    for raw in loaded:
        if not isinstance(raw, dict):
            raise ValueError(
                "every session must be a dictionary"
            )

        session_id = raw.get("session_id")
        target_index = raw.get("target_index")
        query_vectors = raw.get("query_vectors")
        query_texts = raw.get("query_texts")

        if not isinstance(session_id, int) or not isinstance(target_index, int):
            raise ValueError(
                "session_id and target_index must be integers"
            )

        if (
            not isinstance(query_vectors, torch.Tensor)
            or query_vectors.ndim != 2
        ):
            raise ValueError(
                "query_vectors must have shape "
                "[rounds, embedding_dim]"
            )

        if query_texts is None:
            query_texts = [
                "precomputed query"
            ] * query_vectors.shape[0]

        if (
            not isinstance(query_texts, list)
            or len(query_texts) != query_vectors.shape[0]
        ):
            raise ValueError(
                "query_texts must align with query_vectors"
            )

        turns = [
            DialogTurn(
                turn_index=index,
                query_text=str(query_texts[index]),
                query_vector=query_vectors[index],
                beliefs=(
                    belief_store.bundle(
                        session_id,
                        index,
                    )
                    if belief_store
                    else None
                ),
            )
            for index in range(
                query_vectors.shape[0]
            )
        ]

        sessions.append(
            RetrievalSession(
                session_id,
                turns,
                target_index,
            )
        )

    return sessions


def _save(
    output_dir: Path,
    ranks: np.ndarray,
    metrics: dict[str, Any],
) -> None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        output_dir / "ranks.npz",
        ranks=ranks,
    )

    with (output_dir / "metrics.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            _to_jsonable(metrics),
            f,
            indent=2,
        )


def _validate_encoder_dimension(
    encoder: Any,
    corpus_vectors: torch.Tensor,
) -> None:
    """Fail early when the belief encoder does not match retrieval space."""
    probe = encoder.encode(["object"])

    if not torch.is_tensor(probe):
        probe = torch.as_tensor(probe)

    if probe.ndim == 1:
        encoder_dim = int(probe.shape[0])
    elif probe.ndim == 2 and probe.shape[0] == 1:
        encoder_dim = int(probe.shape[-1])
    else:
        raise ValueError(
            "Unexpected text-encoder output shape: "
            f"{tuple(probe.shape)}"
        )

    if corpus_vectors.ndim != 2:
        raise ValueError(
            "Expected corpus vectors with shape [N, D], "
            f"got {tuple(corpus_vectors.shape)}"
        )

    corpus_dim = int(corpus_vectors.shape[-1])

    if encoder_dim != corpus_dim:
        raise ValueError(
            "Text encoder / retrieval-space dimension mismatch: "
            f"encoder={encoder_dim}, corpus={corpus_dim}. "
            "Select the backbone-specific --adapter-module "
            "and --adapter-func."
        )



def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate NACIR retrieval using precomputed "
            "query and corpus embeddings."
        )
    )

    parser.add_argument(
        "--method",
        required=True,
        choices=[
            "h0",
            "current",
            "persistent",
        ],
    )
    parser.add_argument(
        "--corpus-vectors",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--sessions",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--beliefs",
        type=Path,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/nacir_minus_frozen.json"
        ),
    )
    parser.add_argument(
        "--adapter-module",
        default="nacir.adapters.plugir_blip",
    )
    parser.add_argument(
        "--adapter-func",
        default="load_blip_text_encoder",
    )
    parser.add_argument(
        "--device",
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
    )

    args = parser.parse_args()

    config = _load_config(args.config)

    belief_store = (
        BeliefStore.from_path(args.beliefs)
        if args.beliefs
        else None
    )

    sessions = _load_sessions(
        args.sessions,
        belief_store,
    )

    # H0 does not require belief encoding.
    if args.method == "h0":
        class _UnusedEncoder:
            def encode(self, texts):
                raise AssertionError(
                    "H0 must not encode beliefs."
                )

        encoder = _UnusedEncoder()
    else:
        encoder = _load_encoder(
            args.adapter_module,
            args.adapter_func,
            args.device,
            args.allow_download,
        )

    corpus_vectors = _load_vectors(
        args.corpus_vectors
    )

    if args.method != "h0":
        _validate_encoder_dimension(
            encoder,
            corpus_vectors,
        )

    if args.method == "current":
        pipeline = NACIRCurrentTurnPipeline(
            config=config,
            corpus_vectors=corpus_vectors,
            text_encoder=encoder,
            device=args.device,
        )
        eval_mode = "nacir"
    else:
        pipeline = NACIRMinusPipeline(
            config=config,
            corpus_vectors=corpus_vectors,
            text_encoder=encoder,
            device=args.device,
        )

        eval_mode = (
            "h0"
            if args.method == "h0"
            else "nacir"
        )

    outputs = [
        evaluate_session(
            pipeline,
            session,
            eval_mode,
        )
        for session in tqdm(
            sessions,
            desc=f"Evaluating {args.method}",
        )
    ]

    ranks = rank_matrix(outputs)
    metrics = compute_metrics(ranks)

    _save(
        args.output,
        ranks,
        metrics,
    )

    print(
        json.dumps(
            _to_jsonable(metrics),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
