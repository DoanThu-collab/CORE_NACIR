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


def _load_vectors(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")

    if isinstance(obj, torch.Tensor):
        vectors = obj
    elif isinstance(obj, dict):
        for key in (
            "vectors",
            "embeddings",
            "corpus_vectors",
            "features",
        ):
            if key in obj:
                vectors = obj[key]
                break
        else:
            raise ValueError(
                f"Could not find corpus vectors in {path}"
            )
    else:
        raise TypeError(
            f"Unsupported corpus-vector object: {type(obj)}"
        )

    if not isinstance(vectors, torch.Tensor):
        vectors = torch.as_tensor(vectors)

    return vectors.float()


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
    obj: Any = torch.load(path, map_location="cpu")

    sessions = obj["sessions"] if isinstance(obj, dict) and "sessions" in obj else obj

    if not isinstance(sessions, list):
        raise TypeError(
            f"Expected list of sessions, got {type(sessions)}"
        )

    if belief_store is not None:
        for session in sessions:
            for turn in session.turns:
                if turn.beliefs is None:
                    turn.beliefs = belief_store.bundle(
                        session.session_id,
                        turn.turn_index,
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
            metrics,
            f,
            indent=2,
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
        BeliefStore.from_json(args.beliefs)
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
            metrics,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
