#!/usr/bin/env python3
"""Run one frozen H0, H1, or F1 experiment from precomputed tensors.

The input session file is a ``torch.save`` list of dictionaries. Each dictionary
contains ``session_id``, ``target_index``, and ``query_vectors`` with shape
``[rounds, embedding_dim]``. An optional ``query_texts`` list is retained only for
human-readable trace output. Corpus vectors are a tensor or a dictionary with a
``vectors`` tensor. Beliefs are read from a complete NACIR schema-version-2 file.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from nacir.beliefs import BeliefStore
from nacir.config import NACIRMinusConfig, MemoryConfig
from nacir.evaluation import evaluate_session, rank_matrix
from nacir.metrics import compute_metrics
from nacir.pipeline_current_turn import NACIRCurrentTurnPipeline
from nacir.schema import DialogTurn, RetrievalSession


class _UnusedEncoder:
    def encode(self, texts):
        raise RuntimeError("H0 must not request belief embeddings")


def _load_vectors(path: Path) -> torch.Tensor:
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    vectors = loaded.get("vectors") if isinstance(loaded, dict) else loaded
    if not isinstance(vectors, torch.Tensor):
        raise ValueError("vector file must be a tensor or {'vectors': tensor}")
    return vectors


def _sessions(path: Path, beliefs: BeliefStore | None) -> list[RetrievalSession]:
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(loaded, list) or not loaded:
        raise ValueError("session file must be a non-empty list")
    sessions: list[RetrievalSession] = []
    for raw in loaded:
        if not isinstance(raw, dict):
            raise ValueError("every session must be a dictionary")
        session_id = raw.get("session_id")
        target_index = raw.get("target_index")
        query_vectors = raw.get("query_vectors")
        query_texts = raw.get("query_texts")
        if not isinstance(session_id, int) or not isinstance(target_index, int):
            raise ValueError("session_id and target_index must be integers")
        if not isinstance(query_vectors, torch.Tensor) or query_vectors.ndim != 2:
            raise ValueError("query_vectors must have shape [rounds, embedding_dim]")
        if query_texts is None:
            query_texts = ["precomputed query"] * query_vectors.shape[0]
        if not isinstance(query_texts, list) or len(query_texts) != query_vectors.shape[0]:
            raise ValueError("query_texts must align with query_vectors")
        turns = [
            DialogTurn(
                turn_index=index,
                query_text=str(query_texts[index]),
                query_vector=query_vectors[index],
                beliefs=beliefs.bundle(session_id, index) if beliefs else None,
            )
            for index in range(query_vectors.shape[0])
        ]
        sessions.append(RetrievalSession(session_id, turns, target_index))
    return sessions


def _json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["h0", "nacir"], required=True)
    parser.add_argument("--corpus-vectors", type=Path, required=True)
    parser.add_argument("--sessions", type=Path, required=True)
    parser.add_argument("--beliefs", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/nacir_minus_frozen.json"))
    parser.add_argument("--adapter-module", default="nacir.adapters.plugir_blip", help="Python module path of the text encoder adapter")
    parser.add_argument("--adapter-func", default="load_blip_text_encoder", help="Function name in the adapter module to instantiate the text encoder")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()

    if args.mode != "h0" and args.beliefs is None:
        raise ValueError("NACIR mode requires --beliefs")
    store = BeliefStore.from_path(args.beliefs) if args.beliefs else None
    if args.mode != "h0":
        import importlib
        adapter_module = importlib.import_module(args.adapter_module)
        adapter_func = getattr(adapter_module, args.adapter_func)
        encoder = adapter_func(args.device, allow_download=args.allow_download)
    else:
        encoder = _UnusedEncoder()
        
    if args.config.exists():
        with args.config.open("r", encoding="utf-8") as f:
            config_data = json.load(f)
        mem_data = config_data.get("memory", {})
        config = NACIRMinusConfig(
            memory=MemoryConfig(**mem_data),
            top_k=config_data.get("top_k", 1000)
        )
    else:
        config = NACIRMinusConfig()

    pipeline = NACIRCurrentTurnPipeline(
        config=config,
        corpus_vectors=_load_vectors(args.corpus_vectors),
        text_encoder=encoder,
        device=args.device,
    )
    sessions = _sessions(args.sessions, store)

    if args.mode == "nacir":
        outputs = [
            pipeline.run_session(session)
            for session in tqdm(sessions, desc="Evaluating Sessions")
        ]
    else:
        outputs = [
            evaluate_session(pipeline, session, args.mode)
            for session in tqdm(sessions, desc="Evaluating Sessions")
        ]
    ranks = rank_matrix(outputs)
    metrics = compute_metrics(ranks)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "ranks.npz", ranks=np.asarray(ranks, dtype=np.int64))
    with (args.output / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(
            _json_safe(
                {
                    "status": "complete",
                    "run_name": args.mode.upper(),
                    "method": "raw_query" if args.mode == "h0" else args.mode,
                    "metrics": metrics,
                    "method_config": dataclasses.asdict(config),
                    "config_path": str(args.config),
                    "beliefs": str(args.beliefs) if args.beliefs else None,
                    "num_sessions": len(outputs),
                }
            ),
            handle,
            indent=2,
        )
    with (args.output / "turn_traces.jsonl").open("w", encoding="utf-8") as handle:
        for output in outputs:
            for trace in output.turns:
                handle.write(json.dumps(_json_safe({"session_id": output.session_id, **trace.__dict__})) + "\n")


if __name__ == "__main__":
    main()
