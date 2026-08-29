#!/usr/bin/env python3
"""Text-persistent negative baseline.

For each retrieval state, explicitly append all currently remembered negative
concepts to the host query text using natural negation sentences, then re-encode
the resulting text with the same backbone text encoder.

This answers the reviewer question:
    "Why not just keep historical negatives in the text?"

No NACIR query-vector subtraction is used.
"""

from __future__ import annotations

import argparse
import importlib
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from nacir.metrics import compute_metrics


def canon(text: str) -> str:
    return " ".join(str(text).lower().strip().split())


def load_vectors(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    x = obj.get("vectors") if isinstance(obj, dict) else obj
    if not isinstance(x, torch.Tensor) or x.ndim != 2:
        raise ValueError(f"{path}: expected [N,D] tensor")
    return x.float()


def load_sessions(path: Path) -> list[dict]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, list) or len(obj) != 2064:
        raise ValueError("expected 2064 sessions")
    for i, s in enumerate(obj):
        if not isinstance(s.get("query_texts"), list) or len(s["query_texts"]) != 11:
            raise ValueError(
                f"session {i} has no aligned query_texts; Text-Persistent requires them"
            )
    return obj


def load_beliefs(path: Path) -> list[dict]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    dialogs = doc.get("dialogs")
    if not isinstance(dialogs, list) or len(dialogs) != 2064:
        raise ValueError("expected 2064 canonical belief dialogs")
    return dialogs


def load_encoder(module: str, func: str, device: str, allow_download: bool):
    m = importlib.import_module(module)
    return getattr(m, func)(device, allow_download=allow_download)


def build_texts(sessions: list[dict], dialogs: list[dict], max_concepts: int) -> tuple[list[str | None], torch.Tensor, torch.Tensor]:
    texts: list[str | None] = []
    targets = []
    turns = []

    for did, (session, dialog) in enumerate(zip(sessions, dialogs)):
        memory: OrderedDict[str, tuple[str, int]] = OrderedDict()
        for rt in range(11):
            if rt > 0:
                feedback = dialog["turns"][rt - 1]
                for neg in feedback.get("negatives", []) or []:
                    raw = str(neg["attribute"]).strip()
                    key = canon(raw)
                    if key not in memory:
                        memory[key] = (raw, rt)
                    else:
                        # Preserve insertion order but refresh surface form / timestamp.
                        memory[key] = (raw, rt)

                if len(memory) > max_concepts:
                    ordered = sorted(memory, key=lambda k: memory[k][1])
                    for key in ordered[: len(memory) - max_concepts]:
                        del memory[key]

            base = str(session["query_texts"][rt]).strip()
            if memory:
                # Natural-language persistent exclusions.  We deliberately do not
                # manipulate embeddings or use target information.
                excluded = "; ".join(
                    raw for raw, _ in memory.values()
                )
                texts.append(
                    base
                    + " The target image should not match the following "
                    + "excluded visual evidence: "
                    + excluded
                    + "."
                )
            else:
                texts.append(None)  # use exact precomputed host vector
            targets.append(int(session["target_index"]))
            turns.append(rt)

    return texts, torch.tensor(targets), torch.tensor(turns)


@torch.inference_mode()
def encode_queries(
    texts: list[str | None],
    sessions: list[dict],
    encoder,
    *,
    expected_dim: int,
    batch_size: int,
) -> torch.Tensor:
    # Start from exact frozen host query vectors.  Replace only states that have
    # persistent negative text, avoiding re-encoding/no-op drift at empty memory.
    host = torch.stack(
        [
            F.normalize(s["query_vectors"][rt].float().cpu(), dim=0)
            for s in sessions
            for rt in range(11)
        ],
        dim=0,
    )
    idx = [i for i, text in enumerate(texts) if text is not None]
    if not idx:
        return host

    for start in tqdm(range(0, len(idx), batch_size), desc="Encoding Text-Persistent queries"):
        ids = idx[start:start + batch_size]
        batch = [texts[i] for i in ids]
        vec = encoder.encode(batch)
        if vec.ndim != 2 or vec.shape != (len(ids), expected_dim):
            raise ValueError(
                f"encoder dimension mismatch: expected {(len(ids), expected_dim)}, got {tuple(vec.shape)}"
            )
        host[torch.tensor(ids)] = F.normalize(vec.float().cpu(), dim=-1)
    return host


@torch.inference_mode()
def target_ranks(
    queries: torch.Tensor,
    corpus: torch.Tensor,
    targets: torch.Tensor,
    *,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    corpus = F.normalize(corpus.float().to(device), dim=-1)
    indices = torch.arange(corpus.shape[0], device=device)[None, :]
    out = torch.empty(len(queries), dtype=torch.long)
    for start in tqdm(range(0, len(queries), batch_size), desc="Scoring Text-Persistent"):
        stop = min(len(queries), start + batch_size)
        q = F.normalize(queries[start:stop].to(device), dim=-1)
        tgt = targets[start:stop].to(device)
        scores = q @ corpus.T
        ts = scores.gather(1, tgt[:, None])
        ranks = (scores > ts).sum(1) + (
            (scores == ts) & (indices < tgt[:, None])
        ).sum(1)
        out[start:stop] = ranks.cpu()
    return out


def summarize(metrics: dict) -> dict:
    per = [float(x) for x in metrics["per_round_recall"]]
    cum = [float(x) for x in metrics["cumulative_hits"]]
    return {
        "avg_feedback_r10": float(np.mean(per[1:])),
        "final_r10": per[-1],
        "final_cumulative_hits10": cum[-1],
        "bri": float(metrics["bri"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-vectors", type=Path, required=True)
    ap.add_argument("--sessions", type=Path, required=True)
    ap.add_argument("--beliefs", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path("configs/nacir_minus_frozen.json"))
    ap.add_argument("--adapter-module", required=True)
    ap.add_argument("--adapter-func", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--allow-download", action="store_true")
    ap.add_argument("--encode-batch-size", type=int, default=64)
    ap.add_argument("--score-batch-size", type=int, default=128)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    max_concepts = int(cfg["memory"]["max_concepts"])

    corpus = load_vectors(args.corpus_vectors)
    sessions = load_sessions(args.sessions)
    dialogs = load_beliefs(args.beliefs)
    dim = int(corpus.shape[1])

    if any(s["query_vectors"].shape[1] != dim for s in sessions):
        raise ValueError("session/corpus embedding dimension mismatch")

    encoder = load_encoder(
        args.adapter_module, args.adapter_func, args.device, args.allow_download
    )
    probe = encoder.encode(["object"])
    if probe.shape != (1, dim):
        raise ValueError(
            f"adapter/corpus dimension mismatch: {tuple(probe.shape)} vs D={dim}"
        )

    texts, targets, turns = build_texts(sessions, dialogs, max_concepts)
    queries = encode_queries(
        texts,
        sessions,
        encoder,
        expected_dim=dim,
        batch_size=args.encode_batch_size,
    )
    flat_ranks = target_ranks(
        queries, corpus, targets,
        device=args.device, batch_size=args.score_batch_size
    )
    ranks = flat_ranks.numpy().reshape(2064, 11).T.astype(np.int64)
    if not np.array_equal(turns.numpy(), np.tile(np.arange(11), 2064)):
        raise AssertionError("unexpected state ordering")

    metrics = compute_metrics(ranks)
    summary = summarize(metrics)

    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "ranks.npz", ranks=ranks)
    (args.output / "report.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "method": "text_persistent_negatives",
                "template": "The target image does not contain {concept}.",
                "summary": summary,
                "metrics": {
                    "cumulative_hits": [float(x) for x in metrics["cumulative_hits"]],
                    "per_round_recall": [float(x) for x in metrics["per_round_recall"]],
                    "bri": float(metrics["bri"]),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"Text-Persistent: avg feedback R@10={summary['avg_feedback_r10']:.4f} "
        f"final={summary['final_r10']:.4f} "
        f"cum={summary['final_cumulative_hits10']:.4f} "
        f"BRI={summary['bri']:.6f}"
    )
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
