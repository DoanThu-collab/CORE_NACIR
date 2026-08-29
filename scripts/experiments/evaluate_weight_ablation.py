#!/usr/bin/env python3
"""Fast persistent-weight ablation evaluator for frozen NACIR artifacts.

The script leaves core NACIR untouched.  It:
  * pre-encodes every unique raw negative concept once;
  * reconstructs the exact persistent negative memory trajectory;
  * evaluates four principled weighting variants:
      uniform     w=1                 (Rocchio-style negative centroid)
      confidence  w=c
      recency     w=1/(1+rho*age)
      full        w=c/(1+rho*age)     (canonical NACIR)
  * optionally verifies that `full` reproduces a frozen rank matrix.

The only intended experimental variable is the memory weighting rule.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from nacir.metrics import compute_metrics


MODES = ("uniform", "confidence", "recency", "full")


def canon(text: str) -> str:
    return " ".join(str(text).lower().strip().split())


def jsonable(x: Any):
    if torch.is_tensor(x):
        x = x.detach().cpu()
        return x.item() if x.numel() == 1 else x.tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if dataclasses.is_dataclass(x):
        return jsonable(dataclasses.asdict(x))
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    return x


def load_vectors(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    x = obj.get("vectors") if isinstance(obj, dict) else obj
    if not isinstance(x, torch.Tensor) or x.ndim != 2:
        raise ValueError(f"{path}: expected tensor [N,D] or {{'vectors': tensor}}")
    return x.float()


def load_sessions(path: Path) -> list[dict]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, list) or len(obj) != 2064:
        raise ValueError(f"{path}: expected list of 2064 sessions")
    for i, s in enumerate(obj):
        if not isinstance(s, dict):
            raise ValueError(f"session {i}: expected dict")
        q = s.get("query_vectors")
        if not isinstance(q, torch.Tensor) or q.shape[0] != 11 or q.ndim != 2:
            raise ValueError(f"session {i}: query_vectors must be [11,D]")
        if int(s.get("session_id", -1)) != i:
            raise ValueError(f"session id mismatch at {i}")
    return obj


def load_beliefs(path: Path) -> list[dict]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    if (
        doc.get("schema_version") != 2
        or doc.get("status") != "complete"
        or not isinstance(doc.get("dialogs"), list)
        or len(doc["dialogs"]) != 2064
    ):
        raise ValueError("expected complete schema-v2 belief artifact with 2064 dialogs")
    return doc["dialogs"]


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    mem = cfg["memory"]
    required = ["negative_weight", "recency_decay", "max_concepts", "semantic_merge"]
    for k in required:
        if k not in mem:
            raise ValueError(f"config missing memory.{k}")
    if mem["semantic_merge"]:
        raise ValueError("fast ablation evaluator requires frozen semantic_merge=false")
    return cfg


def load_encoder(module: str, func: str, device: str, allow_download: bool):
    m = importlib.import_module(module)
    f = getattr(m, func)
    return f(device, allow_download=allow_download)


@torch.inference_mode()
def encode_text_cache(
    encoder,
    texts: list[str],
    *,
    batch_size: int,
    expected_dim: int,
    cache_path: Path | None,
) -> dict[str, torch.Tensor]:
    if cache_path is not None and cache_path.exists():
        obj = torch.load(cache_path, map_location="cpu", weights_only=False)
        if (
            isinstance(obj, dict)
            and obj.get("texts") == texts
            and isinstance(obj.get("vectors"), torch.Tensor)
            and tuple(obj["vectors"].shape) == (len(texts), expected_dim)
        ):
            print("Loaded concept cache:", cache_path)
            vecs = obj["vectors"].float()
            return {text: vecs[i] for i, text in enumerate(texts)}
        print("Ignoring incompatible cache:", cache_path)

    batches = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Encoding unique negatives"):
        batch = texts[start:start + batch_size]
        v = encoder.encode(batch)
        if not isinstance(v, torch.Tensor) or v.ndim != 2 or v.shape[1] != expected_dim:
            raise ValueError(
                f"text encoder dimension mismatch: expected {expected_dim}, got {tuple(v.shape)}"
            )
        batches.append(F.normalize(v.float().cpu(), dim=-1))
    vecs = torch.cat(batches, dim=0)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"texts": texts, "vectors": vecs}, cache_path)
        print("Saved concept cache:", cache_path)

    return {text: vecs[i] for i, text in enumerate(texts)}


def unique_raw_negatives(dialogs: list[dict]) -> list[str]:
    seen = OrderedDict()
    for d in dialogs:
        for turn in d["turns"]:
            for neg in turn.get("negatives", []) or []:
                raw = str(neg["attribute"]).strip()
                seen.setdefault(raw, None)
    return list(seen)


def memory_weight(mode: str, confidence: float, age: int, rho: float) -> float:
    if mode == "uniform":
        return 1.0
    if mode == "confidence":
        return confidence
    if mode == "recency":
        return 1.0 / (1.0 + rho * age)
    if mode == "full":
        return confidence / (1.0 + rho * age)
    raise ValueError(mode)


def build_queries_for_mode(
    *,
    mode: str,
    sessions: list[dict],
    dialogs: list[dict],
    cache: dict[str, torch.Tensor],
    negative_weight: float,
    rho: float,
    max_concepts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return [22704,D] corrected queries, target indices, retrieval turns."""
    queries = []
    targets = []
    turns_out = []

    for did, (session, dialog) in enumerate(zip(sessions, dialogs)):
        qmat = session["query_vectors"].float().cpu()
        target = int(session["target_index"])
        if len(dialog["turns"]) != 10:
            raise ValueError(f"dialog {did}: expected 10 feedback turns")

        # insertion-ordered canonical memory: key -> [vector, confidence, turn_updated]
        memory: OrderedDict[str, list[Any]] = OrderedDict()

        for retrieval_turn in range(11):
            if retrieval_turn > 0:
                feedback = dialog["turns"][retrieval_turn - 1]
                for neg in feedback.get("negatives", []) or []:
                    raw = str(neg["attribute"]).strip()
                    key = canon(raw)
                    conf = float(neg["confidence"])
                    vec = cache[raw]

                    if key not in memory:
                        memory[key] = [vec, conf, retrieval_turn]
                    else:
                        old = memory[key]
                        old[0] = vec
                        old[1] = max(float(old[1]), conf)
                        old[2] = retrieval_turn

                if len(memory) > max_concepts:
                    # Match ConceptMemory: stable sort by turn_updated, with dict
                    # insertion order breaking ties.
                    ordered = sorted(memory, key=lambda k: memory[k][2])
                    for key in ordered[: len(memory) - max_concepts]:
                        del memory[key]

            q = F.normalize(qmat[retrieval_turn], dim=0)

            if memory:
                vecs = []
                weights = []
                for vec, conf, updated in memory.values():
                    age = max(0, retrieval_turn - int(updated))
                    vecs.append(vec)
                    weights.append(memory_weight(mode, float(conf), age, rho))
                V = torch.stack(vecs, dim=0)
                w = torch.tensor(weights, dtype=V.dtype)
                aggregate = (w[:, None] * V).sum(dim=0)
                if float(aggregate.norm()) > 0.0:
                    q = F.normalize(
                        q - negative_weight * F.normalize(aggregate, dim=0),
                        dim=0,
                    )

            queries.append(q)
            targets.append(target)
            turns_out.append(retrieval_turn)

    return (
        torch.stack(queries, dim=0),
        torch.tensor(targets, dtype=torch.long),
        torch.tensor(turns_out, dtype=torch.long),
    )


@torch.inference_mode()
def target_ranks_batched(
    queries: torch.Tensor,
    corpus: torch.Tensor,
    targets: torch.Tensor,
    *,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    """Compute stable-descending target ranks without sorting all 50k scores.

    Stable descending argsort rank equals:
        # scores strictly greater than target
      + # equal-score corpus items with index < target_index.
    """
    corpus = F.normalize(corpus.float().to(device), dim=-1)
    indices = torch.arange(corpus.shape[0], device=device)[None, :]
    out = torch.empty(len(queries), dtype=torch.long)

    for start in tqdm(range(0, len(queries), batch_size), desc="Scoring"):
        stop = min(len(queries), start + batch_size)
        q = queries[start:stop].to(device)
        q = F.normalize(q.float(), dim=-1)
        tgt = targets[start:stop].to(device)

        scores = q @ corpus.T
        target_scores = scores.gather(1, tgt[:, None])
        greater = (scores > target_scores).sum(dim=1)
        equal_before = (
            (scores == target_scores)
            & (indices < tgt[:, None])
        ).sum(dim=1)
        out[start:stop] = (greater + equal_before).cpu()

    return out


def to_rank_matrix(flat_ranks: torch.Tensor, turns: torch.Tensor) -> np.ndarray:
    # flat state order is session-major: 11 states per session.
    if len(flat_ranks) != 2064 * 11:
        raise ValueError("expected 22704 flat ranks")
    arr = flat_ranks.numpy().reshape(2064, 11).T
    if arr.shape != (11, 2064):
        raise AssertionError(arr.shape)
    expected_turns = np.tile(np.arange(11), 2064)
    if not np.array_equal(turns.numpy(), expected_turns):
        raise AssertionError("unexpected flat turn order")
    return arr.astype(np.int64, copy=False)


def summarize(metrics: dict) -> dict:
    recall = metrics["per_round_recall"]
    cumulative = metrics["cumulative_hits"]
    return {
        "avg_feedback_r10": float(np.mean([float(x) for x in recall[1:]])),
        "final_r10": float(recall[-1]),
        "final_cumulative_hits10": float(cumulative[-1]),
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
    ap.add_argument("--encode-batch-size", type=int, default=128)
    ap.add_argument("--score-batch-size", type=int, default=128)
    ap.add_argument(
        "--modes",
        nargs="+",
        choices=MODES,
        default=list(MODES),
    )
    ap.add_argument("--concept-cache", type=Path)
    ap.add_argument("--verify-full-ranks", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    mem_cfg = cfg["memory"]
    corpus = load_vectors(args.corpus_vectors)
    sessions = load_sessions(args.sessions)
    dialogs = load_beliefs(args.beliefs)

    dim = int(corpus.shape[1])
    if any(int(s["query_vectors"].shape[1]) != dim for s in sessions):
        raise ValueError("session/corpus embedding dimension mismatch")

    raw_texts = unique_raw_negatives(dialogs)
    print(f"Unique raw negative strings: {len(raw_texts)}")
    encoder = load_encoder(
        args.adapter_module, args.adapter_func, args.device, args.allow_download
    )

    # Early adapter dimension guard.
    probe = encoder.encode(["object"])
    if not isinstance(probe, torch.Tensor) or probe.ndim != 2 or probe.shape != (1, dim):
        raise ValueError(
            f"adapter/corpus dimension mismatch: encoder={tuple(probe.shape)}, corpus_dim={dim}"
        )

    cache = encode_text_cache(
        encoder,
        raw_texts,
        batch_size=args.encode_batch_size,
        expected_dim=dim,
        cache_path=args.concept_cache,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "config": cfg,
        "num_unique_raw_negatives": len(raw_texts),
        "variants": {},
    }

    for mode in args.modes:
        print("\n" + "=" * 80)
        print("WEIGHT ABLATION:", mode)
        print("=" * 80)

        q, targets, turns = build_queries_for_mode(
            mode=mode,
            sessions=sessions,
            dialogs=dialogs,
            cache=cache,
            negative_weight=float(mem_cfg["negative_weight"]),
            rho=float(mem_cfg["recency_decay"]),
            max_concepts=int(mem_cfg["max_concepts"]),
        )
        flat = target_ranks_batched(
            q,
            corpus,
            targets,
            device=args.device,
            batch_size=args.score_batch_size,
        )
        ranks = to_rank_matrix(flat, turns)
        metrics = compute_metrics(ranks)
        summary = summarize(metrics)

        np.savez_compressed(args.output / f"{mode}_ranks.npz", ranks=ranks)
        report["variants"][mode] = {
            "summary": summary,
            "metrics": jsonable(metrics),
        }

        print(
            f"{mode}: avg feedback R@10={summary['avg_feedback_r10']:.4f} "
            f"final R@10={summary['final_r10']:.4f} "
            f"cum={summary['final_cumulative_hits10']:.4f} "
            f"BRI={summary['bri']:.6f}"
        )

        if mode == "full" and args.verify_full_ranks is not None:
            frozen = np.load(args.verify_full_ranks, allow_pickle=False)["ranks"].astype(np.int64)
            if frozen.shape != ranks.shape:
                raise ValueError("frozen rank shape mismatch")
            diff = ranks != frozen
            n_diff = int(diff.sum())
            max_abs = int(np.max(np.abs(ranks - frozen))) if ranks.size else 0
            print(
                f"FULL REGRESSION: exact={n_diff == 0} "
                f"num_different={n_diff} max_abs_diff={max_abs}"
            )
            report["full_regression"] = {
                "exact": n_diff == 0,
                "num_different": n_diff,
                "max_abs_diff": max_abs,
            }

    (args.output / "weight_ablation_report.json").write_text(
        json.dumps(jsonable(report), indent=2),
        encoding="utf-8",
    )
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
