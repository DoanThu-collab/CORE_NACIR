#!/usr/bin/env python3
"""Final-round lambda/rho sensitivity using a precomputed negative-concept cache.

This is a robustness analysis, NOT hyperparameter tuning.  It evaluates only
retrieval turn 10 to keep deadline compute small.  The canonical point
(lambda=.275, rho=.10) is not changed or re-selected.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def canon(text: str) -> str:
    return " ".join(str(text).lower().strip().split())


def load_vectors(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    x = obj.get("vectors") if isinstance(obj, dict) else obj
    if not isinstance(x, torch.Tensor) or x.ndim != 2:
        raise ValueError(f"{path}: expected [N,D] tensor")
    return x.float()


def stable_target_ranks(q: torch.Tensor, corpus: torch.Tensor, targets: torch.Tensor, device: str, batch_size: int):
    corpus = F.normalize(corpus.float().to(device), dim=-1)
    indices = torch.arange(corpus.shape[0], device=device)[None, :]
    out = torch.empty(len(q), dtype=torch.long)
    with torch.inference_mode():
        for start in range(0, len(q), batch_size):
            stop = min(len(q), start + batch_size)
            qb = F.normalize(q[start:stop].to(device), dim=-1)
            tb = targets[start:stop].to(device)
            scores = qb @ corpus.T
            ts = scores.gather(1, tb[:, None])
            out[start:stop] = (
                (scores > ts).sum(1)
                + ((scores == ts) & (indices < tb[:, None])).sum(1)
            ).cpu()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-vectors", type=Path, required=True)
    ap.add_argument("--sessions", type=Path, required=True)
    ap.add_argument("--beliefs", type=Path, required=True)
    ap.add_argument("--concept-cache", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--score-batch-size", type=int, default=128)
    ap.add_argument("--lambdas", nargs="+", type=float, default=[0.15, 0.225, 0.275, 0.325, 0.40])
    ap.add_argument("--rhos", nargs="+", type=float, default=[0.0, 0.05, 0.10, 0.20])
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    sessions = torch.load(args.sessions, map_location="cpu", weights_only=False)
    corpus = load_vectors(args.corpus_vectors)
    doc = json.loads(args.beliefs.read_text(encoding="utf-8"))
    dialogs = doc["dialogs"]
    cache_obj = torch.load(args.concept_cache, map_location="cpu", weights_only=False)
    texts = cache_obj["texts"]
    vecs = F.normalize(cache_obj["vectors"].float(), dim=-1)
    cache = {t: vecs[i] for i, t in enumerate(texts)}

    if len(sessions) != 2064 or len(dialogs) != 2064:
        raise ValueError("expected 2064 sessions/dialogs")
    dim = corpus.shape[1]
    if vecs.shape[1] != dim:
        raise ValueError("concept cache/corpus dimension mismatch")

    # Construct final-turn memory once per dialogue.
    final_queries = []
    targets = []
    memories = []
    for did, (session, dialog) in enumerate(zip(sessions, dialogs)):
        memory: OrderedDict[str, list] = OrderedDict()
        for rt in range(1, 11):
            for neg in dialog["turns"][rt - 1].get("negatives", []) or []:
                raw = str(neg["attribute"]).strip()
                key = canon(raw)
                conf = float(neg["confidence"])
                if raw not in cache:
                    raise KeyError(f"concept cache missing {raw!r}")
                if key not in memory:
                    memory[key] = [cache[raw], conf, rt]
                else:
                    memory[key][0] = cache[raw]
                    memory[key][1] = max(float(memory[key][1]), conf)
                    memory[key][2] = rt
        final_queries.append(F.normalize(session["query_vectors"][10].float().cpu(), dim=0))
        targets.append(int(session["target_index"]))
        memories.append(memory)

    base = torch.stack(final_queries)
    targets = torch.tensor(targets, dtype=torch.long)

    rows = []
    for lam in args.lambdas:
        for rho in args.rhos:
            qout = []
            for q, memory in zip(base, memories):
                if not memory:
                    qout.append(q)
                    continue
                vv, ww = [], []
                for vec, conf, updated in memory.values():
                    age = 10 - int(updated)
                    vv.append(vec)
                    ww.append(float(conf) / (1.0 + rho * age))
                V = torch.stack(vv)
                w = torch.tensor(ww, dtype=V.dtype)
                agg = (w[:, None] * V).sum(0)
                if float(agg.norm()) > 0:
                    q = F.normalize(q - lam * F.normalize(agg, dim=0), dim=0)
                qout.append(q)
            Q = torch.stack(qout)
            ranks = stable_target_ranks(
                Q, corpus, targets, args.device, args.score_batch_size
            )
            r10 = 100.0 * float((ranks < 10).float().mean())
            rows.append({"lambda": lam, "rho": rho, "final_r10": r10})
            print(f"lambda={lam:.3f} rho={rho:.3f} final R@10={r10:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["lambda", "rho", "final_r10"])
        w.writeheader()
        w.writerows(rows)
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
