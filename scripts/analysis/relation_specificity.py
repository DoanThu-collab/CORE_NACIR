import json
import random
import gc
import inspect
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

from nacir.adapters.plugir_blip import load_blip_text_encoder
from nacir.adapters.openai_clip_vitl14 import load_clip_text_encoder


ART = Path(
    "artifacts_final/typed_nacir/"
    "chatir_structured_negative_final_v1_1.json"
)

OUTDIR = Path(
    "artifacts_final/typed_nacir/"
    "relation_geometry_v1"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

SEED = 20260829
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


items = json.load(open(ART, encoding="utf-8"))["items"]

rels = [
    x for x in items
    if x["actionable_negative"]
    and x["typing"]["type"] == "RELATION"
]

assert len(rels) == 141


def norm(v):
    return v / v.norm().clamp_min(1e-12)


def residual(e_rel, e_s, e_o):
    e_rel = norm(e_rel)
    e_s = norm(e_s)
    e_o = norm(e_o)

    A = torch.stack([e_s, e_o], dim=1)

    proj = A @ (torch.linalg.pinv(A) @ e_rel)

    r = e_rel - proj

    return norm(r)


def cos(a, b):
    return torch.dot(norm(a), norm(b)).item()


def summarize(name, vals):
    x = np.asarray(vals, dtype=float)

    print(f"\n{name}")
    print("-" * 72)
    print("N      :", len(x))
    print("mean   :", x.mean())
    print("std    :", x.std())
    print("median :", np.median(x))
    print("p05    :", np.quantile(x, .05))
    print("p25    :", np.quantile(x, .25))
    print("p75    :", np.quantile(x, .75))
    print("p95    :", np.quantile(x, .95))


def loader_call(loader):
    sig = inspect.signature(loader)

    kwargs = {}

    if "device" in sig.parameters:
        kwargs["device"] = DEVICE

    if "allow_download" in sig.parameters:
        kwargs["allow_download"] = False

    return loader(**kwargs)


def make_encode(enc):
    cache = {}

    def f(text):
        text = str(text).strip()

        if text not in cache:
            with torch.inference_mode():
                v = enc.encode([text])[0]

            cache[text] = (
                v.detach().float().cpu()
            )

        return cache[text]

    return f


def run(name, loader):
    print("\n" + "=" * 72)
    print(name)
    print("=" * 72)

    enc = loader_call(loader)
    encode = make_encode(enc)

    rng = random.Random(SEED)

    predicates = [
        str(x["typing"].get("predicate") or "").strip()
        for x in rels
    ]

    valid_predicates = [
        p for p in predicates if p
    ]

    print("relations        :", len(rels))
    print("nonempty predicate:", len(valid_predicates))

    own_pred = []
    shuffled_pred = []
    margins = []

    own_res_vs_shuffled_res = []

    rows = []

    # Fixed permutation with no self-match where possible
    order = list(range(len(rels)))

    shuffled = order.copy()

    while True:
        rng.shuffle(shuffled)

        if all(
            i != j
            for i, j in zip(order, shuffled)
        ):
            break

    for i, x in enumerate(rels):
        t = x["typing"]

        s = str(t["subject"]).strip()
        p = str(t.get("predicate") or "").strip()
        o = str(t["object"]).strip()

        if not p:
            continue

        rel_text = f"{s} {p} {o}"

        e_rel = encode(rel_text)
        e_s = encode(s)
        e_o = encode(o)

        r = residual(
            e_rel,
            e_s,
            e_o,
        )

        # Shuffled predicate from another relation
        j = shuffled[i]

        p_shuf = str(
            rels[j]["typing"].get("predicate") or ""
        ).strip()

        # If shuffled relation has empty predicate,
        # deterministically find next usable one.
        if not p_shuf:
            for off in range(1, len(rels)):
                jj = (j + off) % len(rels)
                candidate = str(
                    rels[jj]["typing"].get("predicate") or ""
                ).strip()

                if candidate:
                    p_shuf = candidate
                    break

        e_p = encode(p)
        e_p_shuf = encode(p_shuf)

        own = cos(r, e_p)
        shuf = cos(r, e_p_shuf)

        own_pred.append(own)
        shuffled_pred.append(shuf)
        margins.append(own - shuf)

        # Build same-anchor relation with wrong predicate
        rel_shuf_text = f"{s} {p_shuf} {o}"

        r_shuf = residual(
            encode(rel_shuf_text),
            e_s,
            e_o,
        )

        residual_sim = cos(
            r,
            r_shuf,
        )

        own_res_vs_shuffled_res.append(
            residual_sim
        )

        rows.append({
            "dialog_id": x["dialog_id"],
            "turn": x["turn"],
            "negative_index": x.get(
                "negative_index", 0
            ),
            "subject": s,
            "predicate": p,
            "object": o,
            "shuffled_predicate": p_shuf,
            "cos_res_own_predicate": own,
            "cos_res_shuffled_predicate": shuf,
            "margin": own - shuf,
            "cos_residual_vs_shuffled_residual": (
                residual_sim
            ),
        })

    summarize(
        "cos(residual, own predicate)",
        own_pred,
    )

    summarize(
        "cos(residual, shuffled predicate)",
        shuffled_pred,
    )

    summarize(
        "own-minus-shuffled predicate margin",
        margins,
    )

    summarize(
        "cos(residual, same anchors + shuffled predicate residual)",
        own_res_vs_shuffled_res,
    )

    margins_np = np.asarray(margins)

    print("\nPREFERENCE")
    print(
        "own > shuffled:",
        int((margins_np > 0).sum()),
        "/",
        len(margins_np),
        "=",
        float((margins_np > 0).mean()),
    )

    print(
        "margin > .05:",
        int((margins_np > .05).sum()),
    )

    print(
        "margin > .10:",
        int((margins_np > .10).sum()),
    )

    out = {
        "backbone": name,
        "num_relations": len(rows),
        "seed": SEED,
        "retrieval_results_used": False,
        "rows": rows,
    }

    path = OUTDIR / (
        name.lower()
        .replace("/", "_")
        .replace(" ", "_")
        + "_relation_specificity.json"
    )

    path.write_text(
        json.dumps(
            out,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("saved:", path)

    del encode
    del enc

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


run(
    "BLIP_ITM_LARGE_COCO",
    load_blip_text_encoder,
)

run(
    "OPENAI_CLIP_VIT_L14",
    load_clip_text_encoder,
)

print("\nPASS")
