import json
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

ART = Path(
    "artifacts_final/typed_nacir/"
    "chatir_structured_negative_final_v1_1.json"
)

items = json.load(open(ART, encoding="utf-8"))["items"]

rels = [
    x for x in items
    if x["actionable_negative"]
    and x["typing"]["type"] == "RELATION"
]

assert len(rels) == 141

print("ACTIONABLE RELATIONS:", len(rels))


def normalize(x, eps=1e-12):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def relation_residual(e_rel, e_s, e_o):
    """
    Project e_rel onto span{e_s, e_o}, then remove projection.
    All inputs are 1D tensors [D].
    """

    e_rel = normalize(e_rel.unsqueeze(0))[0]
    e_s = normalize(e_s.unsqueeze(0))[0]
    e_o = normalize(e_o.unsqueeze(0))[0]

    # D x 2
    A = torch.stack([e_s, e_o], dim=1)

    # coefficients using pseudoinverse for numerical stability
    coeff = torch.linalg.pinv(A) @ e_rel
    proj = A @ coeff

    residual = e_rel - proj

    return e_rel, e_s, e_o, proj, residual


def summarize(name, values):
    x = np.asarray(values, dtype=float)

    print(f"\n{name}")
    print("-" * 72)
    print("N      :", len(x))
    print("mean   :", x.mean())
    print("std    :", x.std())
    print("median :", np.median(x))
    print("p05    :", np.quantile(x, 0.05))
    print("p25    :", np.quantile(x, 0.25))
    print("p75    :", np.quantile(x, 0.75))
    print("p95    :", np.quantile(x, 0.95))
    print("min    :", x.min())
    print("max    :", x.max())


def run_audit(name, encode_text):
    metrics = defaultdict(list)
    rows = []

    for x in rels:
        t = x["typing"]

        s = str(t["subject"]).strip()
        p = str(t.get("predicate") or "").strip()
        o = str(t["object"]).strip()

        assert s
        assert o

        # Use the actual rejected relational proposition.
        relation_text = " ".join(
            z for z in [s, p, o] if z
        )

        e_rel = encode_text(relation_text)
        e_s = encode_text(s)
        e_o = encode_text(o)

        e_rel, e_s, e_o, proj, res = relation_residual(
            e_rel, e_s, e_o
        )

        rel_norm = e_rel.norm().item()
        res_norm = res.norm().item()

        residual_ratio = (
            res_norm / max(rel_norm, 1e-12)
        )

        if res_norm > 1e-12:
            res_u = res / res_norm

            cos_rs = torch.dot(res_u, e_s).item()
            cos_ro = torch.dot(res_u, e_o).item()
            cos_rr = torch.dot(res_u, e_rel).item()
        else:
            cos_rs = 0.0
            cos_ro = 0.0
            cos_rr = 0.0

        anchor_cos = torch.dot(e_s, e_o).item()

        metrics["residual_ratio"].append(
            residual_ratio
        )
        metrics["abs_cos_res_subject"].append(
            abs(cos_rs)
        )
        metrics["abs_cos_res_object"].append(
            abs(cos_ro)
        )
        metrics["cos_relation_residual"].append(
            cos_rr
        )
        metrics["cos_subject_object"].append(
            anchor_cos
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
            "relation_text": relation_text,
            "negative_attribute": x[
                "negative_attribute"
            ],
            "residual_ratio": residual_ratio,
            "cos_res_subject": cos_rs,
            "cos_res_object": cos_ro,
            "cos_relation_residual": cos_rr,
            "cos_subject_object": anchor_cos,
        })

    print("\n" + "=" * 72)
    print(name)
    print("=" * 72)

    for k, v in metrics.items():
        summarize(k, v)

    # Degeneracy counts
    ratios = np.asarray(
        metrics["residual_ratio"]
    )

    print("\nDEGENERACY")
    print("ratio < .05 :", int((ratios < .05).sum()))
    print("ratio < .10 :", int((ratios < .10).sum()))
    print("ratio < .20 :", int((ratios < .20).sum()))
    print("ratio < .30 :", int((ratios < .30).sum()))

    # Lowest residual examples
    rows_sorted = sorted(
        rows,
        key=lambda z: z["residual_ratio"]
    )

    print("\nLOWEST 20 RESIDUAL RATIOS")

    for z in rows_sorted[:20]:
        print(
            f"{z['residual_ratio']:.4f} | "
            f"{z['relation_text']}"
        )

    return rows


# ------------------------------------------------------------
# Plug your exact BLIP / CLIP text encoders below.
#
# encode_text(text) MUST return a torch.Tensor [D]
# before any query-side NACIR modification.
# ------------------------------------------------------------

if __name__ == "__main__":
    print()
    print("Dataset loaded and relation extraction PASS.")
    print()
    print(
        "Next: wire this script to the exact frozen BLIP "
        "and CLIP text encoders used in retrieval."
    )
