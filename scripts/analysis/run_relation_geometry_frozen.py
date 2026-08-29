import gc
import json
import inspect
from pathlib import Path

import torch

from audit_relation_geometry import run_audit, rels
from nacir.adapters.plugir_blip import load_blip_text_encoder
from nacir.adapters.openai_clip_vitl14 import load_clip_text_encoder


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUT_DIR = Path(
    "artifacts_final/typed_nacir/relation_geometry_v1"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_frozen(loader):
    """
    Call the repository's exact adapter loader without assuming
    optional argument names that may differ between adapters.
    """
    sig = inspect.signature(loader)

    kwargs = {}

    if "device" in sig.parameters:
        kwargs["device"] = DEVICE

    if "allow_download" in sig.parameters:
        kwargs["allow_download"] = False

    return loader(**kwargs)


def make_encode_fn(encoder):
    def encode_text(text):
        with torch.inference_mode():
            v = encoder.encode([text])

        assert isinstance(v, torch.Tensor)
        assert v.ndim == 2
        assert v.shape[0] == 1

        v = v[0].detach().float().cpu()

        assert torch.isfinite(v).all()
        assert v.norm().item() > 0

        return v

    return encode_text


def save_rows(name, rows):
    path = OUT_DIR / f"{name}_relation_geometry.json"

    payload = {
        "backbone": name,
        "num_relations": len(rows),
        "semantic_artifact": (
            "artifacts_final/typed_nacir/"
            "chatir_structured_negative_final_v1_1.json"
        ),
        "semantic_artifact_sha256": (
            "14b2d452336f1526fd92a2ff1159845cab5cf22aa49dcd26fe004592898d6840"
        ),
        "retrieval_results_used": False,
        "rows": rows,
    }

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("saved:", path)


print("=" * 72)
print("FROZEN RELATION GEOMETRY AUDIT")
print("=" * 72)
print("device   :", DEVICE)
print("relations:", len(rels))

assert len(rels) == 141


# ============================================================
# BLIP
# ============================================================

print("\nLoading frozen BLIP text encoder...")

blip = load_frozen(
    load_blip_text_encoder
)

blip_encode = make_encode_fn(blip)

blip_rows = run_audit(
    "BLIP_ITM_LARGE_COCO",
    blip_encode,
)

save_rows(
    "blip_itm_large_coco",
    blip_rows,
)

del blip_encode
del blip

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\nBLIP released.")


# ============================================================
# CLIP
# ============================================================

print("\nLoading frozen OpenAI CLIP ViT-L/14 text encoder...")

clip_encoder = load_frozen(
    load_clip_text_encoder
)

clip_encode = make_encode_fn(
    clip_encoder
)

clip_rows = run_audit(
    "OPENAI_CLIP_VIT_L14",
    clip_encode,
)

save_rows(
    "openai_clip_vitl14",
    clip_rows,
)

del clip_encode
del clip_encoder

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()


print("\n" + "=" * 72)
print("GEOMETRY AUDIT COMPLETE")
print("=" * 72)
print("PASS")
