import json
import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from nacir.adapters.plugir_blip import (
    NormalizedBLIP,
    MODEL_ID,
    MODEL_REVISION,
)
from transformers import AutoProcessor


ROOT = Path(__file__).resolve().parents[3]

MANIFEST = (
    ROOT
    / "artifacts_final/diagnostic_frozen/"
      "diagnostic_gallery_1653.json"
)

OUT = (
    ROOT
    / "artifacts_final/diagnostic_frozen/"
      "corpus_diagnostic_1653_blip.pt"
)

PROV = OUT.with_suffix(".provenance.json")

DEVICE = "cuda"
BATCH = 128
WORKERS = 8


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


gallery = json.load(open(MANIFEST, encoding="utf-8"))

assert len(gallery) == 1653
assert all(
    row["gallery_index"] == i
    for i, row in enumerate(gallery)
)

print("Loading pinned BLIP...")

model = NormalizedBLIP.from_pretrained(
    MODEL_ID,
    revision=MODEL_REVISION,
    local_files_only=True,
).to(DEVICE).eval()

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    revision=MODEL_REVISION,
    local_files_only=True,
)

revision = str(
    getattr(model.config, "_commit_hash", None)
    or "unresolved"
)

assert revision == MODEL_REVISION, (
    revision,
    MODEL_REVISION,
)


class GalleryDataset(Dataset):
    def __len__(self):
        return len(gallery)

    def __getitem__(self, idx):
        row = gallery[idx]

        path = Path(row["absolute_path"])
        assert path.is_file(), path

        image = Image.open(path).convert("RGB")

        # return PIL; collate below handles processor
        return idx, image


def collate(batch):
    ids = [x[0] for x in batch]
    images = [x[1] for x in batch]

    encoded = processor(
        images=images,
        return_tensors="pt",
    )

    return (
        torch.tensor(ids, dtype=torch.long),
        encoded["pixel_values"],
    )


loader = DataLoader(
    GalleryDataset(),
    batch_size=BATCH,
    shuffle=False,
    num_workers=WORKERS,
    pin_memory=True,
    collate_fn=collate,
)

all_ids = []
all_vectors = []

with torch.inference_mode():

    done = 0

    for ids, pixel_values in loader:

        pixel_values = pixel_values.to(
            DEVICE,
            non_blocking=True,
        )

        vision_outputs = model.vision_model(
            pixel_values=pixel_values,
            return_dict=True,
        )

        image_embeds = vision_outputs.last_hidden_state

        vec = model.vision_proj(
            image_embeds[:, 0, :]
        )

        vec = F.normalize(
            vec.float(),
            dim=-1,
        )

        assert torch.isfinite(vec).all()

        all_ids.append(ids.cpu())
        all_vectors.append(vec.cpu())

        done += len(ids)

        print(
            f"\rEncoded {done}/{len(gallery)}",
            end="",
            flush=True,
        )

print()

ids = torch.cat(all_ids)
vectors = torch.cat(all_vectors)

assert ids.shape == (1653,)
assert torch.equal(
    ids,
    torch.arange(1653),
)

assert vectors.shape[0] == 1653
assert vectors.shape[1] == 256
assert torch.isfinite(vectors).all()

norms = vectors.norm(dim=-1)

assert torch.allclose(
    norms,
    torch.ones_like(norms),
    atol=1e-4,
)

OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

torch.save(
    {
        "vectors": vectors,
        "ids": ids,
    },
    OUT,
)

prov = {
    "status": "complete",
    "benchmark": "NACIR-Diagnostic-1653",
    "encoder": "BLIP",
    "model_id": MODEL_ID,
    "model_revision": MODEL_REVISION,
    "embedding_dim": 256,
    "normalized": True,
    "num_images": 1653,
    "gallery_manifest": str(MANIFEST),
    "gallery_manifest_sha256": sha256(MANIFEST),
    "output": str(OUT),
}

PROV.write_text(
    json.dumps(
        prov,
        indent=2,
    ),
    encoding="utf-8",
)

print("\n[PASS]")
print("vectors:", tuple(vectors.shape))
print(
    "norm min/mean/max:",
    norms.min().item(),
    norms.mean().item(),
    norms.max().item(),
)
print("saved:", OUT)
print("provenance:", PROV)
