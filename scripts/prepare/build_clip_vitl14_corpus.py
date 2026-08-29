import os, json, hashlib
from pathlib import Path

import clip
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader


ROOT = Path("/mlcv1/WorkingSpace/Personal/core_baotg/thu/CORE_NACIR_24H")

CORPUS_JSON = Path(
    "/mlcv1/WorkingSpace/Personal/core_baotg/thuy/PlugIR/"
    "Protocol/Search_Space_val_50k.json"
)

IMAGE_ROOT = Path(
    "/mlcv1/WorkingSpace/Personal/core_baotg/thuy/Dataset/PlugIR"
)

OUT = ROOT / "artifacts_final/corpus_openai_clip_vitl14_vectors.pt"
PROV = ROOT / "artifacts_final/corpus_openai_clip_vitl14_vectors.provenance.json"

DEVICE = "cuda"
BATCH = 256
WORKERS = 8


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


paths = json.load(open(CORPUS_JSON, encoding="utf-8"))
assert len(paths) == 50000


model, preprocess = clip.load("ViT-L/14", device=DEVICE)
model.eval()


class Corpus(Dataset):
    def __len__(self):
        return len(paths)

    def __getitem__(self, idx):
        p = IMAGE_ROOT / paths[idx]
        img = Image.open(p).convert("RGB")
        return idx, preprocess(img)


loader = DataLoader(
    Corpus(),
    batch_size=BATCH,
    shuffle=False,
    num_workers=WORKERS,
    pin_memory=True,
    persistent_workers=True,
)

all_ids = []
all_vec = []

with torch.inference_mode():
    done = 0

    for ids, imgs in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)

        vec = model.encode_image(imgs).float()
        vec = F.normalize(vec, dim=-1)

        all_ids.append(ids.cpu())
        all_vec.append(vec.cpu())

        done += len(ids)

        if done % 2048 < BATCH or done == len(paths):
            print(
                f"[{done:5d}/{len(paths)}] "
                f"{100*done/len(paths):6.2f}%"
            )

ids = torch.cat(all_ids)
vectors = torch.cat(all_vec)

assert ids.shape == (50000,)
assert torch.equal(ids, torch.arange(50000))
assert vectors.shape == (50000, 768)
assert torch.isfinite(vectors).all()

norms = vectors.norm(dim=-1)

torch.save({
    "vectors": vectors,
    "ids": ids,
}, OUT)

prov = {
    "status": "complete",
    "encoder": "OpenAI CLIP",
    "model": "ViT-L/14",
    "embedding_dim": 768,
    "normalized": True,
    "num_images": 50000,
    "corpus_json": str(CORPUS_JSON),
    "corpus_sha256": sha256(CORPUS_JSON),
    "image_root": str(IMAGE_ROOT),
    "output": str(OUT),
}

with open(PROV, "w") as f:
    json.dump(prov, f, indent=2)

print("\n[PASS]")
print("vectors:", vectors.shape)
print(
    "norm min/mean/max:",
    norms.min().item(),
    norms.mean().item(),
    norms.max().item()
)
print("saved:", OUT)
