#!/usr/bin/env python3

import argparse
import hashlib
import json
from pathlib import Path

import clip
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the OpenAI CLIP ViT-L/14 corpus embedding cache."
    )
    parser.add_argument("--corpus-json", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance-output", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--expected-size", type=int, default=50000)
    args = parser.parse_args()

    paths = json.load(args.corpus_json.open(encoding="utf-8"))
    if len(paths) != args.expected_size:
        raise ValueError(
            f"expected {args.expected_size} corpus items, found {len(paths)}"
        )

    model, preprocess = clip.load("ViT-L/14", device=args.device)
    model.eval()

    class Corpus(Dataset):
        def __len__(self):
            return len(paths)

        def __getitem__(self, idx):
            image_path = args.image_root / paths[idx]
            image = Image.open(image_path).convert("RGB")
            return idx, preprocess(image)

    loader = DataLoader(
        Corpus(),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.workers > 0,
    )

    all_ids = []
    all_vectors = []

    with torch.inference_mode():
        done = 0
        for ids, images in loader:
            images = images.to(args.device, non_blocking=True)
            vectors = F.normalize(model.encode_image(images).float(), dim=-1)
            all_ids.append(ids.cpu())
            all_vectors.append(vectors.cpu())
            done += len(ids)
            if done % 2048 < args.batch_size or done == len(paths):
                print(f"[{done:5d}/{len(paths)}] {100 * done / len(paths):6.2f}%")

    ids = torch.cat(all_ids)
    vectors = torch.cat(all_vectors)

    if ids.shape != (args.expected_size,):
        raise RuntimeError(f"unexpected id shape: {tuple(ids.shape)}")
    if not torch.equal(ids, torch.arange(args.expected_size)):
        raise RuntimeError("corpus row order changed during embedding")
    if vectors.shape != (args.expected_size, 768):
        raise RuntimeError(f"unexpected vector shape: {tuple(vectors.shape)}")
    if not torch.isfinite(vectors).all():
        raise RuntimeError("corpus vectors contain non-finite values")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"vectors": vectors, "ids": ids}, args.output)

    provenance_output = args.provenance_output or args.output.with_suffix(
        args.output.suffix + ".provenance.json"
    )
    provenance_output.parent.mkdir(parents=True, exist_ok=True)
    provenance = {
        "status": "complete",
        "encoder": "OpenAI CLIP",
        "model": "ViT-L/14",
        "embedding_dim": 768,
        "normalized": True,
        "num_images": len(paths),
        "corpus_json": str(args.corpus_json),
        "corpus_sha256": sha256(args.corpus_json),
        "image_root": str(args.image_root),
        "output": str(args.output),
    }
    with provenance_output.open("w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)

    norms = vectors.norm(dim=-1)
    print("\n[PASS]")
    print("vectors:", vectors.shape)
    print(
        "norm min/mean/max:",
        norms.min().item(),
        norms.mean().item(),
        norms.max().item(),
    )
    print("saved:", args.output)
    print("provenance:", provenance_output)


if __name__ == "__main__":
    main()
