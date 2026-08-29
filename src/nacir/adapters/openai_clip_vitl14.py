"""Pinned OpenAI CLIP ViT-L/14 text adapter."""

from __future__ import annotations

from collections.abc import Sequence

import clip
import torch
import torch.nn.functional as F


MODEL_ID = "OpenAI CLIP ViT-L/14"
CLIP_REPOSITORY_REVISION = "a1d071733d7111c9c014f024669f959182114e33"
MODEL_REVISION = f"openai/CLIP@{CLIP_REPOSITORY_REVISION}:ViT-L/14"


class OpenAICLIPTextEncoder:
    def __init__(self, model, device: str):
        self.model = model
        self.device = device

    @torch.inference_mode()
    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        values = list(texts)

        if not values or any(
            not isinstance(x, str) or not x.strip()
            for x in values
        ):
            raise ValueError("text encoding requires non-empty strings")

        tokens = clip.tokenize(
            values,
            truncate=True,
        ).to(self.device)

        vectors = self.model.encode_text(tokens).float()
        vectors = F.normalize(vectors, dim=-1)

        if (
            vectors.ndim != 2
            or vectors.shape[0] != len(values)
            or vectors.shape[1] != 768
            or not torch.isfinite(vectors).all()
        ):
            raise RuntimeError(
                f"invalid CLIP text vectors: {tuple(vectors.shape)}"
            )

        return vectors


def load_clip_text_encoder(
    device: str,
    *,
    allow_download: bool = False,
):
    """Load the paper-pinned OpenAI CLIP ViT-L/14 text encoder.

    OpenAI CLIP's public ``clip.load`` API does not expose a strict
    ``local_files_only`` switch. To avoid silently permitting network access,
    callers must opt in with ``allow_download=True``. When the checkpoint is
    already cached, ``clip.load`` will reuse it rather than downloading again.
    """
    if not allow_download:
        raise RuntimeError(
            "OpenAI CLIP loading may access the network when the checkpoint is "
            "not cached. Re-run with --allow-download to explicitly permit "
            "clip.load; cached checkpoints will still be reused."
        )

    model, _ = clip.load(
        "ViT-L/14",
        device=device,
    )

    model = model.eval()

    if tuple(model.text_projection.shape) != (768, 768):
        raise RuntimeError(
            "unexpected ViT-L/14 projection dimension"
        )

    return OpenAICLIPTextEncoder(model, device)
