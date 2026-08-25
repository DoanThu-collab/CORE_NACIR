"""PlugIR-compatible pinned BLIP text adapter used by the paper runs."""

from __future__ import annotations

import os
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, BlipForImageTextRetrieval


MODEL_ID = "Salesforce/blip-itm-large-coco"
MODEL_REVISION = "19502f1e215844f7e48bd48473f86932486d3441"


class NormalizedBLIP(BlipForImageTextRetrieval):
    """BLIP retrieval model with normalized projected text features."""

    def get_text_features(self, input_ids, attention_mask=None, return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
        )
        hidden = output.last_hidden_state if return_dict else output[0]
        return F.normalize(self.text_proj(hidden[:, 0, :]), dim=-1)


class BLIPTextEncoder:
    def __init__(self, model: NormalizedBLIP, processor: AutoProcessor, device: str) -> None:
        self.model = model
        self.processor = processor
        self.device = device

    @torch.inference_mode()
    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        values = list(texts)
        if not values or any(not isinstance(text, str) or not text.strip() for text in values):
            raise ValueError("text encoding requires non-empty strings")
        batch = self.processor(text=values, padding=True, truncation=True, return_tensors="pt")
        batch = {key: value.to(self.device) for key, value in batch.items()}
        vectors = self.model.get_text_features(**batch)
        if vectors.ndim != 2 or vectors.shape[0] != len(values) or not torch.isfinite(vectors).all():
            raise RuntimeError("BLIP returned invalid text vectors")
        return vectors


def load_blip_text_encoder(device: str, *, allow_download: bool = False) -> BLIPTextEncoder:
    """Load the paper-pinned BLIP text encoder.

    The default is offline-only to prevent a silent revision change. Set
    ``allow_download=True`` only after verifying the exact revision is available.
    """

    if not allow_download:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    model = NormalizedBLIP.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=not allow_download,
    ).to(device).eval()
    revision = str(getattr(model.config, "_commit_hash", None) or "unresolved")
    if revision != MODEL_REVISION:
        raise RuntimeError("loaded BLIP revision does not match the paper protocol")
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=not allow_download,
    )
    return BLIPTextEncoder(model, processor, device)
