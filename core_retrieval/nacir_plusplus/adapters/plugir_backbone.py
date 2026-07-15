"""PlugIR adapter for NACIR++.

This module exposes text encoding, image encoding, and ITM scoring
interfaces for PlugIR-based retrieval backbones.
"""

import logging
import concurrent.futures
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, BlipForImageTextRetrieval

logger = logging.getLogger(__name__)

IMAGE_LOAD_TIMEOUT_SEC = 60.0
DEFAULT_ITM_BATCH_SIZE = 16
DEFAULT_MODEL_ID = "Salesforce/blip-itm-large-coco"

class BlipForRetrieval(BlipForImageTextRetrieval):
    """BLIP retrieval model that returns normalized embeddings."""

    def get_text_features(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        question_embeds = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=return_dict
        )
        question_embeds = question_embeds[0] if not return_dict else question_embeds.last_hidden_state
        return F.normalize(self.text_proj(question_embeds[:, 0, :]), dim=-1)

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return F.normalize(self.vision_proj(vision_outputs[0][:, 0, :]), dim=-1)

class TextEncoder:
    """Encode text inputs into normalized vectors."""

    def __init__(self, model: BlipForRetrieval, processor: AutoProcessor, device: str):
        self.model = model
        self.processor = processor
        self.device = device

    def encode_text(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """Encode one or more text strings into normalized embeddings."""
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.processor(text=texts, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            return self.model.get_text_features(**inputs)

    def encode(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """Backward-compatible alias for text encoding."""
        return self.encode_text(texts)

class ImageEncoder:
    """Encode images into normalized vectors."""

    def __init__(self, model: BlipForRetrieval, processor: AutoProcessor, device: str):
        self.model = model
        self.processor = processor
        self.device = device

    def preprocess(self, path: str):
        return self.processor(images=Image.open(path), return_tensors="pt")["pixel_values"][0]

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images into normalized embeddings."""
        return self.model.get_image_features(pixel_values.to(self.device))

    @torch.no_grad()
    def encode_batch(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Backward-compatible alias for image encoding."""
        return self.encode_image(pixel_values)

class ITMScorer:
    """Score image-text pairs using the ITM head."""

    def __init__(
        self,
        model: BlipForRetrieval,
        processor: AutoProcessor,
        device: str,
        batch_size: int = DEFAULT_ITM_BATCH_SIZE,
        cache_size: Optional[int] = None,
    ):
        self.model = model
        self.processor = processor
        self.device = device
        self.batch_size = batch_size
        self.cache_size = cache_size
        self._embeds_cache: Dict[str, torch.Tensor] = {}
        self._cache_order: List[str] = []
        self._invalid_paths: Set[str] = set()

    def _evict_if_needed(self) -> None:
        if self.cache_size is None:
            return
        while len(self._embeds_cache) > self.cache_size:
            oldest = self._cache_order.pop(0)
            self._embeds_cache.pop(oldest, None)

    @staticmethod
    def _load_image(path: str):
        try:
            return Image.open(path).convert("RGB"), True
        except Exception:
            return None, False

    @torch.no_grad()
    def _ensure_embeds_cached(self, image_paths: List[str]) -> None:
        to_compute = [
            p for p in dict.fromkeys(image_paths)
            if p not in self._embeds_cache and p not in self._invalid_paths
        ]
        if not to_compute:
            return

        for i in range(0, len(to_compute), self.batch_size):
            batch_paths = to_compute[i:i + self.batch_size]

            executor = concurrent.futures.ThreadPoolExecutor(max_workers=min(32, len(batch_paths)))
            futures_map = {executor.submit(self._load_image, p): p for p in batch_paths}
            done, not_done = concurrent.futures.wait(futures_map.keys(), timeout=IMAGE_LOAD_TIMEOUT_SEC)

            valid_paths = []
            valid_images = []
            for fut in futures_map:
                p = futures_map[fut]
                if fut in not_done:
                    fut.cancel()
                    continue
                try:
                    img, ok = fut.result()
                    if ok:
                        valid_paths.append(p)
                        valid_images.append(img)
                    else:
                        self._invalid_paths.add(p)
                except Exception:
                    self._invalid_paths.add(p)

            executor.shutdown(wait=False, cancel_futures=True)

            if not valid_images:
                continue

            inputs = self.processor(images=valid_images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)

            vision_outputs = self.model.vision_model(pixel_values=pixel_values)
            image_embeds = vision_outputs.last_hidden_state.detach().cpu()

            for p, emb in zip(valid_paths, image_embeds):
                self._embeds_cache[p] = emb
                self._cache_order.append(p)

            self._evict_if_needed()

    @torch.no_grad()
    def score_itm(self, query_text: str, image_refs: List[str]) -> torch.Tensor:
        """Score a text query against a batch of candidate image paths."""
        self._ensure_embeds_cached(image_refs)

        all_scores = []
        for i in range(0, len(image_refs), self.batch_size):
            batch_paths = image_refs[i:i + self.batch_size]
            
            valid_paths = [p for p in batch_paths
                           if p not in self._invalid_paths and p in self._embeds_cache]

            batch_scores = torch.full((len(batch_paths),), -100.0, device=self.device)

            if valid_paths:
                image_embeds = torch.stack([self._embeds_cache[p] for p in valid_paths]).to(self.device)
                image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=self.device)

                text_inputs = self.processor(
                    text=[query_text] * len(valid_paths),
                    return_tensors="pt", padding=True, truncation=True,
                )
                text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

                question_embeds = self.model.text_encoder(
                    input_ids=text_inputs["input_ids"],
                    attention_mask=text_inputs.get("attention_mask"),
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                ).last_hidden_state

                itm_logits = self.model.itm_head(question_embeds[:, 0, :])
                itm_probs = F.softmax(itm_logits, dim=-1)
                valid_scores = itm_probs[:, 1]

                valid_idx = 0
                for j, p in enumerate(batch_paths):
                    if p not in self._invalid_paths and p in self._embeds_cache:
                        batch_scores[j] = valid_scores[valid_idx]
                        valid_idx += 1

            all_scores.append(batch_scores.cpu())

        return torch.cat(all_scores)

    def score(self, query_text: str, image_refs: List[str]) -> torch.Tensor:
        """Backward-compatible alias for ITM scoring."""
        return self.score_itm(query_text, image_refs)


def build_backbone(device: str, model_id: str = DEFAULT_MODEL_ID) -> Tuple:
    """Build the PlugIR BLIP-ITM backbone adapter."""
    logger.info("Loading BLIP-ITM backbone: %s", model_id)
    model = BlipForRetrieval.from_pretrained(model_id).to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    return (
        TextEncoder(model, processor, device),
        ImageEncoder(model, processor, device),
        ITMScorer(model, processor, device),
    )


def build_plugir_backbone(device: str, model_id: str = DEFAULT_MODEL_ID) -> Tuple:
    """Backward-compatible alias for PlugIR backbone construction."""
    return build_backbone(device, model_id)
