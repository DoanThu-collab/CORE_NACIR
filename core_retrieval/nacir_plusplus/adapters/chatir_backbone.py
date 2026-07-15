"""ChatIR backbone adapter for NACIR.

This module provides the interfaces required by the NACIR pipeline,
including text encoding, image encoding, and ITM scoring.
"""

import os
import sys
import time
import logging
import multiprocessing as mp
import queue
import numpy as np
import concurrent.futures
from typing import Tuple, List, Union

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


IMAGE_SIZE = 224
TEXT_MAX_LENGTH = 200
IMAGE_LOAD_RETRIES = 3
IMAGE_LOAD_SLEEP_SEC = 0.5
IMAGE_LOAD_WORKERS = 25
ITM_CHUNK_SIZE = 50
ITM_SCORE_SCALE = 0.8
INVALID_SCORE = -100.0
IMAGE_LOAD_TIMEOUT_SEC = 30.0


def _worker_load_image(path, q):
    """Load an image in a worker process and send back raw pixel data."""
    try:
        img = Image.open(path).convert("RGB")
        q.put(("ok", np.array(img)))
    except Exception as e:
        q.put(("error", str(e)))


class TextEncoder:
    """Encode text inputs into normalized embedding vectors."""

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self._tokenize_cache = {}

    def _tokenize_cached(self, text_key: str, texts: List[str]):
        """Cache tokenized inputs to avoid repeated tokenization."""
        if text_key not in self._tokenize_cache:
            text = self.model.tokenizer(
                texts,
                padding="longest",
                truncation=True,
                max_length=TEXT_MAX_LENGTH,
                return_tensors="pt",
            ).to(self.device)
            self._tokenize_cache[text_key] = text
        return self._tokenize_cache[text_key]

    def encode_text(self, texts):
        """Encode one or more text strings into normalized embeddings."""
        if isinstance(texts, str):
            texts = [texts]

        text_key = "|".join(texts) if len(texts) > 1 else texts[0]
        text = self._tokenize_cached(text_key, texts)

        with torch.inference_mode():
            text_output = self.model.text_encoder(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
                mode="text",
            )
            shift = self.model.text_proj(text_output.last_hidden_state[:, 0, :])

        return F.normalize(shift, dim=-1)

    def encode(self, texts):
        """Backward-compatible alias for text encoding."""
        return self.encode_text(texts)

    def clear_cache(self):
        """Clear cached tokenized inputs."""
        self._tokenize_cache.clear()


class ImageEncoder:
    """Encode image batches into normalized embedding vectors."""

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ])
        self.preprocess = self._make_preprocess()

    def _make_preprocess(self):
        def _preprocess(image_path):
            for attempt in range(IMAGE_LOAD_RETRIES):
                try:
                    with Image.open(image_path) as img:
                        img = img.convert("RGB")
                        return self.transform(img)
                except OSError:
                    if attempt == IMAGE_LOAD_RETRIES - 1:
                        raise
                    time.sleep(IMAGE_LOAD_SLEEP_SEC)
        return _preprocess

    def encode_image(self, images_batch):
        """Encode a batch of images into normalized embeddings."""
        if isinstance(images_batch, dict):
            pixel_values = images_batch["pixel_values"].to(self.device)
        else:
            pixel_values = images_batch.to(self.device)

        with torch.inference_mode():
            embeds = self.model.visual_encoder(pixel_values)
            projection = self.model.vision_proj(embeds[:, 0, :])

        return F.normalize(projection, dim=-1)

    def encode(self, images_batch):
        """Backward-compatible alias for image encoding."""
        return self.encode_image(images_batch)


class ITMScorer:
    """Score image-text pairs with the ITM head."""

    def __init__(self, model, device, image_transform, read_timeout_sec: float = IMAGE_LOAD_TIMEOUT_SEC):
        self.model = model
        self.device = device
        self.transform = image_transform
        self.read_timeout_sec = read_timeout_sec
        self.itm_score_scale = ITM_SCORE_SCALE
        self._chunk_size = ITM_CHUNK_SIZE

        logger = logging.getLogger(__name__)
        logger.info("ITM score scaling factor: %s", self.itm_score_scale)
 
    @property
    def device_(self):
        return next(self.model.parameters()).device
 
    def _load_one_image_safe(self, path: str):
        """Load one image with a multiprocessing timeout guard."""
        q = mp.Queue()
        p = mp.Process(target=_worker_load_image, args=(path, q))
        p.start()

        try:
            status, data = q.get(timeout=self.read_timeout_sec)
            p.join()
            
            if status == "ok":
                raw = Image.fromarray(data)
                tensor = self.transform(raw)
                return tensor, True
            else:
                logging.warning("Image load failed for %s: %s", path, data)
                return torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE), False
                
        except queue.Empty:
            logging.error("Image load timeout (> %ss): %s", self.read_timeout_sec, path)
            p.kill()
            p.join()
            return torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE), False
 
    def _load_images_parallel(self, image_paths: List[str]):
        """Load images in parallel using worker processes."""
        results = [None] * len(image_paths)
        valid_mask = [False] * len(image_paths)

        with concurrent.futures.ThreadPoolExecutor(max_workers=IMAGE_LOAD_WORKERS) as executor:
            future_to_idx = {
                executor.submit(self._load_one_image_safe, path): i
                for i, path in enumerate(image_paths)
            }
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                tensor, is_valid = future.result()
                results[idx] = tensor
                valid_mask[idx] = is_valid

        return torch.stack(results), torch.tensor(valid_mask, dtype=torch.bool)
 
    def score_itm(self, query_text: Union[str, List[str]], image_refs: List) -> torch.Tensor:
        """Score a query against a batch of candidate images."""
        if isinstance(query_text, list):
            query_text = query_text[0]
 
        if len(image_refs) > 0 and isinstance(image_refs[0], str):
            pixel_values, valid_mask = self._load_images_parallel(image_refs)
            pixel_values = pixel_values.to(self.device)
            valid_mask = valid_mask.to(self.device)
        else:
            pixel_values = image_refs.to(self.device)
            valid_mask = torch.ones(pixel_values.size(0), dtype=torch.bool, device=self.device)
 
        with torch.inference_mode():
            num_images = pixel_values.size(0)

            text = self.model.tokenizer(
                query_text,
                padding="longest",
                truncation=True,
                max_length=TEXT_MAX_LENGTH,
                return_tensors="pt",
            ).to(self.device)

            if num_images <= self._chunk_size * 2:
                image_embeds = self.model.visual_encoder(pixel_values)
                image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=self.device)

                input_ids = text.input_ids.expand(num_images, -1)
                attention_mask = text.attention_mask.expand(num_images, -1)
                
                output = self.model.text_encoder(
                    input_ids,
                    attention_mask=attention_mask,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
                
                itm_logits = self.model.itm_head(output.last_hidden_state[:, 0, :])
                scores = F.softmax(itm_logits, dim=1)[:, 1]
            else:
                all_scores = []
                
                for i in range(0, num_images, self._chunk_size):
                    pv_chunk = pixel_values[i:i+self._chunk_size]
                    current_batch = pv_chunk.size(0)
                    
                    image_embeds = self.model.visual_encoder(pv_chunk)
                    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=self.device)

                    input_ids = text.input_ids.expand(current_batch, -1)
                    attention_mask = text.attention_mask.expand(current_batch, -1)
                    
                    output = self.model.text_encoder(
                        input_ids,
                        attention_mask=attention_mask,
                        encoder_hidden_states=image_embeds,
                        encoder_attention_mask=image_atts,
                        return_dict=True,
                    )
                    
                    itm_logits = self.model.itm_head(output.last_hidden_state[:, 0, :])
                    scores_chunk = F.softmax(itm_logits, dim=1)[:, 1]
                    all_scores.append(scores_chunk)
                
                scores = torch.cat(all_scores)
 
            scores = scores * self.itm_score_scale
            scores = torch.where(
                valid_mask, scores, torch.tensor(INVALID_SCORE, device=self.device)
            )
 
        return scores.to(self.device_)

    def score(self, query_text: Union[str, List[str]], image_refs: List) -> torch.Tensor:
        """Backward-compatible alias for ITM scoring."""
        return self.score_itm(query_text, image_refs)


def build_backbone(
    device: str,
    chatir_repo_dir: str,
    finetuned_ckpt_path: str,
) -> Tuple:
    """Build the BLIP-ITM backbone and wrap it for NACIR."""
    logger = logging.getLogger(__name__)

    blip_subdir = os.path.join(chatir_repo_dir, "BLIP")
    for p in (chatir_repo_dir, blip_subdir):
        if p not in sys.path:
            sys.path.insert(0, p)

    from importlib import import_module

    blip_itm = import_module("BLIP.models.blip_itm").blip_itm

    med_config_path = os.path.join(chatir_repo_dir, "BLIP", "configs", "med_config.json")

    logger.info("Loading BLIP-ITM checkpoint: %s", finetuned_ckpt_path)
    model = blip_itm(
        pretrained=finetuned_ckpt_path,
        med_config=med_config_path,
        image_size=IMAGE_SIZE,
        vit="base",
    )

    model = model.to(device).eval()

    text_encoder = TextEncoder(model, device)
    image_encoder = ImageEncoder(model, device)
    itm_scorer = ITMScorer(model, device, image_encoder.transform)

    return text_encoder, image_encoder, itm_scorer


def build_chatir_backbone(
    device: str,
    chatir_repo_dir: str,
    finetuned_ckpt_path: str,
) -> Tuple:
    """Backward-compatible alias for ChatIR backbone construction."""
    return build_backbone(device, chatir_repo_dir, finetuned_ckpt_path)