"""
NACIR++ Plug-and-Play — BLIP Backbone Adapter
=================================================
Đây là adapter CỤ THỂ dùng để tái lập chính xác kết quả gốc (BRI = 0.6861)
của pipeline PlugIR/VisDial + BLIP. Nó implement 2 interface chuẩn hoá:

    - TextEncoder  (core/query_update.py cũ gọi qua `encoder` lambda + main.py
                    gọi qua `dialog_encoder` lambda)
    - ImageScorer  (core/reranker.py cũ — ITMReranker.compute_itm_score)

Toàn bộ logic BLIP-specific (get_text_features/get_image_features override,
đọc ảnh đa luồng, phạt ảnh lỗi load bằng -100.0...) được giữ nguyên 100% so
với bản gốc — chỉ đóng gói lại thành class tuân theo interface chung, để bất
kỳ backbone nào khác (CLIP, SigLIP, hay model nội bộ của method khác) có thể
thay thế mà KHÔNG đụng vào core/pipeline.
"""

import concurrent.futures
from typing import Any, List, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, BlipForImageTextRetrieval


class BlipForRetrieval(BlipForImageTextRetrieval):
    """Giữ nguyên 100% override gốc trong main.py."""

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


class BlipTextEncoder:
    """Implement interfaces.TextEncoder bằng BLIP text encoder (giữ nguyên `dialog_encoder` gốc)."""

    def __init__(self, model: BlipForRetrieval, processor: AutoProcessor, device: str):
        self.model = model
        self.processor = processor
        self.device = device

    def encode(self, texts: List[str]) -> torch.Tensor:
        inputs = self.processor(text=texts, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            return self.model.get_text_features(**inputs)


class BlipImageEncoder:
    """Tương đương `ImageEmbedder` gốc trong main.py — dùng để build corpus_vectors."""

    def __init__(self, model: BlipForRetrieval, processor: AutoProcessor, device: str):
        self.model = model
        self.processor = processor
        self.device = device

    def preprocess(self, path: str):
        return self.processor(images=Image.open(path), return_tensors="pt")["pixel_values"][0]

    @torch.no_grad()
    def encode_batch(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model.get_image_features(pixel_values.to(self.device))


class BlipITMScorer:
    """
    Implement interfaces.ImageScorer bằng BLIP ITM head.
    Giữ nguyên 100% logic gốc core/reranker.py::ITMReranker.compute_itm_score
    (đọc ảnh đa luồng để tránh nghẽn NAS, phạt -100.0 cho ảnh lỗi load).
    """

    def __init__(self, model: BlipForRetrieval, processor: AutoProcessor, device: str, batch_size: int = 16):
        self.model = model
        self.processor = processor
        self.device = device
        self.batch_size = batch_size

    @torch.no_grad()
    def score(self, query_text: str, image_refs: List[Any]) -> torch.Tensor:
        """`image_refs` ở đây là danh sách đường dẫn ảnh (path)."""
        image_paths = image_refs
        all_scores = []

        def load_image(p):
            try:
                return Image.open(p).convert("RGB"), True
            except Exception:
                return Image.new("RGB", (384, 384)), False

        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i:i + self.batch_size]

            with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, len(batch_paths))) as executor:
                results = list(executor.map(load_image, batch_paths))
                images = [res[0] for res in results]
                valid_mask = torch.tensor([res[1] for res in results], dtype=torch.bool, device=self.device)

            inputs = self.processor(
                text=[query_text] * len(images), images=images,
                return_tensors="pt", padding=True, truncation=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model(**inputs)
            itm_logits = outputs.itm_score
            itm_probs = F.softmax(itm_logits, dim=-1)
            match_scores = itm_probs[:, 1]

            match_scores = torch.where(valid_mask, match_scores, torch.tensor(-100.0, device=self.device))
            all_scores.append(match_scores.cpu())

        return torch.cat(all_scores)


def build_blip_backbone(device: str, model_id: str = "Salesforce/blip-itm-large-coco"):
    """Tiện ích dựng nhanh bộ 3: text_encoder, image_encoder, itm_scorer từ BLIP."""
    model = BlipForRetrieval.from_pretrained(model_id).to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    return (
        BlipTextEncoder(model, processor, device),
        BlipImageEncoder(model, processor, device),
        BlipITMScorer(model, processor, device),
    )
