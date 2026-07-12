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
from typing import Any, Dict, List, Optional, Set

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
    Implement interfaces.ImageScorer bang BLIP ITM head.

    TOI UU (so voi ban goc): cache `image_embeds` (output cua `vision_model`,
    tuc ViT) theo duong dan anh. Phan nay KHONG phu thuoc query text -- xem
    source goc BlipForImageTextRetrieval.forward():

        vision_outputs = self.vision_model(pixel_values=...)
        image_embeds = vision_outputs.last_hidden_state   # <- khong phu thuoc text
        question_embeds = self.text_encoder(
            input_ids=..., encoder_hidden_states=image_embeds, ...
        )                                                  # <- phu thuoc CA HAI

    Vi cung mot anh luon cho ra dung cung image_embeds (ham tat dinh, chay
    trong torch.no_grad(), khong dropout o eval mode), viec cache KHONG lam
    doi bat ky con so nao so voi ban khong cache -- chi tranh tinh lai phan
    ViT (nang nhat) moi khi cung mot anh xuat hien lai o turn/session khac.
    Phan cross-attention (text_encoder + itm_head) van luon duoc tinh lai
    moi lan goi, vi no phu thuoc truc tiep vao query_text.

    Hanh vi phat anh loi load bang -100.0 va doc anh da luong duoc giu
    nguyen 100% so voi ban goc.
    """

    def __init__(
        self,
        model: BlipForRetrieval,
        processor: AutoProcessor,
        device: str,
        batch_size: int = 16,
        cache_size: Optional[int] = None,
    ):
        self.model = model
        self.processor = processor
        self.device = device
        self.batch_size = batch_size
        # cache_size=None -> giu toan bo embeds da tinh trong RAM (mac dinh).
        # Dat mot so nguyen neu corpus qua lon de gioi han RAM (FIFO eviction).
        self.cache_size = cache_size
        self._embeds_cache: Dict[str, torch.Tensor] = {}   # path -> image_embeds [seq, dim] (luu tren CPU)
        self._cache_order: List[str] = []
        self._invalid_paths: Set[str] = set()               # anh loi load -> luon phat -100.0

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
        """Tinh image_embeds (chi phan vision_model) cho cac path CHUA co trong cache."""
        to_compute = [
            p for p in dict.fromkeys(image_paths)  # unique, giu thu tu
            if p not in self._embeds_cache and p not in self._invalid_paths
        ]
        if not to_compute:
            return

        for i in range(0, len(to_compute), self.batch_size):
            batch_paths = to_compute[i:i + self.batch_size]

            # Dùng submit + wait(timeout) thay cho with-block để tránh D-state NAS hang.
            # - Ảnh timeout: bỏ qua lần này, KHÔNG thêm vào _invalid_paths → lần sau retry.
            # - Ảnh lỗi thực sự (Exception): thêm vào _invalid_paths → gán -100.0 vĩnh viễn.
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=min(32, len(batch_paths)))
            futures_map = {executor.submit(self._load_image, p): p for p in batch_paths}
            done, not_done = concurrent.futures.wait(futures_map.keys(), timeout=60.0)

            valid_paths = []
            valid_images = []
            for fut in futures_map:
                p = futures_map[fut]
                if fut in not_done:
                    # NAS chậm / treo — bỏ qua lần này, không phạt vĩnh viễn
                    fut.cancel()
                    continue
                try:
                    img, ok = fut.result()
                    if ok:
                        valid_paths.append(p)
                        valid_images.append(img)
                    else:
                        self._invalid_paths.add(p)  # lỗi thực sự
                except Exception:
                    self._invalid_paths.add(p)

            executor.shutdown(wait=False, cancel_futures=True)

            if not valid_images:
                continue

            inputs = self.processor(images=valid_images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)

            vision_outputs = self.model.vision_model(pixel_values=pixel_values)
            image_embeds = vision_outputs.last_hidden_state.detach().cpu()  # [B, seq, dim]

            for p, emb in zip(valid_paths, image_embeds):
                self._embeds_cache[p] = emb
                self._cache_order.append(p)

            self._evict_if_needed()

    @torch.no_grad()
    def score(self, query_text: str, image_refs: List[Any]) -> torch.Tensor:
        """`image_refs` o day la danh sach duong dan anh (path)."""
        image_paths = image_refs

        # Buoc 1: dam bao image_embeds co san trong cache cho moi anh (chi tinh phan chua co)
        self._ensure_embeds_cached(image_paths)

        # Buoc 2: cross-attention (text_encoder + itm_head) -- luon tinh lai vi phu thuoc query_text
        all_scores = []
        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i:i + self.batch_size]
            # valid_paths: không lỗi vĩnh viễn VÀ đã có embed trong cache.
            # (Ảnh bị timeout NAS sẽ không có trong cache → bỏ qua lần này,
            #  lần sau _ensure_embeds_cached sẽ retry vì chưa có trong _invalid_paths)
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


def build_blip_backbone(device: str, model_id: str = "Salesforce/blip-itm-large-coco"):
    """Tiện ích dựng nhanh bộ 3: text_encoder, image_encoder, itm_scorer từ BLIP."""
    model = BlipForRetrieval.from_pretrained(model_id).to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    return (
        BlipTextEncoder(model, processor, device),
        BlipImageEncoder(model, processor, device),
        BlipITMScorer(model, processor, device),
    )