"""
ChatIR Backbone Adapter cho NACIR++ — bản sửa lại
====================================================
Trước đây dùng transformers.BlipForImageTextRetrieval (naming vision_model.*)
nhưng checkpoint chatir_weights.ckpt là LAVIS/BLIP gốc (naming
visual_encoder.blocks.*, text_encoder.encoder.layer.*) -> load_state_dict
strict=False không khớp phần vision encoder, khiến kết quả sai.

Bản sửa này gọi thẳng blip_itm() từ chính codebase ChatIR/BLIP gốc — đúng
100% cách baselines.py của ChatIR load checkpoint, đảm bảo key khớp hoàn toàn.
"""

import os
import sys
import logging
from typing import Tuple, List, Union

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


# ============================================================
# TEXT ENCODER — wrap dialog_encoder logic từ baselines.py
# ============================================================

class TextEncoder:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        text = self.model.tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=200,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            text_output = self.model.text_encoder(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
                mode="text",
            )
            shift = self.model.text_proj(text_output.last_hidden_state[:, 0, :])

        return F.normalize(shift, dim=-1)


# ============================================================
# IMAGE ENCODER — wrap blip_project_img logic từ baselines.py
# ============================================================

class ImageEncoder:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ])
        self.preprocess = self._make_preprocess()

    def _make_preprocess(self):
        def _preprocess(image_path):
            raw = Image.open(image_path).convert("RGB")
            return self.transform(raw)
        return _preprocess

    def encode(self, images_batch):
        # images_batch: Tensor [B, 3, 224, 224] hoặc dict có "pixel_values"
        if isinstance(images_batch, dict):
            pixel_values = images_batch["pixel_values"].to(self.device)
        else:
            pixel_values = images_batch.to(self.device)

        with torch.no_grad():
            embeds = self.model.visual_encoder(pixel_values)
            projection = self.model.vision_proj(embeds[:, 0, :])

        return F.normalize(projection, dim=-1)


# ============================================================
# ITM SCORER (dùng cross-attention thật của BLIP, không phải cosine)
# ============================================================

class ITMScorer:
    """
    Dùng head ITM gốc của BLIP (model.itm_head trên cross-attention output),
    khác với bản cũ chỉ tính cosine similarity giữa 2 global vector.
    Nếu bạn chỉ cần base retrieval (không rerank ITM thật), có thể không
    dùng scorer này (image_scorer=None trong pipeline).
    """

    def __init__(self, model, device, image_transform):
        self.model = model
        self.device = device
        self.transform = image_transform

    @property
    def device_(self):
        return next(self.model.parameters()).device

    def score(self, query_text: Union[str, List[str]], image_refs: List) -> torch.Tensor:
        if isinstance(query_text, list):
            query_text = query_text[0]

        # Load & preprocess images
        if len(image_refs) > 0 and isinstance(image_refs[0], str):
            imgs = []
            for ref in image_refs:
                try:
                    raw = Image.open(ref).convert("RGB")
                    imgs.append(self.transform(raw))
                except Exception as e:
                    logging.warning(f"Failed to load image {ref}: {e}")
                    imgs.append(torch.zeros(3, 224, 224))
            pixel_values = torch.stack(imgs).to(self.device)
        else:
            pixel_values = image_refs.to(self.device)

        with torch.no_grad():
            image_embeds = self.model.visual_encoder(pixel_values)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.device)

            text = self.model.tokenizer(
                [query_text] * pixel_values.size(0),
                padding="longest",
                truncation=True,
                max_length=200,
                return_tensors="pt",
            ).to(self.device)

            output = self.model.text_encoder(
                text.input_ids,
                attention_mask=text.attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            itm_logits = self.model.itm_head(output.last_hidden_state[:, 0, :])
            scores = F.softmax(itm_logits, dim=1)[:, 1]  # xác suất "match"

        return scores.to(self.device_)


# ============================================================
# BUILD BACKBONE
# ============================================================

def build_chatir_backbone(
    device: str,
    chatir_repo_dir: str,
    finetuned_ckpt_path: str,
) -> Tuple:
    """
    chatir_repo_dir: path tới thư mục chứa folder BLIP/, ví dụ
        "/AIClub_NAS/core_baotg/thudnm/ChatIR"
    finetuned_ckpt_path: path tới chatir_weights.ckpt
        (khuyến nghị copy về local disk trước, vd /tmp/chatir_weights.ckpt,
        để tránh NAS I/O chậm)
    """
    logger = logging.getLogger(__name__)

    # Đảm bảo import được đúng module BLIP gốc của ChatIR repo.
    # Cần CẢ 2 path:
    #  - chatir_repo_dir: để import được "BLIP.models.blip_itm"
    #  - chatir_repo_dir/BLIP: vì bên trong blip_itm.py tự import
    #    "from models.med import ..." (import tương đối kiểu chạy trực tiếp
    #    từ trong thư mục BLIP/, không phải theo package BLIP.models.*)
    blip_subdir = os.path.join(chatir_repo_dir, "BLIP")
    for p in (chatir_repo_dir, blip_subdir):
        if p not in sys.path:
            sys.path.insert(0, p)

    from BLIP.models.blip_itm import blip_itm  # noqa: E402

    med_config_path = os.path.join(chatir_repo_dir, "BLIP", "configs", "med_config.json")

    logger.info(f"Loading BLIP-ITM gốc từ checkpoint: {finetuned_ckpt_path}")
    model = blip_itm(
        pretrained=finetuned_ckpt_path,
        med_config=med_config_path,
        image_size=224,
        vit="base",
    )

    model = model.to(device).eval()

    text_encoder = TextEncoder(model, device)
    image_encoder = ImageEncoder(model, device)
    itm_scorer = ITMScorer(model, device, image_encoder.transform)

    return text_encoder, image_encoder, itm_scorer