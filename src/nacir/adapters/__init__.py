"""Adapter implementations for connecting NACIR to external retrieval pipelines."""

from .plugir_blip import BLIPTextEncoder, NormalizedBLIP, load_blip_text_encoder

__all__ = ["BLIPTextEncoder", "NormalizedBLIP", "load_blip_text_encoder"]
