"""Backward-compatible import path for the paper BLIP text adapter."""

from .adapters.plugir_blip import BLIPTextEncoder, NormalizedBLIP, load_blip_text_encoder

__all__ = ["BLIPTextEncoder", "NormalizedBLIP", "load_blip_text_encoder"]
