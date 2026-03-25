"""Hypernetwork architectures for text-based LoRA adaptation.

This module provides hypernetworks that predict aTLAS coefficients or LoRA weights
from text descriptions, enabling zero-shot and few-shot adaptation without image data.
"""

from .base import BaseHypernetwork
from .text_to_coef import TextToCoefHypernetwork
from .multimodal_to_coef import MultiModalHypernetwork

__all__ = [
    "BaseHypernetwork",
    "TextToCoefHypernetwork",
    "MultiModalHypernetwork",
]
