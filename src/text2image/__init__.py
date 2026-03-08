"""Text-to-image generation backends for synthetic data generation.

This module provides interfaces and implementations for various text-to-image
models including Stable Diffusion XL, DALL-E, and others.
"""

from .base import Text2ImageBackend
from .registry import get_t2i_backend, list_t2i_backends

__all__ = [
    "Text2ImageBackend",
    "get_t2i_backend",
    "list_t2i_backends",
]
