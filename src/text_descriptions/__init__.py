"""Text description management for text-based LoRA adaptation.

This module provides utilities for loading, generating, and managing text descriptions
of datasets for use in synthetic image generation and hypernetwork-based adaptation.
"""

from .loaders import TextDescriptionLoader
from .generators import OpenAIDescriptionGenerator, ClaudeDescriptionGenerator

__all__ = [
    "TextDescriptionLoader",
    "OpenAIDescriptionGenerator",
    "ClaudeDescriptionGenerator",
]
