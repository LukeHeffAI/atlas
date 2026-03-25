"""Meta-learning infrastructure for text-to-coefficient hypernetworks.

This module provides utilities for episode-based meta-training of hypernetworks
across multiple tasks, enabling zero-shot and few-shot adaptation on new tasks.
"""

from .multimodal_sampler import MultiModalEpisodeSampler

__all__ = [
    "MultiModalEpisodeSampler",
]
