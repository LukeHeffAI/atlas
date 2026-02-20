"""Base interface for text-to-image generation backends.

This module defines the abstract interface that all T2I backends must implement.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import torch
from PIL import Image


class Text2ImageBackend(ABC):
    """Base interface for text-to-image generation backends.

    All T2I backends (Stable Diffusion, DALL-E, Imagen, etc.) must implement
    this interface to ensure consistent usage across the codebase.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the T2I backend.

        Args:
            config: Configuration dictionary containing backend-specific params
                    Common keys:
                    - model_id: Model identifier (e.g., "stabilityai/stable-diffusion-xl-base-1.0")
                    - device: Device to use ("cuda" or "cpu")
                    - guidance_scale: CFG scale for generation
                    - num_inference_steps: Number of denoising steps
                    - seed: Random seed for reproducibility
        """
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = config.get('seed', 42)

    @abstractmethod
    def generate(
        self,
        prompts: List[str],
        num_images_per_prompt: int = 1,
        seed: Optional[int] = None,
        **kwargs
    ) -> List[Image.Image]:
        """Generate images from text prompts.

        Args:
            prompts: List of text prompts
            num_images_per_prompt: Number of images to generate per prompt
            seed: Random seed for reproducibility (overrides default)
            **kwargs: Backend-specific generation parameters

        Returns:
            List of PIL images (length = len(prompts) * num_images_per_prompt)
        """
        raise NotImplementedError

    def batch_generate(
        self,
        prompts: List[str],
        batch_size: int = 4,
        num_images_per_prompt: int = 1,
        output_dir: Optional[str] = None, #TODO: why would output directory be optional?
        **kwargs
    ) -> List[Image.Image]:
        """Generate images in batches for efficiency.

        This method handles batching automatically and optionally saves images to disk.

        Args:
            prompts: List of text prompts
            batch_size: Number of prompts to process at once
            num_images_per_prompt: Images per prompt
            output_dir: If provided, save images to this directory
            **kwargs: Backend-specific parameters

        Returns:
            List of PIL images
        """
        all_images = []

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            batch_images = self.generate(
                batch_prompts,
                num_images_per_prompt=num_images_per_prompt,
                **kwargs
            )
            all_images.extend(batch_images)

            # Optional: save images to disk
            if output_dir:
                import os
                os.makedirs(output_dir, exist_ok=True)
                for j, img in enumerate(batch_images):
                    img_path = os.path.join(output_dir, f"image_{i+j:05d}.png")
                    img.save(img_path)

        return all_images

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the backend (e.g., 'stable_diffusion', 'dalle')."""
        raise NotImplementedError

    def set_seed(self, seed: int):
        """Set random seed for reproducible generation.

        Args:
            seed: Random seed
        """
        self.seed = seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
