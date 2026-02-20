"""DALL-E backend for text-to-image generation via OpenAI API.

This module implements the DALL-E 3 backend for generating high-quality synthetic images.
Pricing (2026): $0.040 per 1024×1024 image.
"""

import os
from typing import List, Optional, Dict, Any
import time
from PIL import Image
import requests
from io import BytesIO
from .base import Text2ImageBackend


class DalleBackend(Text2ImageBackend):
    """DALL-E 3 backend for text-to-image generation via OpenAI API.

    Pricing (2026):
    - Standard 1024×1024: $0.040 per image
    - HD 1024×1792 or 1792×1024: $0.120 per image

    Note: OpenAI has recently released GPT Image 1/1 Mini models, but DALL-E 3 is proven and stable.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize DALL-E backend.

        Args:
            config: Configuration dictionary with keys:
                - api_key: OpenAI API key (or read from OPENAI_API_KEY env var)
                - model: Model to use (default: "dall-e-3")
                - size: Image size (default: "1024x1024", options: "1024x1024", "1024x1792", "1792x1024")
                - quality: Quality level (default: "standard", options: "standard", "hd")
                - style: Style (default: "natural", options: "natural", "vivid")
        """
        super().__init__(config)

        # API configuration
        self.api_key = config.get('api_key') or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key in config."
            )

        self.model = config.get('model', 'dall-e-3')
        self.size = config.get('size', '1024x1024')
        self.quality = config.get('quality', 'standard')
        self.style = config.get('style', 'natural')

        # Rate limiting
        self.rate_limit_delay = config.get('rate_limit_delay', 1.0)  # seconds between API calls

        # Initialize OpenAI client
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Install with: pip install openai"
            )

        print(f"Initialized DALL-E backend (model: {self.model}, size: {self.size}, quality: {self.quality})")

    def generate(
        self,
        prompts: List[str],
        num_images_per_prompt: int = 1,
        seed: Optional[int] = None,
        **kwargs
    ) -> List[Image.Image]:
        """Generate images from text prompts using DALL-E.

        Note: DALL-E 3 only supports n=1 (one image per prompt) due to API limitations.
        If num_images_per_prompt > 1, multiple API calls will be made.

        Args:
            prompts: List of text prompts
            num_images_per_prompt: Number of images to generate per prompt
            seed: Not used for DALL-E (API doesn't support seed control)
            **kwargs: Additional parameters:
                - size: Override default size
                - quality: Override default quality
                - style: Override default style

        Returns:
            List of PIL images
        """
        size = kwargs.get('size', self.size)
        quality = kwargs.get('quality', self.quality)
        style = kwargs.get('style', self.style)

        all_images = []

        for prompt in prompts:
            for _ in range(num_images_per_prompt):
                try:
                    # Call DALL-E API
                    response = self.client.images.generate(
                        model=self.model,
                        prompt=prompt,
                        size=size,
                        quality=quality,
                        style=style,
                        n=1  # DALL-E 3 only supports n=1
                    )

                    # Download image from URL
                    image_url = response.data[0].url
                    image_response = requests.get(image_url)
                    image = Image.open(BytesIO(image_response.content))

                    all_images.append(image)

                    # Rate limiting
                    time.sleep(self.rate_limit_delay)

                except Exception as e:
                    print(f"Error generating image for prompt '{prompt}': {e}")
                    # Continue with other prompts
                    continue

        return all_images

    def batch_generate(
        self,
        prompts: List[str],
        batch_size: int = 1,  # DALL-E API is sequential, but keep interface consistent
        num_images_per_prompt: int = 1,
        output_dir: Optional[str] = None,
        **kwargs
    ) -> List[Image.Image]:
        """Generate images sequentially (DALL-E doesn't support true batching).

        Args:
            prompts: List of text prompts
            batch_size: Ignored for DALL-E (kept for interface consistency)
            num_images_per_prompt: Images per prompt
            output_dir: If provided, save images to this directory
            **kwargs: Additional generation parameters

        Returns:
            List of PIL images
        """
        print(f"Generating {len(prompts)} images via DALL-E API...")
        print(f"Estimated cost: ${len(prompts) * num_images_per_prompt * 0.040:.2f} (standard quality)")

        all_images = []

        for i, prompt in enumerate(prompts):
            print(f"Prompt {i+1}/{len(prompts)}: {prompt[:60]}...")

            for j in range(num_images_per_prompt):
                try:
                    # Generate single image
                    response = self.client.images.generate(
                        model=self.model,
                        prompt=prompt,
                        size=kwargs.get('size', self.size),
                        quality=kwargs.get('quality', self.quality),
                        style=kwargs.get('style', self.style),
                        n=1
                    )

                    # Download image
                    image_url = response.data[0].url
                    image_response = requests.get(image_url)
                    image = Image.open(BytesIO(image_response.content))

                    all_images.append(image)

                    # Save if output_dir provided
                    if output_dir:
                        import os
                        os.makedirs(output_dir, exist_ok=True)
                        img_idx = i * num_images_per_prompt + j
                        img_path = os.path.join(output_dir, f"image_{img_idx:05d}.png")
                        image.save(img_path)
                        print(f"  Saved: {img_path}")

                    # Rate limiting
                    time.sleep(self.rate_limit_delay)

                except Exception as e:
                    print(f"  Error: {e}")
                    continue

        print(f"Generated {len(all_images)} images total")
        return all_images

    @property
    def name(self) -> str:
        """Return the name of this backend."""
        return "dalle"
