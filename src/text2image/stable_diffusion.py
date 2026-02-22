"""Stable Diffusion backend for text-to-image generation.

This module implements the Stable Diffusion XL backend for generating synthetic images.
Optimized for RTX 4090 (24GB VRAM) with FP16, attention slicing, and efficient batching.
"""

from typing import List, Optional, Dict, Any
import torch
from PIL import Image
from .base import Text2ImageBackend
from dotenv import load_dotenv
import os

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

class StableDiffusionBackend(Text2ImageBackend):
    """Stable Diffusion XL backend for text-to-image generation.

    Recommended model: stabilityai/stable-diffusion-xl-base-1.0
    - Best quality/performance balance in 2026
    - 1024×1024 resolution
    - Fits RTX 4090 with FP16 and attention slicing

    Memory optimizations:
    - FP16 precision (torch.float16)
    - Attention slicing (reduces memory by ~50%)
    - Optional CPU offloading for very large batches
    - Small batch sizes (default: 4)
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize Stable Diffusion backend.

        Args:
            config: Configuration dictionary with keys:
                - model_id: Model identifier (default: "stabilityai/stable-diffusion-xl-base-1.0")
                - device: Device to use (default: "cuda")
                - guidance_scale: CFG scale (default: 7.5)
                - num_inference_steps: Denoising steps (default: 50)
                - height: Image height (default: 1024)
                - width: Image width (default: 1024)
                - enable_attention_slicing: Enable memory optimization (default: True)
                - use_fp16: Use half precision (default: True)
                - cpu_offload: Enable CPU offloading (default: False)
        """
        super().__init__(config)

        # Model configuration
        self.model_id = config.get('model_id', 'stabilityai/stable-diffusion-xl-base-1.0')
        self.guidance_scale = config.get('guidance_scale', 7.5)
        self.num_inference_steps = config.get('num_inference_steps', 50)
        self.height = config.get('height', 1024)
        self.width = config.get('width', 1024)
        self.enable_attention_slicing = config.get('enable_attention_slicing', True)
        self.use_fp16 = config.get('use_fp16', True)
        self.cpu_offload = config.get('cpu_offload', False)

        # Load model
        self._load_model()

    def _load_model(self):
        """Load Stable Diffusion model with optimizations."""
        from diffusers import StableDiffusionXLPipeline, DiffusionPipeline # type: ignore[attr-defined]

        print(f"Loading Stable Diffusion model: {self.model_id}")

        # Determine dtype
        dtype = torch.float16 if self.use_fp16 else torch.float32

        # Load pipeline
        try:
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            use_safetensors=True,
            variant="fp16" if self.use_fp16 else None
            )
        except:
            # Fallback: load without variant specification
            self.pipe = DiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                use_safetensors=True
            )

        # Move to device
        if not self.cpu_offload:
            self.pipe = self.pipe.to(self.device)

        # Enable memory optimizations
        if self.enable_attention_slicing:
            self.pipe.enable_attention_slicing()
            print("Enabled attention slicing for memory efficiency")

        if self.cpu_offload:
            self.pipe.enable_sequential_cpu_offload()
            print("Enabled CPU offloading for memory efficiency")

        # Enable xformers if available (significant speedup)
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("Enabled xformers memory efficient attention")
        except:
            print("xformers not available, using default attention")

        print(f"Stable Diffusion model loaded on {self.device}")

    def generate(
        self,
        prompts: List[str],
        num_images_per_prompt: int = 1,
        seed: Optional[int] = None,
        **kwargs
    ) -> List[Image.Image]:
        """Generate images from text prompts using Stable Diffusion.

        Args:
            prompts: List of text prompts
            num_images_per_prompt: Number of images to generate per prompt
            seed: Random seed for reproducibility
            **kwargs: Additional parameters:
                - guidance_scale: Override default CFG scale
                - num_inference_steps: Override default denoising steps
                - height: Override default height
                - width: Override default width
                - negative_prompt: Negative prompt for all generations

        Returns:
            List of PIL images
        """
        # Set seed for reproducibility
        if seed is not None:
            self.set_seed(seed)
        else:
            self.set_seed(self.seed)

        # Create generator for reproducibility
        generator = torch.Generator(device=self.device).manual_seed(self.seed)

        # Get parameters
        guidance_scale = kwargs.get('guidance_scale', self.guidance_scale)
        num_inference_steps = kwargs.get('num_inference_steps', self.num_inference_steps)
        height = kwargs.get('height', self.height)
        width = kwargs.get('width', self.width)
        negative_prompt = kwargs.get('negative_prompt', None)

        # Generate images
        try:
            output = self.pipe( # type: ignore[attr-defined]
                prompt=prompts,
                num_images_per_prompt=num_images_per_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                height=height,
                width=width,
                generator=generator,
                negative_prompt=negative_prompt
            )
            images = output.images # type: ignore[attr-defined]
        except Exception as e:
            print(f"Error generating images: {e}")
            # Clear CUDA cache and retry with smaller batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise

        # Clear cache after generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return images # type: ignore[attr-defined]

    def batch_generate(
        self,
        prompts: List[str],
        batch_size: int = 4,
        num_images_per_prompt: int = 1,
        output_dir: Optional[str] = None,
        force_regenerate: bool = False,
        **kwargs
    ) -> List[Image.Image]:
        """Generate images in batches with automatic memory management.

        Supports resuming partial generations: if output_dir is provided and
        some images already exist, only the missing images are generated.
        Set force_regenerate=True to overwrite all existing images.

        Args:
            prompts: List of text prompts
            batch_size: Number of prompts to process at once (default: 4 for RTX 4090)
            num_images_per_prompt: Images per prompt
            output_dir: If provided, save images to this directory
            force_regenerate: If True, delete existing images and regenerate all.
                If False (default), skip prompts whose images already exist.
            **kwargs: Additional generation parameters

        Returns:
            List of PIL images
        """
        import os

        total_count = len(prompts)

        # Determine which indices need generation
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

            if force_regenerate:
                # Delete existing images
                existing = [f for f in os.listdir(output_dir)
                            if f.startswith("image_") and f.endswith(".png")]
                for f in existing:
                    os.remove(os.path.join(output_dir, f))
                print(f"Deleted {len(existing)} existing images (force_regenerate=True)")
                pending_indices = list(range(total_count))
            else:
                # Find which indices already have images on disk
                pending_indices = [
                    i for i in range(total_count)
                    if not os.path.exists(os.path.join(output_dir, f"image_{i:05d}.png"))
                ]
                skipped = total_count - len(pending_indices)
                if skipped > 0:
                    print(f"Skipping {skipped}/{total_count} images that already exist")
        else:
            pending_indices = list(range(total_count))

        if not pending_indices:
            print(f"All {total_count} images already exist, nothing to generate")
            # Load and return existing images
            all_images = []
            if output_dir:
                for i in range(total_count):
                    img_path = os.path.join(output_dir, f"image_{i:05d}.png")
                    all_images.append(Image.open(img_path))
            return all_images

        # Build the subset of prompts to generate
        pending_prompts = [prompts[i] for i in pending_indices]
        num_batches = (len(pending_prompts) + batch_size - 1) // batch_size

        print(f"Generating {len(pending_prompts)}/{total_count} images "
              f"in {num_batches} batches of {batch_size}")

        generated = {}  # maps original index -> Image
        for batch_start in range(0, len(pending_prompts), batch_size):
            batch_prompts = pending_prompts[batch_start:batch_start+batch_size]
            batch_original_indices = pending_indices[batch_start:batch_start+batch_size]
            batch_num = batch_start // batch_size + 1

            print(f"Batch {batch_num}/{num_batches}: Generating {len(batch_prompts)} images...")

            try:
                batch_seed = self.seed + batch_original_indices[0]
                batch_images = self.generate(
                    batch_prompts,
                    num_images_per_prompt=num_images_per_prompt,
                    seed=batch_seed,
                    **kwargs
                )

                for idx, img in zip(batch_original_indices, batch_images):
                    generated[idx] = img
                    if output_dir:
                        img_path = os.path.join(output_dir, f"image_{idx:05d}.png")
                        img.save(img_path)
                        print(f"  Saved: {img_path}")

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"Warning: OOM in batch {batch_num}. Reducing batch size and retrying...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    smaller_batch_size = max(1, batch_size // 2)
                    for j in range(0, len(batch_prompts), smaller_batch_size):
                        sub_batch = batch_prompts[j:j+smaller_batch_size]
                        sub_indices = batch_original_indices[j:j+smaller_batch_size]
                        sub_seed = self.seed + sub_indices[0]
                        sub_images = self.generate(
                            sub_batch,
                            num_images_per_prompt=num_images_per_prompt,
                            seed=sub_seed,
                            **kwargs
                        )

                        for idx, img in zip(sub_indices, sub_images):
                            generated[idx] = img
                            if output_dir:
                                img_path = os.path.join(output_dir, f"image_{idx:05d}.png")
                                img.save(img_path)
                else:
                    raise

        print(f"Generated {len(generated)} new images")

        # Build full result list: load pre-existing from disk, use generated for new
        all_images = []
        for i in range(total_count):
            if i in generated:
                all_images.append(generated[i])
            elif output_dir:
                img_path = os.path.join(output_dir, f"image_{i:05d}.png")
                all_images.append(Image.open(img_path))

        return all_images

    @property
    def name(self) -> str:
        """Return the name of this backend."""
        return "stable_diffusion"
