"""Generate synthetic images from text descriptions for datasets.

This script uses text-to-image models (Stable Diffusion XL, DALL-E) to generate
synthetic images for dataset classes based on text descriptions.

Usage:
    python src/generate_synthetic_data.py \
        --dataset CIFAR10 \
        --text-source manual \
        --t2i-backend stable_diffusion \
        --num-images-per-class 100 \
        --output-dir data/synthetic_images \
        --seed 42
"""

import os
import random
import argparse
from pathlib import Path
from typing import Dict, List
import yaml
from tqdm import tqdm

from text_descriptions.loaders import TextDescriptionLoader
from text2image.registry import get_t2i_backend


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate synthetic images from text descriptions"
    )

    # Dataset and text configuration
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., CIFAR10, EuroSAT)"
    )
    parser.add_argument(
        "--text-source",
        type=str,
        default="manual",
        choices=["manual", "generated", "templates"],
        help="Source of text descriptions"
    )
    parser.add_argument(
        "--text-variant",
        type=str,
        default=None,
        help="Variant for generated descriptions (e.g., gpt4o, claude)"
    )

    # T2I backend configuration
    parser.add_argument(
        "--t2i-backend",
        type=str,
        default="stable_diffusion",
        choices=["stable_diffusion", "sdxl", "dalle", "dall-e"],
        help="Text-to-image backend to use"
    )
    parser.add_argument(
        "--t2i-model",
        type=str,
        default=None,
        help="Specific model ID (e.g., stabilityai/stable-diffusion-xl-base-1.0)"
    )
    parser.add_argument(
        "--t2i-config",
        type=str,
        default=None,
        help="Path to T2I configuration YAML file"
    )

    # Generation parameters
    parser.add_argument(
        "--num-images-per-class",
        type=int,
        default=100,
        help="Number of images to generate per class"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for generation (adjust based on GPU memory)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/synthetic_images",
        help="Output directory for generated images"
    )
    parser.add_argument(
        "--classes",
        type=str,
        nargs="+",
        default=None,
        help="Specific classes to generate (if None, generates for all classes)"
    )

    # Resume / deduplication flags
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Regenerate all images even if they already exist"
    )
    parser.add_argument(
        "--shuffle-existing",
        action="store_true",
        help=(
            "When fewer images are requested than already exist, randomly sample from the existing images instead of taking the first N"
        )
    )

    # Device configuration
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)"
    )

    return parser.parse_args()


def load_t2i_config(backend_name: str, config_path: str = None) -> Dict:
    """Load T2I configuration from YAML file.

    Args:
        backend_name: Name of the T2I backend
        config_path: Path to config file (if None, uses default)

    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Use default config
        default_configs = {
            "stable_diffusion": "configs/text2image/stable_diffusion.yaml",
            "sdxl": "configs/text2image/stable_diffusion.yaml",
            "dalle": "configs/text2image/dalle.yaml",
            "dall-e": "configs/text2image/dalle.yaml",
        }
        config_path = default_configs.get(backend_name.lower())

    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        print(f"Loaded T2I config from: {config_path}")
        return config
    else:
        print(f"Config file not found: {config_path}, using defaults")
        return {}


def generate_prompts_for_class(
    class_name: str,
    descriptions: List[str],
    num_images: int
) -> List[str]:
    """Generate prompts for a class by cycling through descriptions.

    Args:
        class_name: Name of the class
        descriptions: List of text descriptions
        num_images: Number of images to generate

    Returns:
        List of prompts (length = num_images)
    """
    if not descriptions:
        # Fallback to simple template
        descriptions = [f"a photo of a {class_name}"]

    prompts = []
    for i in range(num_images):
        # Cycle through descriptions
        desc = descriptions[i % len(descriptions)]
        prompts.append(desc)

    return prompts


def main():
    args = parse_arguments()

    print("=" * 80)
    print(f"Generating synthetic images for {args.dataset}")
    print("=" * 80)

    # Load text descriptions
    print(f"\nLoading text descriptions (source: {args.text_source})...")
    loader = TextDescriptionLoader()

    try:
        if args.text_source == "templates":
            from src.text_descriptions.templates import generate_template_descriptions
            # We need to get class names first - load from a sample dataset
            # For now, we'll require --classes argument for template mode
            if not args.classes:
                raise ValueError(
                    "--classes argument required when using template mode"
                )
            descriptions = generate_template_descriptions(
                args.dataset,
                args.classes
            )
        else:
            descriptions = loader.load_descriptions(
                args.dataset,
                source=args.text_source,
                variant=args.text_variant
            )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please create text descriptions first or use --text-source templates")
        return

    # Filter to specific classes if requested
    if args.classes:
        descriptions = {k: v for k, v in descriptions.items() if k in args.classes}

    print(f"Loaded descriptions for {len(descriptions)} classes")
    for class_name, descs in list(descriptions.items())[:3]:
        print(f"  {class_name}: {len(descs)} descriptions")

    # Load T2I backend configuration
    print(f"\nInitializing T2I backend: {args.t2i_backend}...")
    config = load_t2i_config(args.t2i_backend, args.t2i_config)

    # Override config with command-line arguments
    if args.t2i_model:
        config['model_id'] = args.t2i_model
    config['device'] = args.device
    config['seed'] = args.seed

    # Create backend
    backend = get_t2i_backend(args.t2i_backend, config)

    # Setup output directory
    output_base = Path(args.output_dir) / backend.name / args.dataset
    print(f"\nOutput directory: {output_base}")

    # Generate images for each class
    print(f"\nGenerating {args.num_images_per_class} images per class...")
    total_images = len(descriptions) * args.num_images_per_class
    print(f"Total images to generate: {total_images}")

    if args.t2i_backend.lower() in ["dalle", "dall-e"]:
        cost_estimate_standard = total_images * config['pricing']['standard']
        cost_estimate_hd = total_images * config['pricing']['hd']
        print(f"Estimated cost:\n\t- ${cost_estimate_standard:.2f} (standard quality)\n\t- ${cost_estimate_hd:.2f} (hd quality)")

    for class_name, class_descriptions in tqdm(descriptions.items(), desc="Classes"):
        print(f"\nGenerating images for class: {class_name}")

        # Create class directory
        class_dir = output_base / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        # Generate prompts
        prompts = generate_prompts_for_class(
            class_name,
            class_descriptions,
            args.num_images_per_class
        )

        # Generate images in batches
        images = backend.batch_generate(
            prompts=prompts,
            batch_size=args.batch_size
        )

        # Save images
        for i, image in enumerate(images):
            image_path = class_dir / f"{i:05d}.png"
            image.save(image_path)

        print(f"  Saved {len(images)} images to {class_dir}")

    print("\n" + "=" * 80)
    print("Synthetic image generation complete!")
    print(f"Output directory: {output_base}")
    print(f"Total classes: {len(descriptions)}")
    print(f"Total images: {total_images}")
    print("=" * 80)


if __name__ == "__main__":
    main()
