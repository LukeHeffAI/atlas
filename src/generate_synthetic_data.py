"""Generate synthetic images from text descriptions for datasets.

This script uses text-to-image models (Stable Diffusion XL, DALL-E) to generate
synthetic images for dataset classes based on text descriptions.

Usage:
    python src/generate_synthetic_data.py \
        --datasets CIFAR10 EuroSAT DTD \
        --text-source manual \
        --t2i-backend stable_diffusion \
        --num-images-per-class 100 \
        --output-dir data/synthetic_images \
        --seed 42
"""

import os
import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional
import yaml
from tqdm import tqdm

from text_descriptions.loaders import TextDescriptionLoader
from text2image.registry import get_t2i_backend
from diversity_modifiers import load_diversity_modifiers


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate synthetic images from text descriptions"
    )

    # Dataset and text configuration
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        required=True,
        metavar="DATASET",
        help="One or more dataset names (e.g., CIFAR10 EuroSAT DTD)"
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

    # Diversity modifiers
    parser.add_argument(
        "--diversity-modifiers-dir",
        type=str,
        default="data/diversity_modifiers",
        help="Directory containing diversity modifier JSON files"
    )
    parser.add_argument(
        "--no-diversity-modifiers",
        action="store_true",
        help="Disable diversity modifiers (use legacy cycling behavior)"
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
    num_images: int,
    modifiers: Optional[Dict[str, List[str]]] = None,
    seed: int = 42
) -> List[str]:
    """Generate diverse prompts for a class by combining descriptions with modifiers.

    Each prompt is composed by cycling through base descriptions and randomly
    sampling one modifier per dimension, producing unique prompts even when
    num_images >> len(descriptions).

    Args:
        class_name: Name of the class
        descriptions: List of text descriptions
        num_images: Number of images to generate
        modifiers: Dict mapping dimension names to lists of modifier strings.
                   If None, falls back to a simple quality suffix.
        seed: Random seed for reproducible modifier sampling

    Returns:
        List of prompts (length = num_images)
    """
    if not descriptions:
        descriptions = [f"a photo of a {class_name}"]

    rng = random.Random(seed)

    prompts = []
    for i in range(num_images):
        desc = descriptions[i % len(descriptions)]

        if modifiers:
            # Sample one modifier per dimension
            parts = [desc]
            for dim_name, dim_values in modifiers.items():
                parts.append(rng.choice(dim_values))
            prompt = ", ".join(parts) + ", hyper-realistic, 4k resolution"
        else:
            prompt = f"{desc}, hyper-realistic, 4k resolution"

        prompts.append(prompt)

    return prompts


def generate_for_dataset(args, dataset_name: str, backend):
    """Generate synthetic images for a single dataset.

    Args:
        args: Parsed CLI arguments
        dataset_name: Name of the dataset to process
        backend: Initialized T2I backend (shared across datasets)
    """
    print("=" * 80)
    print(f"Generating synthetic images for {dataset_name}")
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
                dataset_name,
                args.classes
            )
        else:
            descriptions = loader.load_descriptions(
                dataset_name,
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

    # Load diversity modifiers
    modifiers = None
    if not args.no_diversity_modifiers:
        try:
            modifiers = load_diversity_modifiers(
                dataset_name, base_path=args.diversity_modifiers_dir
            )
            dim_summary = {k: len(v) for k, v in modifiers.items()}
            print(f"Loaded diversity modifiers: {dim_summary}")
        except FileNotFoundError:
            print("No diversity modifiers found, using legacy prompt style")

    # Setup output directory
    output_base = Path(args.output_dir) / backend.name / dataset_name
    print(f"\nOutput directory: {output_base}")

    # Pre-scan existing images to compute actual generation counts and cost
    class_existing = {}
    class_to_generate = {}
    for class_name in descriptions:
        class_dir = output_base / class_name
        existing = sorted(class_dir.glob("*.png")) if class_dir.exists() else []
        class_existing[class_name] = existing
        if args.force_regenerate:
            class_to_generate[class_name] = args.num_images_per_class
        else:
            class_to_generate[class_name] = max(0, args.num_images_per_class - len(existing))

    actual_total = sum(class_to_generate.values())
    skipped_total = sum(
        min(len(class_existing[c]), args.num_images_per_class)
        for c in descriptions
        if not args.force_regenerate
    )

    print(f"Images already on disk (will skip): {skipped_total}")
    print(f"Images to generate: {actual_total}")

    if args.t2i_backend.lower() in ["dalle", "dall-e"]:
        config = load_t2i_config(args.t2i_backend, args.t2i_config)
        cost_estimate_standard = actual_total * config['pricing']['standard']
        cost_estimate_hd = actual_total * config['pricing']['hd']
        print(f"Estimated cost:\n\t- ${cost_estimate_standard:.2f} (standard quality)\n\t- ${cost_estimate_hd:.2f} (hd quality)")

    for class_name, class_descriptions in tqdm(descriptions.items(), desc="Classes"):
        class_dir = output_base / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        existing = class_existing[class_name]
        n_needed = class_to_generate[class_name]

        if args.force_regenerate:
            # Remove existing images and regenerate from scratch
            for p in existing:
                p.unlink()
            existing = []

        if len(existing) >= args.num_images_per_class:
            # More images exist than requested — no generation needed.
            # Selection of first-N vs random subset is handled at load time via SyntheticDataset.
            print(f"\n  {class_name}: {len(existing)} images exist, using {args.num_images_per_class} "
                  f"({'random sample' if args.shuffle_existing else 'first N'} at load time)")
            continue

        if n_needed == 0:
            print(f"\n  {class_name}: already has {len(existing)} images, skipping")
            continue

        print(f"\nGenerating {n_needed} images for class: {class_name} "
              f"({len(existing)} already exist, {args.num_images_per_class} requested)")

        # Determine the starting index for new images (continue numbering after existing)
        next_index = len(existing) if not args.force_regenerate else 0

        # Generate prompts for only the missing images
        # Use a per-class seed derived from the base seed and class name
        class_seed = args.seed + hash(class_name) % (2**31)
        prompts = generate_prompts_for_class(
            class_name,
            class_descriptions,
            n_needed,
            modifiers=modifiers,
            seed=class_seed
        )

        # Generate images in batches.
        original_seed = backend.seed
        backend.seed = args.seed + next_index
        images = backend.batch_generate(
            prompts=prompts,
            batch_size=args.batch_size
        )
        backend.seed = original_seed

        # Save images, starting after the last existing index
        for i, image in enumerate(images):
            image_path = class_dir / f"{next_index + i:05d}.png"
            image.save(image_path)

        print(f"  Saved {len(images)} images to {class_dir}")

    print("\n" + "=" * 80)
    print(f"Synthetic image generation complete for {dataset_name}!")
    print(f"Output directory: {output_base}")
    print(f"Total classes: {len(descriptions)}")
    print(f"Total images generated: {actual_total}")
    print("=" * 80)


def main():
    args = parse_arguments()

    # Initialize backend once (loading SD/SDXL models is expensive)
    config = load_t2i_config(args.t2i_backend, args.t2i_config)
    if args.t2i_model:
        config['model_id'] = args.t2i_model
    config['device'] = args.device
    config['seed'] = args.seed
    print(f"\nInitializing T2I backend: {args.t2i_backend}...")
    backend = get_t2i_backend(args.t2i_backend, config)

    for dataset_name in args.datasets:
        generate_for_dataset(args, dataset_name, backend)


if __name__ == "__main__":
    main()
