"""Template-based text descriptions using existing CLIP templates.

This module provides functionality to generate text descriptions using
the templates already present in the aTLAS codebase.
"""

from typing import List, Dict, Callable
from src.datasets.templates import get_templates


def generate_template_descriptions(
    dataset_name: str,
    class_names: List[str]
) -> Dict[str, List[str]]:
    """Generate text descriptions using existing CLIP templates.

    Args:
        dataset_name: Name of the dataset
        class_names: List of class names

    Returns:
        Dictionary mapping class names to template-based descriptions
    """
    # Get templates for this dataset
    try:
        templates = get_templates(dataset_name)
    except AssertionError:
        # Fallback to generic templates if dataset-specific ones unavailable
        templates = get_default_templates()

    descriptions = {}
    for class_name in class_names:
        # Apply all templates to this class name
        descriptions[class_name] = [template(class_name) for template in templates]

    return descriptions


def get_default_templates() -> List[Callable]:
    """Get default CLIP-style templates.

    Returns:
        List of template functions
    """
    return [
        lambda c: f"a photo of a {c}.",
        lambda c: f"a photo of {c}.",
        lambda c: f"a rendering of a {c}.",
        lambda c: f"a cropped photo of a {c}.",
        lambda c: f"a photo of a clean {c}.",
        lambda c: f"a photo of a dirty {c}.",
        lambda c: f"a bright photo of a {c}.",
        lambda c: f"a dark photo of a {c}.",
        lambda c: f"a photo of my {c}.",
        lambda c: f"a photo of the cool {c}.",
        lambda c: f"a close-up photo of a {c}.",
        lambda c: f"a good photo of a {c}.",
        lambda c: f"a photo of the large {c}.",
        lambda c: f"a photo of the small {c}.",
    ]


# CLI interface for generating template-based descriptions
if __name__ == "__main__":
    import argparse
    from src.text_descriptions.loaders import TextDescriptionLoader

    parser = argparse.ArgumentParser(description="Generate template-based descriptions")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--classes", type=str, nargs="+", required=True, help="Class names")
    parser.add_argument("--output-dir", type=str, default="data/text_descriptions", help="Output directory")

    args = parser.parse_args()

    # Generate descriptions
    print(f"Generating template-based descriptions for {args.dataset}...")
    descriptions = generate_template_descriptions(
        dataset_name=args.dataset,
        class_names=args.classes
    )

    # Save to file
    loader = TextDescriptionLoader(base_path=args.output_dir)
    metadata = {
        "source": "templates",
        "num_templates": len(next(iter(descriptions.values()))),
        "num_classes": len(args.classes)
    }
    loader.save_descriptions(
        descriptions=descriptions,
        dataset_name=args.dataset,
        source="manual",
        metadata=metadata
    )

    print(f"Descriptions saved to {args.output_dir}/manual/{args.dataset}.json")
    print(f"Total classes: {len(descriptions)}")
    print(f"Total descriptions: {sum(len(v) for v in descriptions.values())}")
