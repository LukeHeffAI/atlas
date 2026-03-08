"""Diversity modifier loader for synthetic image generation.

Loads per-dataset modifier dimensions (viewpoint, setting, lighting, etc.)
used to compose varied prompts for text-to-image generation.
"""

import json
import os
from typing import Dict, List


def load_diversity_modifiers(
    dataset_name: str,
    base_path: str = "data/diversity_modifiers"
) -> Dict[str, List[str]]:
    """Load diversity modifiers for a dataset.

    Tries to load dataset-specific modifiers first, falls back to default.json.

    Args:
        dataset_name: Name of the dataset (e.g., 'Cars', 'CIFAR10')
        base_path: Directory containing modifier JSON files

    Returns:
        Dictionary mapping dimension names to lists of modifier strings.
        E.g. {"viewpoint": ["front view", "side view"], "lighting": ["sunny", "overcast"]}
    """
    dataset_path = os.path.join(base_path, f"{dataset_name}.json")
    default_path = os.path.join(base_path, "default.json")

    if os.path.exists(dataset_path):
        path = dataset_path
    elif os.path.exists(default_path):
        path = default_path
    else:
        raise FileNotFoundError(
            f"No diversity modifiers found for '{dataset_name}' "
            f"and no default.json in {base_path}"
        )

    with open(path) as f:
        data = json.load(f)

    return data["dimensions"]
