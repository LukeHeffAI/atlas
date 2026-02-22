"""Loaders for text descriptions from various sources.

This module provides functionality to load text descriptions for datasets
from manual JSON files or LLM-generated files.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional


class TextDescriptionLoader:
    """Load and manage text descriptions for datasets.

    Text descriptions can come from:
    1. Manual curation (data/text_descriptions/manual/)
    2. LLM generation (data/text_descriptions/generated/)
    3. Template-based (using existing CLIP templates)

    Format: {"class_name": ["description1", "description2", ...], ...}
    """

    def __init__(self, base_path: str = "data/text_descriptions"):
        """Initialize the text description loader.

        Args:
            base_path: Base directory containing text descriptions
        """
        self.base_path = Path(base_path)
        if not self.base_path.exists():
            self.base_path.mkdir(parents=True, exist_ok=True)

    def load_descriptions(
        self,
        dataset_name: str,
        source: str = "manual",
        variant: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """Load text descriptions for a dataset.

        Args:
            dataset_name: Name of the dataset (e.g., "CIFAR10", "EuroSAT")
            source: Source of descriptions ("manual" or "generated")
            variant: For generated descriptions, specify the model (e.g., "gpt4o", "claude")

        Returns:
            Dictionary mapping class names to lists of descriptions.
            Format: {"airplane": ["a photo of an airplane", ...], ...}

        Raises:
            FileNotFoundError: If description file doesn't exist
        """
        if source == "manual":
            path = self.base_path / "manual" / f"{dataset_name}.json"
        elif source == "generated":
            if variant is None:
                raise ValueError("variant required for generated descriptions (e.g., 'gpt4o', 'claude')")
            path = self.base_path / "generated" / f"{dataset_name}_{variant}.json"
        else:
            raise ValueError(f"Unknown source: {source}. Must be 'manual' or 'generated'")

        if not path.exists():
            raise FileNotFoundError(
                f"Description file not found: {path}\n"
                f"Please create descriptions for {dataset_name} or use a different source."
            )

        with open(path) as f:
            data = json.load(f)

        # Handle different JSON formats
        if "descriptions" in data:
            # Format: {"dataset": "CIFAR10", "descriptions": {...}, "metadata": {...}}
            return data["descriptions"]
        else:
            # Format: {"class1": [...], "class2": [...]}
            return data

    def save_descriptions(
        self,
        descriptions: Dict[str, List[str]],
        dataset_name: str,
        source: str = "manual",
        variant: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """Save text descriptions to disk.

        Args:
            descriptions: Dictionary mapping class names to descriptions
            dataset_name: Name of the dataset
            source: "manual" or "generated"
            variant: For generated, model name (e.g., "gpt4o")
            metadata: Optional metadata to include (e.g., generation settings)
        """
        if source == "manual":
            path = self.base_path / "manual" / f"{dataset_name}.json"
        elif source == "generated":
            if variant is None:
                raise ValueError("variant required for generated descriptions")
            path = self.base_path / "generated" / f"{dataset_name}_{variant}.json"
        else:
            raise ValueError(f"Unknown source: {source}")

        path.parent.mkdir(parents=True, exist_ok=True)

        # Create structured format
        data = {
            "dataset": dataset_name,
            "source": source,
            "descriptions": descriptions
        }

        if metadata:
            data["metadata"] = metadata

        if variant:
            data["variant"] = variant

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def list_available(self, source: str = "manual") -> List[str]:
        """List available description files.

        Args:
            source: "manual" or "generated"

        Returns:
            List of dataset names with available descriptions
        """
        if source == "manual":
            dir_path = self.base_path / "manual"
        elif source == "generated":
            dir_path = self.base_path / "generated"
        else:
            raise ValueError(f"Unknown source: {source}")

        if not dir_path.exists():
            return []

        files = dir_path.glob("*.json")
        # Extract dataset names (remove .json extension and variant suffix for generated)
        names = []
        for f in files:
            name = f.stem
            if source == "generated" and "_" in name:
                # Remove variant suffix (e.g., "CIFAR10_gpt4o" -> "CIFAR10")
                name = name.rsplit("_", 1)[0]
            names.append(name)

        return sorted(set(names))
