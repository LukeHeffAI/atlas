"""Base interface for hypernetworks.

This module defines the abstract base class that all hypernetworks must implement.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, List, Optional


class BaseHypernetwork(nn.Module, ABC):
    """Base class for hypernetworks that predict model parameters from text.

    Hypernetworks are meta-learned across multiple tasks and can then be used
    for zero-shot or few-shot adaptation on new tasks by predicting parameters
    (e.g., aTLAS coefficients or LoRA weights) from text descriptions alone.
    """

    def __init__(self):
        """Initialize base hypernetwork."""
        super().__init__()

    @abstractmethod
    def forward(self, text_descriptions: List[str]) -> torch.Tensor:
        """Forward pass: predict parameters from text descriptions.

        Args:
            text_descriptions: List of text descriptions (batch_size strings)

        Returns:
            Predicted parameters tensor
        """
        raise NotImplementedError

    @abstractmethod
    def predict_for_dataset(
        self,
        dataset_descriptions: Dict[str, List[str]],
        aggregate: str = "mean"
    ) -> torch.Tensor:
        """Predict parameters for an entire dataset.

        Args:
            dataset_descriptions: Dict mapping class names to text descriptions
                Format: {"class1": ["desc1", "desc2", ...], ...}
            aggregate: How to aggregate multiple descriptions per class
                Options: "mean", "max", "median"

        Returns:
            Single parameter tensor for the dataset
        """
        raise NotImplementedError

    def save(self, path: str):
        """Save hypernetwork to disk.

        Args:
            path: Path to save checkpoint
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_config': self.get_config(),
        }, path)
        print(f"Saved hypernetwork to: {path}")

    @classmethod
    def load(cls, path: str, device: str = "cuda"):
        """Load hypernetwork from disk.

        Args:
            path: Path to checkpoint
            device: Device to load model on

        Returns:
            Loaded hypernetwork instance
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = checkpoint['model_config']

        # Create instance with saved config
        instance = cls(**config)
        instance.load_state_dict(checkpoint['model_state_dict'])
        instance = instance.to(device)

        print(f"Loaded hypernetwork from: {path}")
        return instance

    @abstractmethod
    def get_config(self) -> Dict:
        """Get configuration dictionary for saving/loading.

        Returns:
            Configuration dictionary
        """
        raise NotImplementedError
