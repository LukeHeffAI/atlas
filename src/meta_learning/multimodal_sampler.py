"""Multi-modal episode sampler for meta-training.

This module provides an episode sampler that produces paired (text, image)
episodes for meta-training the MultiModalHypernetwork.  Each episode
corresponds to one dataset (task) and contains:

1. **Text descriptions** for all classes in the dataset.
2. **Support images** (N-way K-shot) randomly sampled from the training split.

The sampler supports variable shot counts per episode and is designed to be
used inside the outer meta-training loop of ``learn_multimodal_to_coef.py``.

Design notes:
    - Datasets are lazily loaded and cached after first access to amortise
      the cost of repeated episode sampling from the same task.
    - Class-balanced sampling ensures every class contributes equally.
    - A fixed random seed per episode is optional (for reproducibility).
"""

import random
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Subset

from datasets.registry import get_dataset
from utils import load_text_descriptions


class MultiModalEpisodeSampler:
    """Sample multi-modal episodes (text + images) for meta-training.

    Each call to :meth:`sample_episode` returns the data needed for one
    training step of the multi-modal hypernetwork: text descriptions for the
    sampled task and a small support set of images.

    Args:
        datasets: List of dataset names available for sampling.
        num_shots: Number of support images per class (K in K-shot).
        args: Parsed CLI arguments (used for data paths, text source, etc.).
        preprocess: Image preprocessing transform (from the CLIP model).
        variable_shots: If True, randomly sample num_shots from [1, num_shots]
            per episode to improve robustness across shot counts.
    """

    def __init__(
        self,
        datasets: List[str],
        num_shots: int,
        args,
        preprocess,
        variable_shots: bool = False,
    ):
        self.dataset_names = datasets
        self.num_shots = num_shots
        self.args = args
        self.preprocess = preprocess
        self.variable_shots = variable_shots

        # Lazy-loaded caches
        self._dataset_cache: Dict[str, object] = {}
        self._text_cache: Dict[str, Dict[str, List[str]]] = {}
        self._class_indices_cache: Dict[str, Dict[int, List[int]]] = {}

    def _get_dataset(self, dataset_name: str):
        """Load (or retrieve cached) dataset."""
        if dataset_name not in self._dataset_cache:
            dataset = get_dataset(
                dataset_name + "Val",
                self.preprocess,
                location=self.args.data_location,
                batch_size=self.args.batch_size,
            )
            self._dataset_cache[dataset_name] = dataset
        return self._dataset_cache[dataset_name]

    def _get_text_descriptions(self, dataset_name: str) -> Dict[str, List[str]]:
        """Load (or retrieve cached) text descriptions."""
        if dataset_name not in self._text_cache:
            self._text_cache[dataset_name] = load_text_descriptions(
                dataset_name, self.args
            )
        return self._text_cache[dataset_name]

    def _get_class_indices(
        self, dataset_name: str, dataset
    ) -> Dict[int, List[int]]:
        """Build (or retrieve cached) mapping from class label → sample indices.

        Works with both torchvision-style datasets (with ``targets``) and
        aTLAS-style datasets that expose ``train_dataset``/``test_dataset``.
        """
        if dataset_name not in self._class_indices_cache:
            # Determine the underlying dataset object
            if hasattr(dataset, "train_dataset"):
                raw = dataset.train_dataset
            elif hasattr(dataset, "dataset"):
                raw = dataset.dataset
            else:
                raw = dataset

            # Extract targets
            if hasattr(raw, "targets"):
                targets = raw.targets
                if isinstance(targets, torch.Tensor):
                    targets = targets.tolist()
            else:
                # Fallback: iterate and collect labels
                targets = []
                for i in range(len(raw)):
                    _, label = raw[i]
                    targets.append(int(label))

            class_indices: Dict[int, List[int]] = {}
            for idx, label in enumerate(targets):
                class_indices.setdefault(label, []).append(idx)

            self._class_indices_cache[dataset_name] = class_indices

        return self._class_indices_cache[dataset_name]

    def sample_episode(
        self,
        dataset_name: Optional[str] = None,
    ) -> Tuple[str, Dict[str, List[str]], torch.Tensor, torch.Tensor]:
        """Sample one multi-modal episode.

        Args:
            dataset_name: Force a specific dataset.  If None, one is chosen
                uniformly at random from ``self.dataset_names``.

        Returns:
            A 4-tuple:
                - **dataset_name** (*str*): The sampled dataset name.
                - **text_descriptions** (*dict*): Class → description list.
                - **support_images** (*Tensor*): Shape [num_classes, K, C, H, W].
                - **support_labels** (*Tensor*): Shape [num_classes, K] (class indices).
        """
        if dataset_name is None:
            dataset_name = random.choice(self.dataset_names)

        # Load data
        dataset = self._get_dataset(dataset_name)
        text_descriptions = self._get_text_descriptions(dataset_name)
        class_indices = self._get_class_indices(dataset_name, dataset)

        # Determine shot count for this episode
        k = self.num_shots
        if self.variable_shots and k > 1:
            k = random.randint(1, k)

        # Determine underlying raw dataset for indexing
        if hasattr(dataset, "train_dataset"):
            raw = dataset.train_dataset
        elif hasattr(dataset, "dataset"):
            raw = dataset.dataset
        else:
            raw = dataset

        # Sample K-shot per class
        support_images = []
        support_labels = []
        num_classes = len(class_indices)

        for class_idx in sorted(class_indices.keys()):
            indices = class_indices[class_idx]

            # Sample K indices (with replacement if fewer available)
            if len(indices) >= k:
                shot_indices = random.sample(indices, k)
            else:
                shot_indices = random.choices(indices, k=k)

            class_images = []
            for idx in shot_indices:
                img, _ = raw[idx]
                if not isinstance(img, torch.Tensor):
                    img = torch.tensor(img)
                class_images.append(img)

            support_images.append(torch.stack(class_images))
            support_labels.append(
                torch.full((k,), class_idx, dtype=torch.long)
            )

        support_images = torch.stack(support_images)   # [C, K, ch, H, W]
        support_labels = torch.stack(support_labels)    # [C, K]

        return dataset_name, text_descriptions, support_images, support_labels

    def __len__(self) -> int:
        """Number of available datasets."""
        return len(self.dataset_names)
