"""Synthetic dataset loader for generated images.

This module provides a PyTorch dataset wrapper for synthetically generated images
created via text-to-image models.
"""

import os
import random
from pathlib import Path
from PIL import Image
import torch
from torchvision.datasets import VisionDataset
from typing import Optional, Callable


class SyntheticDataset(VisionDataset):
    """Dataset wrapper for synthetically generated images.

    Expected directory structure:
        root/
            backend/        # e.g., stable_diffusion
                dataset/    # e.g., CIFAR10
                    class1/
                        00000.png
                        00001.png
                        ...
                    class2/
                        00000.png
                        ...
    """

    def __init__(
        self,
        root: str,
        dataset_name: str,
        t2i_backend: str,
        transform: Optional[Callable] = None,
        split: str = "train",
        train_val_split: float = 0.8,
        max_images_per_class: Optional[int] = None,
        shuffle_selection: bool = False,
    ):
        """Initialize synthetic dataset.

        Args:
            root: Root directory for synthetic images (e.g., "data/synthetic_images")
            dataset_name: Name of the dataset (e.g., "CIFAR10")
            t2i_backend: T2I backend name (e.g., "stable_diffusion")
            transform: Optional transform to apply to images
            split: Dataset split ("train" or "val")
            train_val_split: Fraction of data to use for training (default: 0.8)
            max_images_per_class: If set, cap each class at this many images.
            shuffle_selection: When capping, randomly sample rather than taking the first N.
        """
        super().__init__(root=root, transform=transform)

        self.dataset_path = Path(root) / t2i_backend / dataset_name

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Synthetic dataset not found: {self.dataset_path}\n"
                f"Generate it using: python src/generate_synthetic_data.py "
                f"--dataset {dataset_name} --t2i-backend {t2i_backend}"
            )

        # Get class names from directory structure.
        # Folder names may be sanitized (e.g. "/" replaced with "_") when
        # the canonical class name contains filesystem-unsafe characters.
        mapping_file = self.dataset_path / "class_mapping.json"
        if mapping_file.exists():
            import json
            with open(mapping_file) as f:
                self._folder_to_class = json.load(f)
        else:
            self._folder_to_class = {}

        self._class_to_folder = {}
        for d in sorted(self.dataset_path.iterdir()):
            if d.is_dir() and not d.name.startswith('.'):
                canonical = self._folder_to_class.get(d.name, d.name)
                self._class_to_folder[canonical] = d.name

        self.classes = sorted(self._class_to_folder.keys())

        if not self.classes:
            raise ValueError(
                f"No class directories found in {self.dataset_path}"
            )

        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # Load image paths, optionally capping per class
        self.samples = []
        for class_name in self.classes:
            class_dir = self.dataset_path / self._class_to_folder[class_name]
            paths = sorted(class_dir.glob("*.png"))
            if max_images_per_class is not None and len(paths) > max_images_per_class:
                if shuffle_selection:
                    paths = random.sample(paths, max_images_per_class)
                else:
                    paths = paths[:max_images_per_class]
            for img_path in paths:
                self.samples.append((str(img_path), self.class_to_idx[class_name]))

        # Split into train/val
        n_train = int(len(self.samples) * train_val_split)
        if split == "train":
            self.samples = self.samples[:n_train]
        elif split == "val":
            self.samples = self.samples[n_train:]
        else:
            raise ValueError(f"Unknown split: {split}. Must be 'train' or 'val'")

        self.split = split
        print(f"Loaded Synthetic{dataset_name} ({split}): {len(self.samples)} images, {len(self.classes)} classes")

    def __getitem__(self, index):
        """Get image and label at index.

        Args:
            index: Index

        Returns:
            Tuple of (image, target) where target is the class index
        """
        img_path, target = self.samples[index]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, target

    def __len__(self):
        """Return the number of samples."""
        return len(self.samples)

    @property
    def classnames(self):
        """Return list of class names (for compatibility with other datasets)."""
        return self.classes


class SyntheticDatasetWrapper:
    """Wrapper to match the interface of other datasets in the registry.

    This provides train_dataset, train_loader, test_dataset, test_loader, and classnames
    to match the expected interface.
    """

    def __init__(
        self,
        preprocess,
        location: str = "data/synthetic_images",
        dataset_name: str = "CIFAR10",
        t2i_backend: str = "stable_diffusion",
        batch_size: int = 128,
        num_workers: int = 8,
        train_val_split: float = 0.8,
        max_images_per_class: Optional[int] = None,
        shuffle_selection: bool = False,
    ):
        """Initialize synthetic dataset wrapper.

        Args:
            preprocess: Transform/preprocess function
            location: Root directory for synthetic images
            dataset_name: Name of the dataset
            t2i_backend: T2I backend used for generation
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            train_val_split: Fraction for train split
            max_images_per_class: If set, cap each class at this many images.
            shuffle_selection: When capping, randomly sample rather than taking the first N.
        """
        # Create datasets
        self.train_dataset = SyntheticDataset(
            root=location,
            dataset_name=dataset_name,
            t2i_backend=t2i_backend,
            transform=preprocess,
            split="train",
            train_val_split=train_val_split,
            max_images_per_class=max_images_per_class,
            shuffle_selection=shuffle_selection,
        )

        self.test_dataset = SyntheticDataset(
            root=location,
            dataset_name=dataset_name,
            t2i_backend=t2i_backend,
            transform=preprocess,
            split="val",
            train_val_split=train_val_split,
            max_images_per_class=max_images_per_class,
            shuffle_selection=shuffle_selection,
        )

        # Create dataloaders
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        self.classnames = self.train_dataset.classnames


class MixedDatasetWrapper:
    """Combines real and synthetic datasets with approximately equal samples.

    Training data is balanced 50/50 between real and synthetic.
    Test data always comes from the real dataset.
    """

    def __init__(self, real_dataset, synthetic_dataset, batch_size=128, num_workers=8):
        real_train = real_dataset.train_dataset
        synth_train = synthetic_dataset.train_dataset

        n_real = len(real_train)
        n_synth = len(synth_train)
        target_per_source = min(n_real, n_synth)

        if n_real > target_per_source:
            indices = torch.randperm(n_real)[:target_per_source]
            real_train = torch.utils.data.Subset(real_train, indices)
        if n_synth > target_per_source:
            indices = torch.randperm(n_synth)[:target_per_source]
            synth_train = torch.utils.data.Subset(synth_train, indices)

        self.train_dataset = torch.utils.data.ConcatDataset([real_train, synth_train])
        self.classnames = real_dataset.classnames

        # Test always on real data
        self.test_dataset = real_dataset.test_dataset
        self.test_loader = real_dataset.test_loader

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        print(f"MixedDataset: {target_per_source} real + {target_per_source} synthetic = {len(self.train_dataset)} total training samples")


def get_synthetic_dataset(
    dataset_name,
    preprocess,
    location,
    batch_size,
    num_workers,
    t2i_backend="stable_diffusion",
    max_images_per_class=None,
    shuffle_selection=False,
):
    """Factory function to create a synthetic dataset.

    Args:
        dataset_name: Name of the dataset (e.g., "CIFAR10")
        preprocess: Preprocessing transform
        location: Root directory for synthetic images
        batch_size: Batch size
        num_workers: Number of workers
        t2i_backend: T2I backend name
        max_images_per_class: If set, cap each class at this many images.
        shuffle_selection: When capping, randomly sample rather than taking the first N.

    Returns:
        SyntheticDatasetWrapper instance
    """
    return SyntheticDatasetWrapper(
        preprocess=preprocess,
        location=location,
        dataset_name=dataset_name,
        t2i_backend=t2i_backend,
        batch_size=batch_size,
        num_workers=num_workers,
        max_images_per_class=max_images_per_class,
        shuffle_selection=shuffle_selection,
    )
