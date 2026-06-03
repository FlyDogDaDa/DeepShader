from __future__ import annotations

from pathlib import Path

import torch
import torchvision.transforms.v2 as T
from PIL import Image
from torch.utils.data import DataLoader

from src.utility import train_val_test_split


def load_image(path: Path) -> torch.Tensor:
    """Load an image from disk and return a normalized Tensor (C, H, W) in [0, 1]."""
    img = Image.open(path).convert("RGB")
    return T.functional.to_image(img).to(torch.float32) / 255.0


def default_transform() -> T.Transform:
    """Return the default transform pipeline (no resize)."""
    return T.Compose([T.RandomHorizontalFlip(p=0.5), T.RandomVerticalFlip(p=0.5)])


class ImageDataset:
    """Dataset that loads images from a list of paths and returns stacked tensors."""

    def __init__(
        self,
        paths: list[Path],
        transform: T.Transform | None = None,
    ) -> None:
        self.paths = paths
        self._transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    @property
    def transform(self) -> T.Transform | None:
        """Return the current transform, or None."""
        return self._transform

    @transform.setter
    def transform(self, value: T.Transform | None) -> None:
        """Set the transform pipeline."""
        self._transform = value

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = load_image(self.paths[idx])
        t = self._transform if self._transform is not None else default_transform()
        return t(img)

    def get_batch(self, indices: list[int]) -> torch.Tensor:
        """Return a stacked batch of images (B, C, H, W)."""
        return torch.stack([self[i] for i in indices])


def _make_loader(
    dataset: ImageDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    drop_last: bool,
) -> DataLoader:
    """Create a DataLoader for *dataset* with shared defaults."""
    prefetch = None if num_workers == 0 else 2
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        prefetch_factor=prefetch,
    )


def create_dataloaders(
    paths: list[Path],
    *,
    batch_size: int = 64,
    num_workers: int = 16,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train / val / test DataLoaders from a list of image paths.

    Args:
        paths: All image paths.
        batch_size: Batch size for each DataLoader.
        num_workers: Number of DataLoader workers.
        train_ratio: Fraction for training (default 0.8).
        val_ratio: Fraction for validation (default 0.1).
        test_ratio: Fraction for testing (default 0.1).
        seed: Random seed for the split.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    train_paths, val_paths, test_paths = train_val_test_split(
        paths,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    train_transform = default_transform()

    train_dataset = ImageDataset(train_paths, transform=train_transform)
    val_dataset = ImageDataset(val_paths, transform=None)
    test_dataset = ImageDataset(test_paths, transform=None)

    train_loader = _make_loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = _make_loader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    test_loader = _make_loader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
