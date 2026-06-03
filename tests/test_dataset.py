"""Tests for src/dataset.py."""

from __future__ import annotations

import torch
import torchvision.transforms.v2 as T

from src.dataset import (
    ImageDataset,
    create_dataloaders,
    default_transform,
    load_image,
)

# ── load_image ────────────────────────────────────────────────────


def test_load_image_shape(sample_jpg) -> None:
    """load_image() returns (3, H, W)."""
    t = load_image(sample_jpg)
    assert t.shape == (3, 512, 512)
    assert t.dtype == torch.float32


def test_load_image_range(sample_jpg) -> None:
    """Values are in [0, 1]."""
    t = load_image(sample_jpg)
    assert t.min() >= 0.0
    assert t.max() <= 1.0


# ── default_transform ─────────────────────────────────────────────


def test_default_transform_returns_compose() -> None:
    """default_transform() returns a T.Compose."""
    t = default_transform()
    assert isinstance(t, T.Compose)


# ── ImageDataset ──────────────────────────────────────────────────


def test_dataset_len(small_paths) -> None:
    assert len(ImageDataset(small_paths)) == len(small_paths)


def test_dataset_getitem(sample_jpg) -> None:
    ds = ImageDataset([sample_jpg])
    img = ds[0]
    assert img.shape == (3, 512, 512)
    assert img.dtype == torch.float32


def test_dataset_getitem_transform(sample_jpg) -> None:
    """Applying transform still returns (C, H, W)."""
    ds = ImageDataset([sample_jpg], transform=default_transform())
    img = ds[0]
    assert img.shape == (3, 512, 512)


def test_dataset_transform_property_setter(sample_jpg) -> None:
    """Setting transform via property works."""
    ds = ImageDataset([sample_jpg])
    assert ds.transform is None

    ds.transform = default_transform()
    assert isinstance(ds.transform, T.Compose)

    ds.transform = None
    assert ds.transform is None


def test_dataset_getitem_default_transform(sample_jpg) -> None:
    """With no transform set, __getitem__ still returns a tensor."""
    ds = ImageDataset([sample_jpg], transform=None)
    img = ds[0]
    assert img.shape == (3, 512, 512)


def test_dataset_get_batch(sample_jpg) -> None:
    """get_batch() stacks images into (B, C, H, W)."""
    ds = ImageDataset([sample_jpg])
    batch = ds.get_batch([0, 0, 0, 0])
    assert batch.shape == (4, 3, 512, 512)
    assert batch.dtype == torch.float32


# ── create_dataloaders ────────────────────────────────────────────


def test_create_dataloaders_split(small_paths) -> None:
    """Split proportions are roughly correct with 30 paths."""
    train_loader, val_loader, test_loader = create_dataloaders(
        small_paths,
        batch_size=4,
        num_workers=0,
        seed=42,
    )
    total = (
        len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset)
    )
    assert total == 30


def test_create_dataloaders_shapes(small_paths) -> None:
    """Dataloader iteration yields (B, C, H, W) tensors."""
    train_loader, val_loader, test_loader = create_dataloaders(
        small_paths,
        batch_size=4,
        num_workers=0,
        seed=42,
    )

    batch = next(iter(train_loader))
    assert batch.ndim == 4
    assert batch.shape[1] == 3
    assert batch.dtype == torch.float32


def test_create_dataloaders_no_shuffle(small_paths) -> None:
    """val_loader uses SequentialSampler (no shuffle)."""
    from torch.utils.data import SequentialSampler

    _, val_loader, _ = create_dataloaders(
        small_paths, batch_size=4, num_workers=0, seed=42
    )
    assert isinstance(val_loader.sampler, SequentialSampler)


def test_create_dataloaders_shuffle(small_paths) -> None:
    """train_loader uses RandomSampler (shuffled)."""
    from torch.utils.data import RandomSampler

    train_loader, _, _ = create_dataloaders(
        small_paths, batch_size=4, num_workers=0, seed=42
    )
    assert isinstance(train_loader.sampler, RandomSampler)


def test_create_dataloaders_custom_ratios(artificial_paths) -> None:
    """Custom ratios produce expected split sizes."""
    train_loader, val_loader, test_loader = create_dataloaders(
        artificial_paths,
        train_ratio=0.6,
        val_ratio=0.25,
        test_ratio=0.15,
        batch_size=32,
        num_workers=0,
        seed=99,
    )
    assert len(train_loader.dataset) == 60, f"train={len(train_loader.dataset)}"
    assert len(val_loader.dataset) == 25, f"val={len(val_loader.dataset)}"
    assert len(test_loader.dataset) == 15, f"test={len(test_loader.dataset)}"
