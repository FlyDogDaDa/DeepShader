"""Integration test — mirrors main.py end-to-end."""

from __future__ import annotations

import random
from pathlib import Path

import torch

from src.dataset import ImageDataset, create_dataloaders, default_transform
from src.utility import scan_dataset


def test_full_pipeline(tmp_path: Path) -> None:
    """Scan → dataset → batch → dataloader → iteration."""
    # Setup temp dataset
    base = tmp_path / "danbooru-images"
    base.mkdir()
    for subdir in ("0000",):
        d = base / subdir
        d.mkdir()
        for i in range(64):
            from PIL import Image

            img = Image.new("RGB", (512, 512), color=(i * 3 % 255, 64, 200))
            img.save(d / f"{i:07d}.jpg", "JPEG")

    paths = scan_dataset(base, use_cache=False)
    assert len(paths) == 64

    # Single item
    ds = ImageDataset(paths)
    img = ds[0]
    assert img.shape == (3, 512, 512)

    # Batch
    batch = ds.get_batch([0, 1, 2, 3])
    assert batch.shape == (4, 3, 512, 512)

    # Transform
    t = default_transform()
    assert isinstance(t, torch.nn.Module)

    # DataLoader split
    random.seed(42)
    train_loader, val_loader, test_loader = create_dataloaders(
        paths, batch_size=16, num_workers=0, seed=42
    )
    assert (
        len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset)
        == 64
    )

    # Iterate
    batch = next(iter(train_loader))
    assert batch.ndim == 4
