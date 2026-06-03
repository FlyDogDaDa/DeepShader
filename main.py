"""DeepShader main entry point."""

from __future__ import annotations

import os
import random
from pathlib import Path

import torch
from dotenv import load_dotenv

from src.dataset import ImageDataset, create_dataloaders, default_transform
from src.utility import scan_dataset

load_dotenv()


def main() -> None:
    root = Path(os.environ["DATASET_ROOT"]) / "danbooru-images" / "danbooru-images"
    paths = scan_dataset(root)
    print(f"Found {len(paths)} images in {root}")

    dataset = ImageDataset(paths)
    print(f"Dataset length: {len(dataset)}")

    # Test single item
    idx = random.randint(0, len(dataset) - 1)
    img = dataset[idx]
    print(f"Single item - shape: {img.shape}, dtype: {img.dtype}")

    # Test batch
    batch_indices = [random.randint(0, len(dataset) - 1) for _ in range(4)]
    batch = dataset.get_batch(batch_indices)
    print(f"Batch - shape: {batch.shape}, dtype: {batch.dtype}")

    # Verify default_transform is accessible
    t = default_transform()
    print(f"Default transform: {t}")

    # Test create_dataloaders with a small subset
    random.seed(42)
    small_sample = random.sample(paths, min(1000, len(paths)))
    train_loader, val_loader, test_loader = create_dataloaders(
        small_sample,
        batch_size=32,
        num_workers=4,
    )
    print(
        f"Split — train: {len(train_loader.dataset)}, val: {len(val_loader.dataset)}, test: {len(test_loader.dataset)}"
    )
    print(
        f"DataLoader batches — train: {len(train_loader)}, val: {len(val_loader)}, test: {len(test_loader)}"
    )

    # Test a forward iteration
    batch = next(iter(train_loader))
    print(f"Train batch shape: {batch.shape}")

    print("Dataloader test passed ✓")
    print("All tests passed ✓")


if __name__ == "__main__":
    main()
