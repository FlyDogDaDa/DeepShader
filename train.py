#!/usr/bin/env python3
"""Train DinoToVAE: DINOv3 patch tokens → FLUX VAE latents.

Usage:
    # Default (MLP mapper, 100 epochs)
    uv run python -m train

    # Custom output dir (default: runs/exp-YYYYMMDD_HHMMSS)
    uv run python -m train --output runs/my-exp

    # Linear mapper, custom epochs
    uv run python -m train --mapper linear --epochs 50 --output runs/linear

    # Resume from checkpoint
    uv run python -m train --output runs/my-exp --resume runs/my-exp/checkpoints/epoch_0020.pt

    # Quick test on a tiny subset
    uv run python -m train --debug
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from dotenv import load_dotenv

from src import (
    TrainingConfig,
    create_dataloaders,
    load_dino,
    load_vae,
    run_training,
    scan_dataset,
)

# ── Output dir helper ─────────────────────────────────────────────


def _make_output_dir(base: str) -> str:
    """Generate output dir: ``base`` if exists, else ``base_YYYYMMDD_HHMMSS``."""
    base_path = Path(base)
    if base_path.exists():
        return base
    # Generate timestamped name
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{base}_{ts}"


# ── CLI ─────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DinoToVAE")

    # Data
    g = p.add_argument_group("data")
    g.add_argument(
        "--dataset",
        type=Path,
        default=Path("$DATASET_ROOT/danbooru-images/danbooru-images"),
        help="Root directory with image subfolders",
    )
    g.add_argument("--batch-size", type=int, default=8)
    g.add_argument("--num-workers", type=int, default=16)
    g.add_argument(
        "--prefetch", type=int, default=20, help="DataLoader prefetch_factor"
    )
    g.add_argument("--val-ratio", type=float, default=0.1)

    # Model
    g = p.add_argument_group("model")
    g.add_argument(
        "--mapper",
        type=str,
        default="mlp",
        choices=["linear", "mlp"],
        help="Mapper architecture",
    )
    g.add_argument("--hidden-channels", type=int, default=256)
    g.add_argument("--num-layers", type=int, default=4)

    # Training
    g = p.add_argument_group("training")
    g.add_argument("--epochs", type=int, default=100)
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--warmup-steps", type=int, default=1000)
    g.add_argument("--log-freq", type=int, default=100, help="Log every N steps")
    g.add_argument("--val-freq", type=int, default=5)
    g.add_argument(
        "--resume", type=str, default=None, help="Checkpoint path to resume from"
    )

    # Output
    g = p.add_argument_group("output")
    g.add_argument(
        "--output",
        type=str,
        default="runs/exp",
        help="Output directory (default: runs/exp or runs/exp_YYYYMMDD_HHMMSS)",
    )

    # Debug
    g = p.add_argument_group("debug")
    g.add_argument("--debug", action="store_true", help="Quick run on a tiny subset")

    return p.parse_args()


# ── Main ────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    load_dotenv()

    # ── Resolve dataset ──────────────────────────────────────────
    dataset_root = Path(os.environ.get("DATASET_ROOT", str(args.dataset)))

    # ── Output dir ───────────────────────────────────────────────
    output_dir = _make_output_dir(args.output)

    # ── Config ───────────────────────────────────────────────────
    config = TrainingConfig(
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        warmup_steps=args.warmup_steps,
        log_freq=args.log_freq,
        val_freq=args.val_freq,
        output_dir=output_dir,
        mapper=args.mapper,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        dataset=str(dataset_root),
    )

    # ── Debug mode: tiny subset ──────────────────────────────────
    if args.debug:
        config.epochs = 2
        config.batch_size = 2
        config.warmup_steps = 2
        config.val_freq = 1
        config.debug = True
        print(
            f"[train] Debug mode: {config.epochs} epochs, batch={config.batch_size}, val_freq={config.val_freq}"
        )

    # ── Load models ──────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] Device: {device}")
    print(f"[train] Loading models...")
    vae = load_vae(device)
    dino = load_dino(device)
    print(f"[train] VAE: {vae.latent_channels} latent channels")
    print(f"[train] DINO: {dino.feature_dim} feature dim")

    # ── Load dataset ─────────────────────────────────────────────
    print(f"[train] Scanning dataset: {dataset_root}")
    t0 = time.time()
    jpgs = scan_dataset(dataset_root, pattern="**/*.jpg", use_cache=True)
    pngs = scan_dataset(dataset_root, pattern="**/*.png", use_cache=True)
    all_paths = jpgs + pngs
    print(f"[train] Found {len(all_paths):,} images ({time.time() - t0:.1f}s)")

    # Debug: subsample
    if args.debug:
        all_paths = all_paths[:40]
        print(f"[train] Debug: subsampled to {len(all_paths)} images")

    if len(all_paths) == 0:
        print(f"[train] ERROR: No images found at {dataset_root}")
        print(f"[train] Set DATASET_ROOT environment variable or use --dataset")
        return

    # ── Create DataLoaders ───────────────────────────────────────
    # Create DataLoaders
    train_loader, val_loader, _ = create_dataloaders(
        all_paths,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        prefetch_factor=args.prefetch,
        train_ratio=0.9,
        val_ratio=config.val_ratio,
    )
    print(
        f"[train] Train: {len(train_loader.dataset):,} images, "
        f"Val: {len(val_loader.dataset):,} images"
    )

    # ── Print output structure ──────────────────────────────────
    print(f"\n[train] Output directory: {output_dir}/")
    print(f"  ├── config.yaml           # training config")
    print(f"  ├── checkpoints/")
    print(f"  │   ├── epoch_XXXX.pt     # per-epoch checkpoints")
    print(f"  │   └── final.pt          # final model")
    print(f"  ├── samples/")
    print(f"  │   └── epoch_XXXX/       # decoded sample images")
    print(f"  └── logs/")
    print(f"      └── train.log         # training log")

    # ── Run training ─────────────────────────────────────────────
    print(
        f"\n[train] Starting training: {config.epochs} epochs, mapper={config.mapper}, batch={config.batch_size}"
    )
    resume_path = args.resume if args.resume else None
    mapper, optimizer = run_training(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        dino=dino,
        vae=vae,
        resume_from=resume_path,
    )

    print(f"\n[train] Training complete!")
    print(f"[train] Final checkpoint: {output_dir}/checkpoints/final.pt")


if __name__ == "__main__":
    main()
