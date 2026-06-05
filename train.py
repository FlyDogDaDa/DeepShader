#!/usr/bin/env python3
"""Train DinoToVAE: DINOv3 patch tokens → FLUX VAE latents.

Two training modes:
    Image mode (default): images → DINOv3 → mapper → VAE → latents
    Cached mode:          patch_tokens + gt_latents from shard cache (faster)

Usage:
    # Image mode (backward compatible, loads DINO + VAE each epoch)
    uv run python -m train --output runs/my-exp

    # Cached mode (recommended: pre-encode with `python -m src.encode` first)
    uv run python -m train --cache-dir ~/hdd/cache --output runs/cached-exp

    # Linear mapper, custom epochs
    uv run python -m train --mapper linear --epochs 50 --output runs/linear

    # Resume from checkpoint
    uv run python -m train --output runs/my-exp --resume runs/my-exp/checkpoints/epoch_0020.pt

    # Quick debug run
    uv run python -m train --debug
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from dotenv import load_dotenv

from src import (
    CachedDataset,
    ShardCacheConfig,
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
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{base}_{ts}"


# ── Cached DataLoader factory ─────────────────────────────────────


def _make_cached_dataloaders(
    cache_dir: Path,
    batch_size: int,
    num_workers: int,
    val_ratio: float,
    shard_size: int = 1000,
    seed: int = 42,
) -> tuple[Any, Any, Any]:
    """Create train/val DataLoaders from cached shards.

    Uses ``ShardAwareSampler`` for high LRU cache hit rate (~97%):
    Level 1 — shuffle shard order
    Level 2 — sequential within each shard
    """
    from torch.utils.data import DataLoader

    config = ShardCacheConfig(
        cache_dir=cache_dir,
        shard_size=shard_size,
        max_cached_shards=8,
    )
    dataset = CachedDataset(config)

    # ── Train sampler: ShardAwareSampler with epoch-level seed ──────
    # ShardAwareSampler handles the train/val split internally via
    # ``train_ratio``, shuffling shard order but keeping sequential
    # access within shards → ~97% LRU hit rate.
    from src import ShardAwareSampler

    train_sampler = ShardAwareSampler(
        dataset,
        train_ratio=0.9,
        seed=seed,
        is_train=True,
    )
    val_sampler = ShardAwareSampler(
        dataset,
        train_ratio=0.9,
        seed=seed,
        is_train=False,
    )

    def _collate_fn(batch):
        tokens = torch.stack([b[0].squeeze(0) for b in batch])
        latents = torch.stack([b[1].squeeze(0) for b in batch])
        return tokens, latents

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=_collate_fn,
    )

    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=0,
        drop_last=False,
        collate_fn=_collate_fn,
    )

    return train_loader, val_loader, dataset


# ── CLI ─────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DinoToVAE")

    # ── Mode selection ──────────────────────────────────────────
    g = p.add_argument_group("mode")
    g.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Cache directory for pre-encoded shards (cached mode)",
    )
    g.add_argument(
        "--no-cache",
        action="store_true",
        help="Force image mode (ignore cache)",
    )

    # ── Data (image mode) ──────────────────────────────────────
    g = p.add_argument_group("data (image mode)")
    g.add_argument("--num-workers", type=int, default=16)
    g.add_argument(
        "--dataset",
        type=Path,
        default=Path("$DATASET_ROOT/danbooru-images/danbooru-images"),
        help="Root directory with image subfolders (image mode only)",
    )
    g.add_argument("--batch-size", type=int, default=8)
    g.add_argument(
        "--prefetch", type=int, default=20, help="DataLoader prefetch_factor"
    )
    g.add_argument("--val-ratio", type=float, default=0.1)

    # ── Model ──────────────────────────────────────────────────
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

    # ── Training ───────────────────────────────────────────────
    g = p.add_argument_group("training")
    g.add_argument("--epochs", type=int, default=100)
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--warmup-steps", type=int, default=1000)
    g.add_argument("--log-freq", type=int, default=100, help="Log every N steps")
    g.add_argument("--val-freq", type=int, default=5)
    g.add_argument(
        "--resume", type=str, default=None, help="Checkpoint path to resume from"
    )

    # ── Output ─────────────────────────────────────────────────
    g = p.add_argument_group("output")
    g.add_argument(
        "--output",
        type=str,
        default="runs/exp",
        help="Output directory (default: runs/exp or runs/exp_YYYYMMDD_HHMMSS)",
    )

    # ── Debug ──────────────────────────────────────────────────
    g = p.add_argument_group("debug")
    g.add_argument("--debug", action="store_true", help="Quick run on a tiny subset")

    return p.parse_args()


# ── Main ────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    load_dotenv()

    # ── Determine mode ─────────────────────────────────────────
    use_cached = args.cache_dir is not None and not args.no_cache
    mode_label = "cached" if use_cached else "image"
    print(f"[train] Mode: {mode_label}")

    # ── Resolve dataset ────────────────────────────────────────
    dataset_root = Path(os.environ.get("DATASET_ROOT", str(args.dataset)))

    # ── Output dir ─────────────────────────────────────────────
    output_dir = _make_output_dir(args.output)

    # ── Config ─────────────────────────────────────────────────
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
        dataset=str(dataset_root) if not use_cached else str(args.cache_dir),
    )

    # ── Debug mode: tiny subset ────────────────────────────────
    if args.debug:
        config.epochs = 2
        config.batch_size = 2
        config.warmup_steps = 2
        config.val_freq = 1
        config.debug = True
        print(
            f"[train] Debug mode: {config.epochs} epochs, batch={config.batch_size}, val_freq={config.val_freq}"
        )

    # ── Mode-specific setup ────────────────────────────────────
    if use_cached:
        # ── Cached mode: load shards ─────────────────────────────
        print(f"[train] Cache dir: {args.cache_dir}")

        # Check cache exists
        if not args.cache_dir.exists():
            print(f"[train] ERROR: Cache directory does not exist: {args.cache_dir}")
            print(f"[train] Run `python -m src.encode` first to create cache")
            return

        # CachedDataset uses internal LRU cache, no need for multiple workers.
        # Using num_workers=0 (main process) avoids spawning subprocesses that
        # each load shard tensors into RAM → OOM.
        train_loader, val_loader, dataset = _make_cached_dataloaders(
            cache_dir=args.cache_dir,
            batch_size=config.batch_size,
            num_workers=0,
            val_ratio=config.val_ratio,
            shard_size=1000,
            seed=42,
        )
        print(f"[train] Dataset: {dataset.info()}")
        vae = None
        dino = None

    else:
        # ── Image mode: scan + encode per batch (backward compat) ─
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
            return

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

        # Load DINO + VAE for image mode
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[train] Device: {device}")
        print(f"[train] Loading models...")
        vae = load_vae(device)
        dino = load_dino(device)
        print(f"[train] VAE: {vae.latent_channels} latent channels")
        print(f"[train] DINO: {dino.feature_dim} feature dim")

    # ── Print output structure ─────────────────────────────────
    print(f"\n[train] Output directory: {output_dir}/")
    print(f"  ├── config.yaml           # training config")
    print(f"  ├── checkpoints/")
    print(f"  │   ├── epoch_XXXX.pt     # per-epoch checkpoints")
    print(f"  │   └── final.pt          # final model")
    print(f"  ├── samples/")
    print(f"  │   └── epoch_XXXX/       # decoded sample images")
    print(f"  └── logs/")
    print(f"      └── train.log         # training log")

    # ── Run training ───────────────────────────────────────────
    print(
        f"\n[train] Starting training: {config.epochs} epochs, "
        f"mapper={config.mapper}, batch={config.batch_size}, mode={mode_label}"
    )
    resume_path = args.resume if args.resume else None

    # Cached mode: pass dino=None, vae=None → run_training auto-detects
    # Image mode: pass dino+vae → run_training uses image mode
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
