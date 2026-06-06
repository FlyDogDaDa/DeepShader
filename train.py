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
    max_cached_shards: int = 8,
    seed: int = 42,
    subset_size: int | None = None,
    no_val: bool = False,
    in_memory: bool = False,
    in_memory_shards: int | None = None,
) -> tuple[Any, Any, Any]:
    """Create train/val DataLoaders from cached shards.

    If ``in_memory`` is True, all shards are loaded into RAM via ``InMemoryDataset``
    — no ``ShardAwareSampler``, no disk I/O per batch, just pure tensor slicing.
    (Takes precedence over ``subset_size``.)

    ``in_memory_shards`` directly specifies how many shards to load into RAM.
    If not provided, it's inferred from ``subset_size`` (each shard ≈ 1000 samples).

    If ``no_val`` is True, only a training loader is returned (val is None).
    If ``subset_size`` is given, only the first N samples are loaded.
    """
    from torch.utils.data import DataLoader

    config = ShardCacheConfig(
        cache_dir=cache_dir,
        shard_size=shard_size,
        max_cached_shards=max_cached_shards,
    )

    if in_memory:
        # ── In-memory mode: load all shards, simple DataLoader ──
        from src import InMemoryDataset

        # Prefer explicit --in-memory-shards, fall back to inferring from --subset
        if in_memory_shards is not None:
            num_shards = in_memory_shards
        elif subset_size is not None:
            num_shards = (subset_size + shard_size - 1) // shard_size
        else:
            # No subset → load ALL shards into memory
            num_shards = load_manifest(cache_dir).total_shards

        dataset = InMemoryDataset(config, num_shards=num_shards)

        def _collate_fn(batch):
            indices = [idx for idx in batch]
            tokens = dataset._tokens[indices].to(torch.float32)
            means = dataset._latents[indices].to(torch.float32)
            logvars = dataset._logvars[indices].to(torch.float32)
            return tokens, means, logvars

        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,  # simple shuffle — no sampler needed
            num_workers=num_workers,
            drop_last=not no_val,
            collate_fn=_collate_fn,
            pin_memory=True,  # faster CPU→GPU transfer
        )

        if no_val:
            return train_loader, None, dataset

        # Val: sequential subset of the in-memory dataset
        # Use a simple SequentialSampler over the first N validation samples
        from torch.utils.data import SequentialSampler

        val_size = int(len(dataset) * val_ratio) if val_ratio < 1.0 else 0
        val_dataset = torch.utils.data.Subset(
            dataset,
            range(len(dataset) - val_size, len(dataset))
            if val_size > 0
            else range(len(dataset)),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            collate_fn=_collate_fn,
        )

        return train_loader, val_loader, dataset

    # ── Disk-backed CachedDataset mode (existing) ──
    subset_indices = set(range(subset_size)) if subset_size else None
    dataset = CachedDataset(config, subset_indices=subset_indices)

    from src import ShardAwareSampler

    def _collate_fn(batch):
        tokens = torch.stack([b[0].squeeze(0) for b in batch])
        means = torch.stack([b[1].squeeze(0) for b in batch])
        logvars = torch.stack([b[2].squeeze(0) for b in batch])
        return tokens, means, logvars

    train_sampler = ShardAwareSampler(
        dataset,
        train_ratio=0.9 if not no_val else 1.0,
        seed=seed,
        is_train=True,
    )
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        drop_last=not no_val,
        collate_fn=_collate_fn,
        pin_memory=True,
    )

    if no_val:
        return train_loader, None, dataset

    val_sampler = ShardAwareSampler(
        dataset,
        train_ratio=0.9 if not no_val else 1.0,
        seed=seed,
        is_train=False,
    )
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=0,
        drop_last=False,
        collate_fn=_collate_fn,
        pin_memory=True,
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
        "--cache-shards",
        type=int,
        default=8,
        help="Max shards to keep in RAM (default: 8, ~10 GB)",
    )
    g.add_argument(
        "--no-cache",
        action="store_true",
        help="Force image mode (ignore cache)",
    )
    g.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Limit to first N samples from cache (None=all). Use with --cache-shards to keep all data in RAM.",
    )
    g.add_argument(
        "--no-val",
        action="store_true",
        help="Use all data for training (no train/val split). Useful with small subsets.",
    )
    g.add_argument(
        "--in-memory",
        action="store_true",
        help="Load shards into RAM (InMemoryDataset) — no ShardAwareSampler, pure tensor slicing.",
    )
    g.add_argument(
        "--in-memory-shards",
        type=int,
        default=None,
        help="Number of shards to load into RAM. Overrides --subset for in-memory mode (each shard ≈ 1000 samples).",
    )
    g.add_argument(
        "--save-freq",
        type=int,
        default=5,
        help="Save checkpoint & samples every N epochs (default: 5)",
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
        choices=["linear", "mlp", "resnet", "transformer", "transformer_resnet"],
        help="Mapper architecture",
    )
    g.add_argument("--hidden-channels", type=int, default=256)
    g.add_argument("--num-layers", type=int, default=4)
    g.add_argument(
        "--num-heads", type=int, default=8, help="Transformer attention heads"
    )
    g.add_argument(
        "--num-transformer-layers",
        type=int,
        default=1,
        help="Number of transformer encoder blocks (for transformer mappers)",
    )
    g.add_argument(
        "--mlp-ratio",
        type=float,
        default=4.0,
        help="FFN hidden dim / dim ratio in transformer",
    )
    g.add_argument(
        "--num-resnet-layers",
        type=int,
        default=32,
        help="Number of ResNet blocks (for transformer_resnet mapper)",
    )

    # ── Training ───────────────────────────────────────────────
    g = p.add_argument_group("training")
    g.add_argument("--epochs", type=int, default=100)
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--warmup-steps", type=int, default=1000)
    g.add_argument("--log-freq", type=int, default=100, help="Log every N steps")
    g.add_argument(
        "--beta",
        type=float,
        default=1e-4,
        help="KL divergence weight β (MSE + β*KLD, default: 1e-4)",
    )
    g.add_argument(
        "--resume", type=str, default=None, help="Checkpoint path to resume from"
    )
    g.add_argument(
        "--sample-indices",
        type=str,
        default=None,
        help="Fixed dataset indices for sampling (e.g. '0,2,4,8'). Saves tokens/gt/pred.pt for manual decoding.",
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
    sample_indices: list[int] = []
    if args.sample_indices:
        sample_indices = [int(i) for i in args.sample_indices.split(",")]

    config = TrainingConfig(
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        warmup_steps=args.warmup_steps,
        log_freq=args.log_freq,
        beta=args.beta,
        output_dir=output_dir,
        mapper=args.mapper,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_transformer_layers=args.num_transformer_layers,
        mlp_ratio=args.mlp_ratio,
        num_resnet_layers=args.num_resnet_layers,
        subset_size=args.subset,
        save_every=args.save_freq,
        dataset=str(dataset_root) if not use_cached else str(args.cache_dir),
        sample_indices=sample_indices,
    )

    # ── Debug mode: tiny subset ────────────────────────────────
    if args.debug:
        config.epochs = 2
        config.batch_size = 2
        config.warmup_steps = 2
        config.debug = True
        print(f"[train] Debug mode: {config.epochs} epochs, batch={config.batch_size}")

    # ── Mode-specific setup ────────────────────────────────────
    if use_cached:
        # ── Cached mode: load shards ─────────────────────────────
        print(f"[train] Cache dir: {args.cache_dir}")

        # Check cache exists
        if not args.cache_dir.exists():
            print(f"[train] ERROR: Cache directory does not exist: {args.cache_dir}")
            print(f"[train] Run `python -m src.encode` first to create cache")
            return

        # Debug mode: avoid ShardAwareSampler (pre-builds 337K index list)
        # and avoid building full CachedDataset index map (reads 338 meta.json).
        # Use RandomSampler over a minimal CachedDataset with subset_indices.
        from torch.utils.data import RandomSampler, SequentialSampler, Subset

        config_ = ShardCacheConfig(
            cache_dir=args.cache_dir,
            shard_size=1000,
            max_cached_shards=args.cache_shards,
        )
        debug_size = 4 if args.debug else 0

        if args.debug:
            # Debug path: tiny subset, no ShardAwareSampler, no full index map
            debug_indices = list(range(debug_size))

            def _collate_fn(batch):
                tokens = torch.stack([b[0].squeeze(0) for b in batch])
                means = torch.stack([b[1].squeeze(0) for b in batch])
                logvars = torch.stack([b[2].squeeze(0) for b in batch])
                return tokens, means, logvars

            # Only build index entries for debug_size items (~1 meta.json read)
            # len() == debug_size, indices [0..debug_size-1] all work
            debug_dataset = CachedDataset(config_, subset_indices=set(debug_indices))
            train_loader = torch.utils.data.DataLoader(
                debug_dataset,
                batch_size=config.batch_size,
                sampler=RandomSampler(debug_dataset, replacement=False),
                num_workers=0,
                drop_last=True,
                collate_fn=_collate_fn,
            )
            val_loader = torch.utils.data.DataLoader(
                debug_dataset,
                batch_size=config.batch_size,
                sampler=SequentialSampler(debug_dataset),
                num_workers=0,
                drop_last=False,
                collate_fn=_collate_fn,
            )
            print(
                f"[train] Debug: {len(debug_dataset)} samples "
                f"(skipped 338 shards metadata + ShardAwareSampler)"
            )
            dataset = debug_dataset
        else:
            # Normal path: full dataset with ShardAwareSampler
            train_loader, val_loader, dataset = _make_cached_dataloaders(
                cache_dir=args.cache_dir,
                batch_size=config.batch_size,
                num_workers=0,
                val_ratio=config.val_ratio,
                shard_size=1000,
                max_cached_shards=args.cache_shards,
                seed=42,
                subset_size=config.subset_size,
                no_val=config.no_val,
                in_memory=args.in_memory,
                in_memory_shards=args.in_memory_shards,
            )

        print(f"[train] Dataset: {dataset.info()}")
        vae = None
        dino = None

    else:
        # ── Image mode: scan + encode per batch (backward compat) ─
        t0 = time.time()
        jpgs = scan_dataset(dataset_root, pattern="**/*.jpg", use_cache=True)
        pngs = scan_dataset(dataset_root, pattern="**/*.png", use_cache=True)
        all_paths = jpgs + pngs
        print(f"[train] Found {len(all_paths):,} images ({time.time() - t0:.1f}s)")

        # Debug: subsample to a handful of images
        if args.debug:
            all_paths = all_paths[:4]
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
