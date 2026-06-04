#!/usr/bin/env python3
"""Pre-encode pipeline: Images → DINOv3 + VAE.encode → shard cache.

Run this ONCE before training. It encodes all images into pre-computed
shard files that the training pipeline will read.

Usage:
    # Basic: encode full dataset
    uv run python -m src.encode --dataset /data/danbooru --cache-dir ~/hdd/cache

    # Resume from checkpoint (skip already encoded shards)
    uv run python -m src.encode --dataset /data/danbooru --cache-dir ~/hdd/cache --resume

    # Dry run: validate manifest only, don't encode
    uv run python -m src.encode --dataset /data/danbooru --cache-dir ~/hdd/cache --dry-run

    # Custom shard size
    uv run python -m src.encode --dataset /data/danbooru --cache-dir ~/hdd/cache --shard-size 500
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from dotenv import load_dotenv
from tqdm import tqdm

from src import (
    CacheManifest,
    ShardCacheConfig,
    ShardMeta,
    compute_dataset_hash,
    load_dino,
    load_manifest,
    load_vae,
    save_manifest,
    scan_dataset,
    validate_cache,
)
from src.dataset import ImageDataset, default_transform

# ── CLI ───────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pre-encode dataset: Images → DINOv3 + VAE.encode → shard cache",
    )

    # Data
    g = p.add_argument_group("data")
    g.add_argument(
        "--dataset",
        type=Path,
        default=Path("$DATASET_ROOT/danbooru-images/danbooru-images"),
        help="Root directory with image subfolders",
    )
    g.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / "hdd" / "cache",
        help="Directory to store pre-encoded shards",
    )
    g.add_argument(
        "--shard-size",
        type=int,
        default=1000,
        help="Images per shard (default: 1000)",
    )

    # Model
    g = p.add_argument_group("model")
    g.add_argument(
        "--dino-model",
        type=str,
        default="facebook/dinov3-vits16-pretrain-lvd1689m",
        help="DINOv3 model identifier",
    )
    g.add_argument(
        "--vae-model",
        type=str,
        default="advokat/AnimePro-FLUX",
        help="VAE model identifier",
    )

    # Execution
    g = p.add_argument_group("execution")
    g.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Encode batch size per step (VRAM-bound; default: 16)",
    )
    g.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last completed shard",
    )
    g.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate manifest only, skip encoding",
    )
    g.add_argument(
        "--force",
        action="store_true",
        help="Force re-encoding even if cache exists",
    )
    g.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run encoding on (default: cuda)",
    )

    return p.parse_args()


# ── Shard Encoder ───────────────────────────────────────────────────


class BatchShardEncoder:
    """Encodes a batch of images into DINOv3 patch tokens + VAE latents.

    Buffers are stored as contiguous 1-D tensors (dim=0 = sample dimension).
    Each feed() appends via ``torch.cat`` on dim=0.  When the buffer reaches
    ``shard_size`` the head chunk is flushed to disk and removed.

    This avoids Python-list overhead and makes flush a simple slice operation.
    """

    def __init__(
        self, cache_dir: Path, shard_size: int, dino_model: str, vae_model: str
    ):
        self.cache_dir = cache_dir
        self.shard_size = shard_size
        self.dino_model = dino_model
        self.vae_model = vae_model

        # Buffers – start empty (None means "not yet initialized")
        self._buffer_tokens: torch.Tensor | None = None
        self._buffer_latents: torch.Tensor | None = None
        self._buffer_paths: list[str] = []
        self._shard_id: int = 0

    @property
    def buffer_size(self) -> int:
        """Number of samples currently in the buffer tensors."""
        if self._buffer_tokens is None:
            return 0
        return self._buffer_tokens.shape[0]

    @property
    def shard_id(self) -> int:
        return self._shard_id

    def feed(self, tokens: torch.Tensor, latents: torch.Tensor, paths: list[str]):
        """Append a batch to the buffer.

        * ``tokens`` shape:  ``[B, 1024, 384]`` (can be on any device)
        * ``latents`` shape: ``[B, 16, 64, 64]`` (can be on any device)

        Tensors are moved to **CPU** immediately via ``.to('cpu')`` so the
        buffer only holds RAM tensors — GPU memory is released before the
        next batch.

        When the buffer reaches ``shard_size`` the first ``shard_size``
        samples are flushed to disk and removed from the buffer.
        """
        # .to('cpu') copies to RAM; original GPU tensor is freed when the
        # caller drops its reference (patch_tokens / gt_latents).
        tokens = tokens.to("cpu", non_blocking=True)
        latents = latents.to("cpu", non_blocking=True)

        if self._buffer_tokens is None:
            # First batch – initialize buffer tensors
            self._buffer_tokens = tokens
            self._buffer_latents = latents
        else:
            self._buffer_tokens = torch.cat([self._buffer_tokens, tokens], dim=0)
            self._buffer_latents = torch.cat([self._buffer_latents, latents], dim=0)

        self._buffer_paths.extend(paths)

        # Flush while we have enough (handles last-overflow edge case)
        while self.buffer_size >= self.shard_size:
            self._flush_shard()

    def flush(self):
        """Flush remaining items in buffer (last incomplete shard)."""
        if self.buffer_size > 0:
            self._flush_shard()

    def _flush_shard(self) -> None:
        """Write the first ``shard_size`` samples from the buffer to disk."""
        # Slice: first ``shard_size`` samples go to disk, remainder stays.
        tokens = self._buffer_tokens[: self.shard_size].to(torch.float16)
        latents = self._buffer_latents[: self.shard_size].to(torch.float16)
        shard_paths = self._buffer_paths[: self.shard_size]
        n = tokens.shape[0]

        # Write to disk
        shard_dir = self.cache_dir / f"shard_{self._shard_id:05d}"
        shard_dir.mkdir(parents=True, exist_ok=True)

        torch.save(tokens, shard_dir / "patch_tokens.pt")
        torch.save(latents, shard_dir / "gt_latents.pt")

        meta = ShardMeta(
            shard_id=self._shard_id,
            n_images=n,
            image_paths=shard_paths,
        )
        with open(shard_dir / "meta.json", "w") as f:
            json.dump(
                {
                    "shard_id": meta.shard_id,
                    "n_images": meta.n_images,
                    "image_paths": meta.image_paths,
                },
                f,
                indent=2,
            )

        print(
            f"  shard_{self._shard_id:05d}: {n} images "
            f"({tokens.numel() * tokens.element_size() / 1e6:.1f} MB)"
        )

        self._shard_id += 1

        # Remove flushed samples from buffer (slice remainder)
        if self._buffer_tokens is not None:
            remainder = self.shard_size
            self._buffer_tokens = self._buffer_tokens[remainder:]
            self._buffer_latents = self._buffer_latents[remainder:]
            self._buffer_paths = self._buffer_paths[remainder:]
        # If buffer was already <= shard_size before slice, remainder is empty.
        # torch.cat on empty would error, so after the above slice the tensors
        # may become size [0, ...].  That's fine – next feed() will re-init.


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    load_dotenv()

    # Resolve dataset path
    dataset_root = Path(str(args.dataset).replace("$DATASET_ROOT", ""))
    cache_dir = args.cache_dir

    print(f"[encode] Dataset: {dataset_root}")
    print(f"[encode] Cache:   {cache_dir}")
    print(f"[encode] Shard:   {args.shard_size} images/shard")
    print(f"[encode] Device:  {args.device}")

    # ── Scan dataset ─────────────────────────────────────────
    print("\n[encode] Scanning dataset...")
    t0 = time.time()
    jpgs = scan_dataset(dataset_root, pattern="**/*.jpg", use_cache=True)
    pngs = scan_dataset(dataset_root, pattern="**/*.png", use_cache=True)
    paths = jpgs + pngs
    total = len(paths)
    print(f"[encode] Found {total:,} images ({time.time() - t0:.1f}s)")

    if total == 0:
        print("[encode] ERROR: No images found!")
        sys.exit(1)

    # ── Estimate ─────────────────────────────────────────────
    shard_count = (total + args.shard_size - 1) // args.shard_size
    est_cache_gb = (
        shard_count * args.shard_size * (1024 * 384 * 2 + 16 * 64 * 64 * 2) / (1024**3)
    )
    print(f"[encode] Estimated: {shard_count} shards, ~{est_cache_gb:.1f} GB cache")

    # ── Check existing cache ─────────────────────────────────
    if cache_dir.exists() and not args.force and not args.dry_run:
        print("\n[encode] Existing cache found")
        try:
            manifest = load_manifest(cache_dir)
            print(f"[encode]   Version: {manifest.version}")
            print(f"[encode]   Dataset hash: {manifest.dataset_hash}")
            print(f"[encode]   Shards: {manifest.shard_count}/{manifest.total_shards}")
        except FileNotFoundError:
            print("[encode]   (manifest missing — will re-encode from start)")
            manifest = None

        if manifest:
            if args.resume and manifest:
                # Check cache validity
                valid = validate_cache(
                    cache_dir, paths, args.dino_model, args.vae_model
                )
                if not valid:
                    print("[encode] Cache is INVALID (dataset/model changed)")
                    print("[encode] Use --force to re-encode")
                    sys.exit(1)

                # Resume from last completed shard
                completed = set(manifest.completed_shards)
                print(
                    f"[encode] Resuming: {len(completed)} shards done, {shard_count - len(completed)} remaining"
                )
                start_shard = max(completed) + 1 if completed else shard_count
            else:
                start_shard = shard_count
        else:
            # Manifest missing — start fresh
            start_shard = 0
    else:
        manifest = None
        start_shard = 0

    # ── Dry run ──────────────────────────────────────────────
    if args.dry_run:
        print("\n[encode] Dry run complete. Cache is valid.")
        return

    # ── Load models ──────────────────────────────────────────
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dino = load_dino(device)
    vae = load_vae(device)
    print(f"[encode] DINOv3: {dino.feature_dim} dim")
    print(f"[encode] VAE: {vae.latent_channels} latent channels")

    # Only load dataset starting from partial shard
    if start_shard > 0:
        start_idx = start_shard * args.shard_size
        remaining = total - start_idx
    else:
        start_idx = 0
        remaining = total

    remaining_paths = paths[start_idx : start_idx + remaining]

    print(f"\n[encode] Encoding shards {start_shard}..{shard_count - 1}")
    t_total = time.time()

    encoder = BatchShardEncoder(
        cache_dir, args.shard_size, args.dino_model, args.vae_model
    )
    if start_shard > 0:
        encoder._shard_id = start_shard

    # Encode in batches. VAE encode is the VRAM bottleneck —
    # tune --batch-size to your GPU memory (16 is safe on 12GB).
    batch_size = args.batch_size
    transform = default_transform()
    dataset = ImageDataset(remaining_paths, transform=transform)

    with torch.no_grad():
        # tqdm: one batch iteration with overall progress bar
        batch_count = (len(dataset) + batch_size - 1) // batch_size
        pbar = tqdm(
            range(batch_count),
            desc="Encoding",
            unit="batches",
            initial=0,
        )
        for batch_idx in pbar:
            i = batch_idx * batch_size
            batch_tensors = [
                dataset[j] for j in range(i, min(i + batch_size, len(dataset)))
            ]

            if not batch_tensors:
                continue

            batch_images = torch.stack(batch_tensors).to(
                device
            )  # [B, 3, 512, 512] in [0, 1]

            # DINOv3 → patch tokens
            patch_tokens = dino.extract(batch_images)  # [B, 1024, 384]

            # VAE encode → gt latents
            gt_latents = vae.encode(batch_images * 2 - 1)  # [B, 16, 64, 64]

            # Encode and flush to disk
            batch_end = min(i + batch_size, len(dataset))
            batch_paths_str = [str(remaining_paths[j]) for j in range(i, batch_end)]
            encoder.feed(patch_tokens, gt_latents, batch_paths_str)

            # Explicitly drop GPU references — encoder.feed() already
            # moved the data to CPU, so these are now harmless.
            del patch_tokens
            del gt_latents
            torch.cuda.empty_cache()

            # Update tqdm with shard info
            elapsed = time.time() - t_total
            shard_num = encoder.shard_id
            pbar.set_postfix(
                {
                    "shards": shard_num,
                    "img": shard_num * args.shard_size,
                    "rate": f"{elapsed / max(shard_num, 1):.1f}s/shard",
                }
            )

    encoder.flush()

    # ── Write manifest ───────────────────────────────────────
    print(f"\n[encode] Encoding complete! {shard_count} shards written.")
    elapsed_total = time.time() - t_total
    print(
        f"[encode] Total time: {elapsed_total:.0f}s ({shard_count / max(elapsed_total, 1) * 60:.1f} shards/min)"
    )

    # Create/update manifest
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = CacheManifest(
        version=1,
        dataset_root=str(dataset_root),
        dataset_hash=compute_dataset_hash(paths),
        model_versions={
            "dino": args.dino_model,
            "vae": args.vae_model,
        },
        total_shards=shard_count,
        shard_count=shard_count,
        completed_shards=list(range(shard_count)),
    )
    save_manifest(
        ShardCacheConfig(cache_dir=cache_dir),
        manifest,
    )
    print(f"[encode] Manifest saved to {cache_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
