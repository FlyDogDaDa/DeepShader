#!/usr/bin/env python3
"""Decode VAE latent samples to images.

Usage:
    # Batch decode all samples from a checkpoint's samples directory
    uv run python -m src.decode --samples runs/exp/samples

    # Decode specific index
    uv run python -m src.decode --idx 0

    # Decode a specific pred_latents file
    uv run python -m src.decode --latent runs/exp/samples/idx_00000/pred_latents.pt

    # CPU decoding (no GPU needed)
    uv run python -m src.decode --samples runs/exp/samples --device cpu
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from src import load_vae


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Decode VAE latents to images")

    g = p.add_argument_group("input")
    g.add_argument(
        "--samples",
        type=str,
        default=None,
        help="Path to samples/ directory (decode all pred_latents.pt)",
    )
    g.add_argument(
        "--latent",
        type=str,
        default=None,
        help="Path to a single pred_latents.pt file",
    )
    g.add_argument(
        "--idx",
        type=int,
        default=None,
        help="Decode a single dataset index from runs/<name>/samples/idx_XXXXX/",
    )
    g.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for decoded images (default: same as input)",
    )

    g = p.add_argument_group("options")
    g.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run decoding on (default: cuda)",
    )
    g.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "jpg"],
        help="Output image format",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.backends.cudnn.benchmark = True

    vae = load_vae(args.device)
    print(f"[decode] VAE loaded on {args.device}")

    # ── Determine input files ──────────────────────────────
    latents: list[tuple[Path, Path]] = []  # (pred_latents_path, output_path)

    if args.latent:
        latents.append(_resolve_single(Path(args.latent), args.output_dir))

    elif args.samples:
        samples_dir = Path(args.samples)
        if not samples_dir.exists():
            print(f"[decode] ERROR: samples directory not found: {samples_dir}")
            return
        latents.extend(_resolve_batch(samples_dir, args.output_dir))

    elif args.idx is not None:
        # Try to find samples directory from current directory
        best = _find_samples_dir()
        if best is None:
            print(
                "[decode] ERROR: Could not find samples directory. Use --samples or --latent directly."
            )
            return
        latents.extend(_resolve_batch(best, args.output_dir))

    else:
        print("[decode] ERROR: Provide --samples, --latent, or --idx")
        return

    if not latents:
        print("[decode] No files to decode.")
        return

    # ── Decode ─────────────────────────────────────────────
    decoded = 0
    with torch.no_grad():
        for latents_path, out_path in tqdm(latents, desc="Decoding"):
            pred = torch.load(latents_path, map_location=args.device, weights_only=True)
            decoded_img = vae.decode(pred)  # [B, 3, 512, 512]
            from torchvision.utils import save_image

            save_image(
                decoded_img,
                out_path,
                normalize=True,
                value_range=(-1, 1),
            )
            decoded += 1

    print(f"[decode] Decoded {decoded} images to {out_path.parent}")


def _resolve_single(latents_path: Path, output_dir: str | None) -> tuple[Path, Path]:
    """Resolve a single latent file to output path."""
    out_dir = Path(output_dir) if output_dir else latents_path.parent
    name = latents_path.stem  # e.g. "pred_latents"
    out_path = out_dir / f"{name}_decoded.png"
    return latents_path, out_path


def _resolve_batch(
    samples_dir: Path, output_dir: str | None
) -> list[tuple[Path, Path]]:
    """Resolve all pred_latents.pt files in samples directory tree."""
    results = []
    out_dir = Path(output_dir) if output_dir else samples_dir

    for latents_path in sorted(samples_dir.rglob("pred_latents.pt")):
        # Compute output path: samples/idx_00000/pred_latents.pt
        # → samples/idx_00000_decoded.png
        idx_dir = latents_path.parent
        rel = idx_dir.relative_to(samples_dir)
        out_path = out_dir / f"{rel}_decoded.png"
        results.append((latents_path, out_path))

    return results


def _find_samples_dir() -> Path | None:
    """Find the most recent samples/ directory in runs/ subdirectories."""
    import os

    for root, dirs, files in os.walk("."):
        if "samples" in dirs:
            return Path(root) / "samples"
    return None


if __name__ == "__main__":
    main()
