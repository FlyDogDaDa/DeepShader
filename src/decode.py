#!/usr/bin/env python3
"""Decode VAE latent samples to images.

Usage:
    # Batch decode all sample_*.pt files (epoch-by-epoch)
    uv run python -m src.decode --samples runs/exp/samples

    # Decode a specific sample file
    uv run python -m src.decode --latent runs/exp/samples/sample_0001.pt

    # CPU decoding (no GPU needed)
    uv run python -m src.decode --samples runs/exp/samples --device cpu

New format: samples/sample_{epoch}.pt is a dict
    {"idx_0": {tokens, gt_latents, pred_latents}, ...}
Each file outputs {epoch}_idx{idx}.png per dataset index.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from dotenv import load_dotenv
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
    load_dotenv()
    torch.backends.cudnn.benchmark = True

    vae = load_vae(args.device)
    print(f"[decode] VAE loaded on {args.device}")

    # ── Determine input files ──────────────────────────────
    # Each entry: (sample_path, out_dir, epoch_stem, fmt)
    latents: list[tuple[Path, Path, str, str]] = []

    if args.latent:
        latents.append(_resolve_single(Path(args.latent), args.output_dir))

    elif args.samples:
        samples_dir = Path(args.samples)
        if not samples_dir.exists():
            print(f"[decode] ERROR: samples directory not found: {samples_dir}")
            return
        latents.extend(_resolve_batch(samples_dir, args.output_dir, args.format))
        print(f"[decode] Found {len(latents)} samples to decode")

    elif args.idx is not None:
        # Try to find samples directory from current directory
        best = _find_samples_dir()
        if best is None:
            print(
                "[decode] ERROR: Could not find samples directory. Use --samples or --latent directly."
            )
            return
        latents.extend(_resolve_batch(best, args.output_dir, args.format))

    else:
        print("[decode] ERROR: Provide --samples, --latent, or --idx")
        return

    if not latents:
        print("[decode] No files to decode.")
        return

    # ── Decode ─────────────────────────────────────────────
    # Each entry: (sample_path, out_dir, epoch_stem, fmt)
    # sample_*.pt contains {"idx_0": {pred_latents: ...}, ...}
    # Output: {out_dir}/idx_{N}/epoch_{epoch}.<fmt> for easy vertical comparison
    decoded = 0
    with torch.no_grad():
        for sample_path, out_dir, epoch_stem, fmt in tqdm(latents, desc="Decoding"):
            data = torch.load(sample_path, map_location=args.device, weights_only=False)
            from torchvision.utils import save_image

            for idx_key, item in sorted(data.items()):
                pred_latents = item["pred_latents"]
                decoded_img = vae.decode(pred_latents)  # [B, 3, 512, 512]
                # Parse idx from "idx_0"
                idx_num = idx_key.split("_")[1]
                # Parse epoch from stem: "sample_0001" → "0001"
                epoch_num = (
                    epoch_stem.split("_")[1] if "_" in epoch_stem else epoch_stem
                )
                # Create idx subdirectory for vertical comparison
                idx_dir = out_dir / f"idx_{idx_num}"
                idx_dir.mkdir(parents=True, exist_ok=True)
                img_path = idx_dir / f"epoch_{epoch_num}.{fmt}"
                save_image(
                    decoded_img,
                    img_path,
                    normalize=True,
                    value_range=(-1, 1),
                )
                decoded += 1

    print(f"[decode] Decoded {decoded} images to {out_dir}/idx_<N>/")


def _resolve_single(
    latents_path: Path, output_dir: str | None, fmt: str
) -> list[tuple[dict[str, torch.Tensor], Path]]:
    """Resolve a single sample_*.pt file to output paths (one per idx)."""
    out_dir = Path(output_dir) if output_dir else latents_path.parent
    # Parse epoch from filename: sample_0001.pt → 0001
    stem = latents_path.stem  # e.g. "sample_0001"
    return [(latents_path, out_dir, stem, fmt)]


def _resolve_batch(
    samples_dir: Path, output_dir: str | None, fmt: str
) -> list[tuple[dict[str, torch.Tensor], Path, str, str]]:
    """Resolve all sample_*.pt files in samples directory."""
    results = []
    out_dir = Path(output_dir) if output_dir else samples_dir

    for sample_path in sorted(samples_dir.glob("sample_*.pt")):
        results.append((sample_path, out_dir, sample_path.stem, fmt))

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
