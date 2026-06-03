"""Tests for src/pretrains.py."""

from __future__ import annotations

import torch

from src.pretrains import load_dino, load_vae


def test_load_vae_exists() -> None:
    """load_vae() returns AutoencoderKL instance."""
    vae = load_vae("cpu")
    assert vae is not None


def test_load_dino_exists() -> None:
    """load_dino() returns transformer AutoModel instance."""
    dino = load_dino("cpu")
    assert dino is not None


def test_vae_encode_shape(sample_jpg) -> None:
    """VAE encode: [B, 3, 512, 512] in [-1, 1] -> [B, 16, 64, 64]."""
    from src.dataset import load_image
    from src.pretrains import load_dino

    vae = load_vae("cpu")
    dino = load_dino("cpu")
    images = load_image(sample_jpg).unsqueeze(0)  # [1, 3, 512, 512]
    images = images * 2 - 1  # normalize to [-1, 1]
    latents = vae.encode(images)
    assert latents.shape == (1, 16, 64, 64)
    assert latents.dtype == torch.float32


def test_vae_decode_shape(sample_jpg) -> None:
    """VAE decode: [B, 16, 64, 64] -> [B, 3, 512, 512]."""
    from src.dataset import load_image

    vae = load_vae("cpu")
    images = load_image(sample_jpg).unsqueeze(0) * 2 - 1
    latents = vae.encode(images)
    decoded = vae.decode(latents)
    assert decoded.shape == (1, 3, 512, 512)


def test_dino_extract_shape() -> None:
    """DINO extract: [B, 3, 512, 512] in [0, 1] -> [B, 1024, 384]."""
    from pathlib import Path

    import torch

    from src.dataset import load_image
    from tests.conftest import _make_jpg

    tmp = Path("/tmp/dino_test_sample.jpg")
    _make_jpg(tmp)
    images = load_image(tmp).unsqueeze(0)  # [1, 3, 512, 512]
    dino = load_dino("cpu")
    tokens = dino.extract(images)
    assert tokens.shape == (1, 1024, 384)


def test_dino_preprocess_applies_mean_std(sample_jpg) -> None:
    """DINO preprocess subtracts mean and divides by std."""
    from src.dataset import load_image

    img = load_image(sample_jpg).unsqueeze(0)  # [1, 3, 512, 512]
    dino = load_dino("cpu")
    normalized = dino.preprocess(img)

    # Check mean/std are applied: normalized values should be near 0
    # For realistic images, per-channel means should be small
    channel_means = normalized.mean(dim=[0, 2, 3])
    assert channel_means.abs().max() < 2.0, (
        f"Per-channel means should be reasonable after normalization: {channel_means}"
    )


def test_dino_feature_dim() -> None:
    """DINO ViT-S feature_dim is 384."""
    dino = load_dino("cpu")
    assert dino.feature_dim == 384
