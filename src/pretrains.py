"""Pretrained model wrappers (VAE, DINOv3) and image preprocessing.

Provides convenient factory functions + wrapper classes:
    * ``load_vae(device)``  → returns ``VAEModel``
    * ``load_dino(device)`` → returns ``DINOModel``

Both wrappers hide HuggingFace internals so training code stays clean.
"""

from __future__ import annotations

import torch
from diffusers import AutoencoderKL
from transformers import AutoModel

# ── Default constants ─────────────────────────────────────────────

# DINOv3 mean / std (ImageNet default)
_DINO_MEAN = [0.485, 0.456, 0.406]
_DINO_STD = [0.229, 0.224, 0.225]

# FLUX VAE repo
_VAE_REPO = "advokat/AnimePro-FLUX"
# DINOv3 ViT-S pretrain
_DINO_REPO = "facebook/dinov3-vits16-pretrain-lvd1689m"


# ── VAE Wrapper ───────────────────────────────────────────────────


class VAEModel:
    """Thin wrapper around HuggingFace AutoencoderKL."""

    def __init__(self, vae: AutoencoderKL) -> None:
        self._vae = vae

    @property
    def latent_channels(self) -> int:
        return self._vae.config.latent_channels

    @property
    def device(self) -> torch.device:
        return next(self._vae.parameters()).device

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images ``[B, 3, H, W]`` in ``[-1, 1]`` → latents ``[B, 16, 64, 64]``.

        Args:
            images: RGB images scaled to ``[-1, 1]``.

        Returns:
            Latent tensor (mode of the VAE posterior).
        """
        return self._vae.encode(images).latent_dist.mode()

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents ``[B, 16, 64, 64]`` → images ``[B, 3, H, W]`` in ``[-1, 1]``.

        Args:
            latents: VAE latent codes.

        Returns:
            Image tensor in ``[-1, 1]``.
        """
        return self._vae.decode(latents).sample


def load_vae(device: str | torch.device = "cuda") -> VAEModel:
    """Load the AnimePro-FLUX VAE.

    Args:
        device: ``"cuda"`` or ``"cpu"``.

    Returns:
        Wrapped ``VAEModel`` in ``eval()`` mode.
    """
    vae = AutoencoderKL.from_pretrained(_VAE_REPO, subfolder="vae").to(device)
    vae.eval()
    return VAEModel(vae)


# ── DINOv3 Wrapper ────────────────────────────────────────────────


class DINOModel:
    """Thin wrapper around HuggingFace ``AutoModel`` for DINOv3."""

    def __init__(
        self,
        model: AutoModel,
        mean: list[float] | None = None,
        std: list[float] | None = None,
    ) -> None:
        self._model = model
        self._mean = torch.tensor(mean or _DINO_MEAN).view(1, 3, 1, 1)
        self._std = torch.tensor(std or _DINO_STD).view(1, 3, 1, 1)

    @property
    def feature_dim(self) -> int:
        return self._model.config.hidden_size

    @property
    def device(self) -> torch.device:
        return next(self._model.parameters()).device

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """Normalize images ``[B, 3, H, W]`` in ``[0, 1]`` for DINOv3.

        Args:
            images: RGB images scaled to ``[0, 1]``.

        Returns:
            Normalized tensor ``((x - mean) / std)``.
        """
        return (images - self._mean.to(images.device)) / self._std.to(images.device)

    def extract(self, images: torch.Tensor, skip_tokens: int = 5) -> torch.Tensor:
        """Run DINOv3 forward and return patch tokens.

        Args:
            images: ``[B, 3, 512, 512]`` in ``[0, 1]``.
            skip_tokens: Number of leading tokens to discard
                         (class token + position embeddings; default 5).

        Returns:
            Patch tokens ``[B, N, hidden_dim]`` (no class/pos tokens).
        """
        normalized = self.preprocess(images)
        with torch.no_grad():
            out = self._model(normalized)
        del normalized
        # Skip class token + position embeddings
        return out.last_hidden_state[:, skip_tokens:, :]


def load_dino(device: str | torch.device = "cuda") -> DINOModel:
    """Load DINOv3 ViT-S pretrain model.

    Args:
        device: ``"cuda"`` or ``"cpu"``.

    Returns:
        Wrapped ``DINOModel`` in ``eval()`` mode.
    """
    model = AutoModel.from_pretrained(_DINO_REPO).to(device)
    model.eval()
    return DINOModel(model)


# ── Shared preprocessor constants (exposed for convenience) ──────

DINO_MEAN = _DINO_MEAN
DINO_STD = _DINO_STD
