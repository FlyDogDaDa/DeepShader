"""Mapping networks from DINOv3 patch tokens to FLUX VAE latents.

Probabilistic variants (mean + logvar output for KL divergence loss):
  * DinoToVAE_Linear — single linear layer (6,160 params for ViT-S)
  * DinoToVAE_MLP    — 4-layer MLP with optional learnable LayerNorm
                       (~302K params, non-linear, better overfit results)

Deterministic variants (mean-only output, for eval / decoding):
  * DinoToVAE_LinearDet — deterministic linear
  * DinoToVAE_MLPDet    — deterministic MLP

"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _reshape_and_upsample(
    mean: torch.Tensor, latent_dim: int, patch_grid: int, latent_size: int
) -> torch.Tensor:
    """Reshape [B, N, C] → [B, C, H, W] with bilinear interpolation.

    Args:
        mean: Per-token mean [B, 1024, latent_dim].
        latent_dim: Number of latent channels (16).
        patch_grid: Spatial grid size before upsample (32 for 512×512).
        latent_size: Target spatial size (64 for 512×512, downsample=8).

    Returns:
        Latent tensor [B, latent_dim, latent_size, latent_size].
    """
    b = mean.shape[0]
    out = mean.permute(0, 2, 1).reshape(b, latent_dim, patch_grid, patch_grid)
    return F.interpolate(out, size=latent_size, mode="bilinear", align_corners=False)


def kl_divergence(mean_m, logvar_m, mean_v, logvar_v) -> torch.Tensor:
    """Compute KL( Normal(mean_m, logvar_m) || Normal(mean_v, logvar_v) ).

    Computes the mean over all elements (batch × channels).

    KL = 0.5 * Σ [ log(var_v/var_m) + (var_m + (μ_m - μ_v)²)/var_v - 1 ]

    Args:
        mean_m: Mapper posterior mean [B, C, H, W].
        logvar_m: Mapper posterior log-variance.
        mean_v: Target (VAE) posterior mean [B, C, H, W].
        logvar_v: Target log-variance.

    Returns:
        Scalar KL loss (mean over all elements).
    """
    var_m = logvar_m.exp()
    var_v = logvar_v.exp()

    kl = 0.5 * (
        torch.log(var_v)
        - torch.log(var_m)
        + (var_m + (mean_m - mean_v) ** 2) / var_v
        - 1
    )
    return kl.mean()


# ── Probabilistic Mappers ────────────────────────────────────────────


class DinoToVAE_Linear(nn.Module):
    """Probabilistic linear mapper: outputs mean + logvar for KL divergence.

    Forward returns (mean, logvar, z) where z uses reparameterization
    during training and z=mean during evaluation.
    """

    def __init__(
        self,
        hidden_dim: int = 384,
        latent_dim: int = 16,
        patch_size: int = 16,
        image_size: int = 512,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        patch_grid = image_size // patch_size
        latent_size = patch_grid * 2
        # Split into two projections for mean and logvar
        self.mean_project = nn.Linear(hidden_dim, latent_dim)
        self.logvar_project = nn.Linear(hidden_dim, latent_dim)
        self.patch_grid = patch_grid
        self.latent_size = latent_size

    def forward(
        self, patch_tokens: torch.Tensor, sample: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            patch_tokens: [B, 1024, hidden_dim].
            sample: If True, reparameterize from the predicted distribution.
                    If False, use mean only (for evaluation).

        Returns:
            Tuple of (latent_z, mean, logvar), each [B, latent_dim, latent_size, latent_size].
        """
        projected = self.mean_project(patch_tokens)
        logvar = self.logvar_project(patch_tokens)

        if sample:
            std = (0.5 * logvar).exp()
            z = projected + std * torch.randn_like(projected)
        else:
            z = projected

        return (
            _reshape_and_upsample(
                z, self.latent_dim, self.patch_grid, self.latent_size
            ),
            _reshape_and_upsample(
                projected, self.latent_dim, self.patch_grid, self.latent_size
            ),
            _reshape_and_upsample(
                logvar, self.latent_dim, self.patch_grid, self.latent_size
            ),
        )


class DinoToVAE_MLP(nn.Module):
    """Probabilistic MLP mapper: outputs mean + logvar for KL divergence.

    Architecture:
        [patch_tokens]   → [1024, 384]  (local spatial tokens)
        ↓ Linear + GELU + LayerNorm → [1024, hidden]
        ↓ Linear + GELU + LayerNorm → [1024, hidden]
        ↓ Linear + GELU + LayerNorm → [1024, hidden]
        ↓ Linear + GELU + LayerNorm → [1024, hidden]
        ↓ Linear → [1024, latent]     (mean)
        ↓ Linear → [1024, latent]     (logvar)
        ↓ reshape + upsample → [B, latent, 64, 64]
    """

    def __init__(
        self,
        hidden_dim: int = 384,
        latent_dim: int = 16,
        patch_size: int = 16,
        image_size: int = 512,
        hidden_channels: int = 256,
        num_layers: int = 4,
        learnable_norm: bool = True,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        patch_grid = image_size // patch_size
        latent_size = patch_grid * 2
        self.patch_grid = patch_grid
        self.latent_size = latent_size
        self.learnable_norm = learnable_norm

        layers = []
        for i in range(num_layers):
            in_ch = hidden_dim if i == 0 else hidden_channels
            layers.append(nn.Linear(in_ch, hidden_channels))
            if learnable_norm:
                layers.append(nn.LayerNorm(hidden_channels))
            layers.append(nn.GELU())

        self.mlp = nn.Sequential(*layers)
        # Split into two projections for mean and logvar
        self.mean_project = nn.Linear(hidden_channels, latent_dim)
        self.logvar_project = nn.Linear(hidden_channels, latent_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, patch_tokens: torch.Tensor, sample: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            Tuple of (latent_z, mean, logvar).
        """
        out = self.mlp(patch_tokens)  # [B, 1024, H]

        mean = self.mean_project(out)
        logvar = self.logvar_project(out)

        if sample:
            std = (0.5 * logvar).exp()
            z = mean + std * torch.randn_like(mean)
        else:
            z = mean

        return (
            _reshape_and_upsample(
                z, self.latent_dim, self.patch_grid, self.latent_size
            ),
            _reshape_and_upsample(
                mean, self.latent_dim, self.patch_grid, self.latent_size
            ),
            _reshape_and_upsample(
                logvar, self.latent_dim, self.patch_grid, self.latent_size
            ),
        )


class DinoToVAE_ResNet(nn.Module):
    """Probabilistic ResNet mapper: outputs mean + logvar for KL divergence.

    Architecture:
        [patch_tokens]   → [1024, 384]
        ↓ Linear + GELU + LN → [1024, 512]
        ↓ ResBlock × 32
        ↓ Linear → [1024, latent]     (mean)
        ↓ Linear → [1024, latent]     (logvar)
        ↓ reshape + upsample → [B, 16, 64, 64]
    """

    def __init__(
        self,
        hidden_dim: int = 384,
        latent_dim: int = 16,
        patch_size: int = 16,
        image_size: int = 512,
        hidden_channels: int = 512,
        num_layers: int = 32,
        learnable_norm: bool = True,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        patch_grid = image_size // patch_size
        latent_size = patch_grid * 2
        self.patch_grid = patch_grid
        self.latent_size = latent_size
        self.learnable_norm = learnable_norm

        # Project input to wider width
        layers = [
            nn.Linear(hidden_dim, hidden_channels),
            nn.GELU(),
            nn.LayerNorm(hidden_channels),
        ]

        # Residual blocks
        for _ in range(num_layers):
            layers.append(_ResBlock(hidden_channels, learnable_norm))

        self.blocks = nn.Sequential(*layers)
        # Split into two projections for mean and logvar
        self.mean_project = nn.Linear(hidden_channels, latent_dim)
        self.logvar_project = nn.Linear(hidden_channels, latent_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, patch_tokens: torch.Tensor, sample: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            Tuple of (latent_z, mean, logvar).
        """
        out = self.blocks(patch_tokens)  # [B, 1024, 512]

        mean = self.mean_project(out)
        logvar = self.logvar_project(out)

        if sample:
            std = (0.5 * logvar).exp()
            z = mean + std * torch.randn_like(mean)
        else:
            z = mean

        return (
            _reshape_and_upsample(
                z, self.latent_dim, self.patch_grid, self.latent_size
            ),
            _reshape_and_upsample(
                mean, self.latent_dim, self.patch_grid, self.latent_size
            ),
            _reshape_and_upsample(
                logvar, self.latent_dim, self.patch_grid, self.latent_size
            ),
        )


# ── Transformer Mappers ──────────────────────────────────────────────


class DinoToVAE_Transformer(nn.Module):
    """Probabilistic Transformer mapper: outputs mean + logvar for KL divergence.

    Architecture:
        [patch_tokens]   → [B, 1024, 384]
        ↓ LayerNorm
        ↓ Transformer encoder blocks (MHA + FFN with residuals)
        ↓ LayerNorm
        ↓ Linear → [B, 1024, latent]     (mean)
        ↓ Linear → [B, 1024, latent]     (logvar)
        ↓ reshape + upsample → [B, 16, 64, 64]
    """

    def __init__(
        self,
        hidden_dim: int = 384,
        latent_dim: int = 16,
        patch_size: int = 16,
        image_size: int = 512,
        num_heads: int = 8,
        num_layers: int = 1,
        mlp_ratio: float = 4.0,
        learnable_norm: bool = True,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        patch_grid = image_size // patch_size
        latent_size = patch_grid * 2
        self.patch_grid = patch_grid
        self.latent_size = latent_size

        norm_cls = nn.LayerNorm if learnable_norm else nn.Identity

        layers = []
        if learnable_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        for _ in range(num_layers):
            layers.append(
                _TransformerBlock(
                    dim=hidden_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    learnable_norm=learnable_norm,
                )
            )
        if learnable_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        self.transformer = nn.Sequential(*layers)

        # Split into two projections for mean and logvar
        self.mean_project = nn.Linear(hidden_dim, latent_dim)
        self.logvar_project = nn.Linear(hidden_dim, latent_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, patch_tokens: torch.Tensor, sample: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            Tuple of (latent_z, mean, logvar).
        """
        out = self.transformer(patch_tokens)  # [B, 1024, 384]

        mean = self.mean_project(out)
        logvar = self.logvar_project(out)

        if sample:
            std = (0.5 * logvar).exp()
            z = mean + std * torch.randn_like(mean)
        else:
            z = mean

        return (
            _reshape_and_upsample(
                z, self.latent_dim, self.patch_grid, self.latent_size
            ),
            _reshape_and_upsample(
                mean, self.latent_dim, self.patch_grid, self.latent_size
            ),
            _reshape_and_upsample(
                logvar, self.latent_dim, self.patch_grid, self.latent_size
            ),
        )


class DinoToVAE_TransformerResNet(nn.Module):
    """Probabilistic Transformer+ResNet hybrid mapper: outputs mean + logvar.

    Architecture:
        [patch_tokens]   → [B, 1024, 384]
        ↓ Transformer encoder blocks
        ↓ Linear + GELU + LN → [B, 1024, 512]
        ↓ ResBlock × N
        ↓ Linear → [B, 1024, latent]     (mean)
        ↓ Linear → [B, 1024, latent]     (logvar)
        ↓ reshape + upsample → [B, 16, 64, 64]
    """

    def __init__(
        self,
        hidden_dim: int = 384,
        latent_dim: int = 16,
        patch_size: int = 16,
        image_size: int = 512,
        hidden_channels: int = 512,
        num_heads: int = 8,
        num_transformer_layers: int = 1,
        num_resnet_layers: int = 32,
        mlp_ratio: float = 4.0,
        learnable_norm: bool = True,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        patch_grid = image_size // patch_size
        latent_size = patch_grid * 2
        self.patch_grid = patch_grid
        self.latent_size = latent_size

        # ── Transformer encoder for context ──────────────────
        t_layers = []
        if learnable_norm:
            t_layers.append(nn.LayerNorm(hidden_dim))
        for _ in range(num_transformer_layers):
            t_layers.append(
                _TransformerBlock(
                    dim=hidden_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    learnable_norm=learnable_norm,
                )
            )
        if learnable_norm:
            t_layers.append(nn.LayerNorm(hidden_dim))
        self.transformer = nn.Sequential(*t_layers)

        # ── ResNet blocks on top of attended features ────────
        res_layers = [
            nn.Linear(hidden_dim, hidden_channels),
            nn.GELU(),
            nn.LayerNorm(hidden_channels),
        ]
        for _ in range(num_resnet_layers):
            res_layers.append(_ResBlock(hidden_channels, learnable_norm))
        self.resnet = nn.Sequential(*res_layers)

        # Split into two projections for mean and logvar
        self.mean_project = nn.Linear(hidden_channels, latent_dim)
        self.logvar_project = nn.Linear(hidden_channels, latent_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, patch_tokens: torch.Tensor, sample: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            Tuple of (latent_z, mean, logvar).
        """
        out = self.transformer(patch_tokens)  # [B, 1024, 384]
        out = self.resnet(out)  # [B, 1024, 512]

        mean = self.mean_project(out)
        logvar = self.logvar_project(out)

        if sample:
            std = (0.5 * logvar).exp()
            z = mean + std * torch.randn_like(mean)
        else:
            z = mean

        return (
            _reshape_and_upsample(
                z, self.latent_dim, self.patch_grid, self.latent_size
            ),
            _reshape_and_upsample(
                mean, self.latent_dim, self.patch_grid, self.latent_size
            ),
            _reshape_and_upsample(
                logvar, self.latent_dim, self.patch_grid, self.latent_size
            ),
        )


# ── Residual / Transformer Blocks ────────────────────────────────────


class _ResBlock(nn.Module):
    """Standard residual block: 2× Linear + GELU + LN with shortcut."""

    def __init__(self, dim: int, learnable_norm: bool = True):
        super().__init__()
        layers = []
        layers.append(nn.Linear(dim, dim))
        layers.append(nn.GELU())
        if learnable_norm:
            layers.append(nn.LayerNorm(dim))
        layers.append(nn.Linear(dim, dim))
        layers.append(nn.GELU())
        if learnable_norm:
            layers.append(nn.LayerNorm(dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + x


class _TransformerBlock(nn.Module):
    """Transformer encoder block with multi-head self-attention + FFN."""

    def __init__(
        self,
        dim: int = 384,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        learnable_norm: bool = True,
    ):
        super().__init__()
        attn_norm_cls = nn.LayerNorm if learnable_norm else nn.Identity
        self.attn_norm = attn_norm_cls(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.ffn_norm = attn_norm_cls(dim)
        hidden_ffn = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_ffn),
            nn.GELU(),
            nn.Linear(hidden_ffn, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.attn_norm(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x
