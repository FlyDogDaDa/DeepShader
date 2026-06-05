"""Mapping networks from DINOv3 patch tokens to FLUX VAE latents.

Two variants:
  * DinoToVAE_Linear — single linear layer (6,160 params for ViT-S)
  * DinoToVAE_MLP    — 4-layer MLP with optional learnable LayerNorm
                       (~302K params, non-linear, better overfit results)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DinoToVAE_Linear(nn.Module):
    """Baseline: single linear layer from patch tokens to latent channels."""

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
        self.project = nn.Linear(hidden_dim, latent_dim)
        self.patch_grid = patch_grid
        self.latent_size = latent_size

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        projected = self.project(patch_tokens)
        b = projected.shape[0]
        out = projected.permute(0, 2, 1).reshape(
            b, self.latent_dim, self.patch_grid, self.patch_grid
        )
        return F.interpolate(
            out, size=self.latent_size, mode="bilinear", align_corners=False
        )


class DinoToVAE_MLP(nn.Module):
    """MLP mapper on patch tokens with optional learnable normalization.

    Architecture:
        [patch_tokens]   → [1024, 384]  (local spatial tokens)
        ↓ Linear + GELU + LayerNorm → [1024, hidden]
        ↓ Linear + GELU + LayerNorm → [1024, hidden]
        ↓ Linear + GELU + LayerNorm → [1024, hidden]
        ↓ Linear + GELU + LayerNorm → [1024, hidden]
        ↓ Linear → [1024, latent]
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
        self.latent_project = nn.Linear(hidden_channels, latent_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """[B, 1024, 384] → [B, 16, 64, 64]."""
        out = self.mlp(patch_tokens)  # [B, 1024, H]
        out = self.latent_project(out)  # [B, 1024, L]
        b = out.shape[0]
        out = out.permute(0, 2, 1).reshape(
            b, self.latent_dim, self.patch_grid, self.patch_grid
        )
        return F.interpolate(
            out, size=self.latent_size, mode="bilinear", align_corners=False
        )


class DinoToVAE_ResNet(nn.Module):
    """Deep ResNet-style mapper: 32 residual blocks + final projection.

    Architecture:
        [patch_tokens]   → [1024, 384]
        ↓ Linear + GELU + LN → [1024, 512]
        ↓ ResBlock × 32  (each: Linear + GELU + LN + residual shortcut)
        ↓ Linear → [1024, 16]
        ↓ reshape + upsample → [B, 16, 64, 64]

    Capacity: ~2.4M params (512-wide × 32 blocks)
    This mapper has much higher capacity than the 4-layer MLP (~300K),
    suitable for complex datasets requiring fine-grained mapping.
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
        self.latent_project = nn.Linear(hidden_channels, latent_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """[B, 1024, 384] → [B, 16, 64, 64]."""
        out = self.blocks(patch_tokens)  # [B, 1024, 512]
        out = self.latent_project(out)  # [B, 1024, 16]
        b = out.shape[0]
        out = out.permute(0, 2, 1).reshape(
            b, self.latent_dim, self.patch_grid, self.patch_grid
        )
        return F.interpolate(
            out, size=self.latent_size, mode="bilinear", align_corners=False
        )


class _ResBlock(nn.Module):
    """Standard residual block: 2× Linear + GELU + LN with shortcut.

    Each block:
        x → Linear → GELU → [LN] → Linear → GELU → [LN] + x
    """

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
    """Transformer encoder block with multi-head self-attention + FFN.

    Designed for the per-token dimension (1024 patches), operating along
    the sequence dimension of [B, 1024, D] tensors.
    """

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
        # Multi-head self-attention with residual
        x_norm = self.attn_norm(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        # FFN with residual
        x = x + self.ffn(self.ffn_norm(x))
        return x


class DinoToVAE_Transformer(nn.Module):
    """Transformer encoder mapper: self-attention over patch tokens + MLP projection.

    Architecture:
        [patch_tokens]   → [B, 1024, 384]
        ↓ LayerNorm
        ↓ Transformer encoder blocks (MHA + FFN with residuals)
        ↓ LayerNorm
        ↓ Linear → [B, 1024, 16]
        ↓ reshape + upsample → [B, 16, 64, 64]

    This allows each patch token to attend to all others, capturing
    contextual / spatial relationships that per-token MLPs miss.

    Capacity: ~390K params (8 attention heads × 1 block)
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

        # Project per-token features to latent channels
        self.latent_project = nn.Linear(hidden_dim, latent_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """[B, 1024, 384] → [B, 16, 64, 64]."""
        out = self.transformer(patch_tokens)  # [B, 1024, 384]
        out = self.latent_project(out)  # [B, 1024, 16]
        b = out.shape[0]
        out = out.permute(0, 2, 1).reshape(
            b, self.latent_dim, self.patch_grid, self.patch_grid
        )
        return F.interpolate(
            out, size=self.latent_size, mode="bilinear", align_corners=False
        )


class DinoToVAE_TransformerResNet(nn.Module):
    """Transformer encoder + ResNet mapper: attention for context + deep residual blocks.

    Architecture:
        [patch_tokens]   → [B, 1024, 384]
        ↓ LayerNorm
        ↓ Transformer encoder blocks (MHA + FFN)
        ↓ Linear + GELU + LN → [B, 1024, 512]
        ↓ ResBlock × N (per-token residual layers)
        ↓ Linear → [B, 1024, 16]
        ↓ reshape + upsample → [B, 16, 64, 64]

    Combines global context from self-attention with representational
    capacity from deep residual blocks.

    Capacity: ~2.8M params (512-wide × 32 ResNet blocks + transformer)
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

        self.latent_project = nn.Linear(hidden_channels, latent_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """[B, 1024, 384] → [B, 16, 64, 64]."""
        out = self.transformer(patch_tokens)  # [B, 1024, 384]
        out = self.resnet(out)  # [B, 1024, 512]
        out = self.latent_project(out)  # [B, 1024, 16]
        b = out.shape[0]
        out = out.permute(0, 2, 1).reshape(
            b, self.latent_dim, self.patch_grid, self.patch_grid
        )
        return F.interpolate(
            out, size=self.latent_size, mode="bilinear", align_corners=False
        )
