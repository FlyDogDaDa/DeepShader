"""Test mapping network: DINOv3 patch tokens → FLUX VAE latents.

Architecture:
    [batch, seq_len=1024, hidden_dim=384]  (DINOv3 ViT-S patch tokens)
         ↓ Linear(384 → 16)
    [batch, seq_len=1024, latent_dim=16]
         ↓ reshape
    [batch, latent_dim=16, spatial=32, spatial=32]
         ↑ interpolate(2x)
    [batch, latent_dim=16, spatial=64, spatial=64]  (FLUX VAE latent space)

This is the ONLY trainable component in the full pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DinoToVAE(nn.Module):
    """Lightweight mapping from DINOv3 patch tokens to VAE latents.

    Args:
        hidden_dim: DINOv3 embedding dimension (384 for ViT-S, 768 for ViT-B, 1024 for ViT-L)
        latent_dim: FLUX VAE latent channels (always 16)
        patch_size: DINOv3 patch size (always 16)
        image_size: Input image size (512 for our dataset)
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

        # Calculate spatial dimensions
        patch_grid = image_size // patch_size  # 512 / 16 = 32
        latent_size = patch_grid * 2  # 64 (VAE downsampling = 8, so 512/8 = 64)

        # Projection layer: hidden_dim → latent_dim
        self.project = nn.Linear(hidden_dim, latent_dim)

        # Store shape info
        self.patch_grid = patch_grid  # 32
        self.latent_size = latent_size  # 64
        self.seq_len = patch_grid * patch_grid  # 1024

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """Map patch tokens to VAE latents.

        Args:
            patch_tokens: [batch, seq_len, hidden_dim] from DINOv3 (after removing
                         class token and register tokens)

        Returns:
            latents: [batch, latent_dim, latent_size, latent_size] ready for VAE decode
        """
        # Step 1: Linear projection
        # [batch, seq_len, hidden_dim] → [batch, seq_len, latent_dim]
        projected = self.project(patch_tokens)

        # Step 2: Reshape to 2D spatial
        # [batch, seq_len, latent_dim] → [batch, latent_dim, grid, grid]
        batch_size = projected.shape[0]
        reshaped = projected.permute(0, 2, 1).reshape(
            batch_size, self.latent_dim, self.patch_grid, self.patch_grid
        )

        # Step 3: Upsample 2x to match VAE latent size
        # [batch, latent_dim, 32, 32] → [batch, latent_dim, 64, 64]
        latents = F.interpolate(
            reshaped,
            size=self.latent_size,
            mode="bilinear",
            align_corners=False,
        )

        return latents


def test_mapping():
    """Test the mapping network with dummy data."""
    print("=" * 60)
    print("Test: DINOv3 patch tokens → VAE latents")
    print("=" * 60)

    # Simulate DINOv3 ViT-S patch tokens: [1, 1024, 384]
    batch_size = 1
    seq_len = 1024  # 32×32 patch grid for 512×512 input
    hidden_dim = 384  # ViT-S

    patch_tokens = torch.randn(batch_size, seq_len, hidden_dim)
    print(f"\nInput: DINOv3 patch tokens = {patch_tokens.shape}")

    # Create mapping network
    mapper = DinoToVAE(hidden_dim=384, latent_dim=16)
    print(f"Parameters: {sum(p.numel() for p in mapper.parameters()):,}")

    # Forward pass
    with torch.no_grad():
        latents = mapper(patch_tokens)

    print(f"Output: VAE latents = {latents.shape}")
    print(
        f"Expected: [1, 16, 64, 64] ✓"
        if latents.shape == (1, 16, 64, 64)
        else f"Expected: [1, 16, 64, 64] ✗"
    )

    # Test with ViT-B and ViT-L dimensions too
    print("\n--- ViT-B (768 dim) ---")
    mapper_b = DinoToVAE(hidden_dim=768, latent_dim=16)
    tokens_b = torch.randn(batch_size, seq_len, 768)
    with torch.no_grad():
        latents_b = mapper_b(tokens_b)
    print(f"Parameters: {sum(p.numel() for p in mapper_b.parameters()):,}")
    print(
        f"Output: {latents_b.shape} {'✓' if latents_b.shape == (1, 16, 64, 64) else '✗'}"
    )

    print("\n--- ViT-L (1024 dim) ---")
    mapper_l = DinoToVAE(hidden_dim=1024, latent_dim=16)
    tokens_l = torch.randn(batch_size, seq_len, 1024)
    with torch.no_grad():
        latents_l = mapper_l(tokens_l)
    print(f"Parameters: {sum(p.numel() for p in mapper_l.parameters()):,}")
    print(
        f"Output: {latents_l.shape} {'✓' if latents_l.shape == (1, 16, 64, 64) else '✗'}"
    )

    print("\n" + "=" * 60)
    print("Mapping network test complete ✓")
    print("=" * 60)

    return mapper


if __name__ == "__main__":
    test_mapping()
