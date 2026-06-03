"""Mini-batch training PoC for DinoToVAE.

Validates the training pipeline:
    Input batch → DINOv3 → patch tokens → DinoToVAE → latents → VAE decode → output
    Compare decoded output with ground truth → compute loss → backward → update weights

Keeps everything frozen except DinoToVAE (6,160 params for ViT-S).

Key design: use .detach() after each stage to break the computation graph.
This ensures backward() only computes gradients through the mapper,
not through DINOv3 or VAE (which don't need gradients).
"""

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL
from dotenv import load_dotenv
from PIL import Image
from torchvision.transforms import v2
from transformers import AutoModel

from src.utility import scan_dataset

load_dotenv()


# ── DinoToVAE mapping network ──────────────────────────────────────


class DinoToVAE(nn.Module):
    """Lightweight mapping from DINOv3 patch tokens to VAE latents."""

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
        batch_size = projected.shape[0]
        reshaped = projected.permute(0, 2, 1).reshape(
            batch_size, self.latent_dim, self.patch_grid, self.patch_grid
        )
        latents = F.interpolate(
            reshaped, size=self.latent_size, mode="bilinear", align_corners=False
        )
        return latents


# ── Training step ─────────────────────────────────────────────────


def training_step(
    dino_model, vae, mapper, batch_data, device, latent_loss: bool = True
):
    """One forward+backward step.

    Architecture (exactly as discussed):

        pred_latents = mapper(patch_tokens)       # train
        gt_latents = vae.encode(gt_image).mode()  # no_grad
        loss = MSE(pred_latents, gt_latents)      # backward → mapper only

    VAE decode is never called during training — it's only for inference.
    This saves VRAM and time.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    # Load images [B, 3, 512, 512] in [0, 1]
    images = torch.stack(
        [
            v2.ToImage()(Image.open(p).convert("RGB").resize((512, 512)))
            for p in batch_data
        ]
    ).to(device)

    # ── Stage 1: DINOv3 (no_grad) → patch_tokens ──────────────────
    images_norm = (images - mean) / std
    with torch.no_grad():
        patch_tokens = dino_model(images_norm).last_hidden_state[:, 5:, :]
    del images_norm

    # ── Stage 2: Mapper → pred_latents ─────────────────────────────
    pred_latents = mapper(patch_tokens)

    # ── Stage 3: VAE encode (no_grad) → gt_latents ─────────────────
    # Ground truth latent, taken from mode (mean).  No graph.
    with torch.no_grad():
        gt_latents = vae.encode(images.float() * 2 - 1).latent_dist.mode()
    del images  # free original images

    # ── Loss: MSE between pred and gt latents ──────────────────────
    # backward() flows ONLY through mapper (vae.encode and dino_model
    # are inside .no_grad() blocks).
    loss = F.mse_loss(pred_latents, gt_latents)
    return loss, pred_latents


# ── Main ───────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 60)
    print("Mini-batch Training PoC: DinoToVAE")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load models
    vae = AutoencoderKL.from_pretrained("advokat/AnimePro-FLUX", subfolder="vae").to(
        device
    )
    vae.eval()
    print(f"  VAE loaded: latent_channels={vae.config.latent_channels}")

    dino_model = AutoModel.from_pretrained(
        "facebook/dinov3-vits16-pretrain-lvd1689m"
    ).to(device)
    dino_model.eval()
    print(f"  DINOv3 ViT-S loaded: hidden_dim={dino_model.config.hidden_size}")

    # Mapper (trainable)
    mapper = DinoToVAE(hidden_dim=384, latent_dim=16).to(device)
    print(f"  DinoToVAE params: {sum(p.numel() for p in mapper.parameters()):,}")
    print(f"  VAE .requires_grad: {next(vae.parameters()).requires_grad}")
    print(f"  DINOv3 .requires_grad: {next(dino_model.parameters()).requires_grad}")

    optimizer = torch.optim.AdamW(mapper.parameters(), lr=1e-3, weight_decay=1e-4)

    # Dataset: small subset for PoC
    dataset_root = (
        Path(os.environ["DATASET_ROOT"]) / "danbooru-images" / "danbooru-images"
    )
    all_paths = scan_dataset(dataset_root)
    batch_size = 8
    sample_size = 16
    sample_paths = all_paths[:sample_size]
    print(f"  Dataset: {sample_size} images from {len(all_paths)} total")

    # Training
    epochs = 10
    print(f"\n=== Training ({epochs} epochs, {batch_size} batches/epoch) ===")

    for epoch in range(epochs):
        for b in range(0, sample_size, batch_size):
            batch_paths = [
                sample_paths[i] for i in range(b, min(b + batch_size, sample_size))
            ]
            n_actual = len(batch_paths)

            optimizer.zero_grad()
            loss, pred_latents = training_step(
                dino_model, vae, mapper, batch_paths, device, latent_loss=True
            )

            if b == 0 and epoch == 0:
                # Decode for visualization only
                with torch.no_grad():
                    decoded = vae.decode(pred_latents).sample
                save_batch(decoded[:n_actual], "/tmp/train_poc_before.png")
                del decoded

            loss.backward()
            optimizer.step()

        print(f"  Epoch {epoch + 1}/{epochs} done")

    # After training
    print("\n=== Compare ===")
    batch_paths = [sample_paths[i] for i in range(0, batch_size)]
    with torch.no_grad():
        _, final = training_step(
            dino_model, vae, mapper, batch_paths, device, latent_loss=True
        )
        # Decode for visualization
        decoded = vae.decode(final).sample
    save_batch(decoded, "/tmp/train_poc_after.png")

    print("\n" + "=" * 60)
    print("Mini-batch Training PoC Complete ✅")
    print("=" * 60)
    print("  Before: /tmp/train_poc_before.png")
    print("  After:  /tmp/train_poc_after.png")


def save_batch(images: torch.Tensor, path: str) -> None:
    from torchvision.utils import save_image

    save_image(images, path, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    main()
