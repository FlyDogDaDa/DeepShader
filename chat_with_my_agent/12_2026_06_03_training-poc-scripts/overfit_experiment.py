"""Overfit experiment: can the DinoToVAE pipeline learn at all?

Two mapper variants compared:
  1. Linear (baseline)      — nn.Linear(384 → 16), 6,160 params
  2. MLP (deeper, wider)    — 3 hidden layers, ~500K params

Strategy: overfit on a SINGLE image, 500 steps.
If the pipeline works, decoded output should look like the input.
If still noise → fundamental issue (feature-space mismatch, etc.)

Also logs latent statistics to diagnose where the signal is lost.
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

# ── Mapper variants ───────────────────────────────────────────────


class DinoToVAE_Linear(nn.Module):
    """Baseline: single linear layer."""

    def __init__(
        self,
        hidden_dim: int = 384,
        latent_dim: int = 16,
        patch_size: int = 16,
        image_size: int = 512,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        patch_grid = image_size // patch_size
        latent_size = patch_grid * 2
        self.project = nn.Linear(hidden_dim, latent_dim)
        self.patch_grid = patch_grid
        self.latent_size = latent_size

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        out = self.project(patch_tokens)
        b = out.shape[0]
        out = out.permute(0, 2, 1).reshape(
            b, self.latent_dim, self.patch_grid, self.patch_grid
        )
        return F.interpolate(
            out, size=self.latent_size, mode="bilinear", align_corners=False
        )


class DinoToVAE_MLP(nn.Module):
    """Deeper mapper: non-linear MLP on patch tokens → latent channels.

    Architecture:
        [patch_tokens]         → [1024, 384]  (local spatial)
        ↓ Linear → MLP blocks → [1024, H]
        ↓ Linear               → [1024, L]
        ↓ reshape + upsample   → [1, L, 64, 64]  VAE latent

    NOTE: Input does NOT contain class token — DINOv3's [5:, :] is passed in.
    """

    def __init__(
        self,
        hidden_dim: int = 384,
        latent_dim: int = 16,
        patch_size: int = 16,
        image_size: int = 512,
        hidden_channels: int = 256,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        patch_grid = image_size // patch_size
        latent_size = patch_grid * 2

        # Non-linear MLP on each patch
        layers = []
        for i in range(num_layers):
            in_ch = hidden_dim if i == 0 else hidden_channels
            layers.extend(
                [
                    nn.Linear(in_ch, hidden_channels),
                    nn.GELU(),
                    nn.LayerNorm(hidden_channels),
                ]
            )
        self.mlp = nn.Sequential(*layers)

        # Project to latent channels
        self.latent_project = nn.Linear(hidden_channels, latent_dim)
        self.patch_grid = patch_grid
        self.latent_size = latent_size

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        patch_tokens: [B, 1024, 384]  (patch tokens only, no class token)
        """
        out = self.mlp(patch_tokens)  # [B, 1024, H]
        out = self.latent_project(out)  # [B, 1024, L]
        b = out.shape[0]
        # [B, 1024, L] → [B, L, 32, 32] → upsample → [B, L, 64, 64]
        out = out.permute(0, 2, 1).reshape(
            b, self.latent_dim, self.patch_grid, self.patch_grid
        )
        latents = F.interpolate(
            out, size=self.latent_size, mode="bilinear", align_corners=False
        )
        return latents


# ── Debug helpers ─────────────────────────────────────────────────


def log_latents(name: str, latents: torch.Tensor):
    print(
        f"  {name:20s}  mean={latents.mean():+8.4f}  std={latents.std():8.4f}  "
        f"min={latents.min():+8.4f}  max={latents.max():+8.4f}"
    )


def save_image_batch(images: torch.Tensor, path: str) -> None:
    """Save tensor [-1, 1] as image."""
    from torchvision.utils import save_image

    save_image(images, path, normalize=True, value_range=(-1, 1))


def save_raw_image(tensor: torch.Tensor, path: str) -> None:
    """Save tensor [0, 1] as image."""
    img = tensor.squeeze(0).clamp(0, 1).cpu()
    from torchvision.transforms import ToPILImage

    ToPILImage()(img).save(path)


# ── Training step (same as train_poc.py, with debug) ──────────────


def training_step(dino_model, vae, mapper, image_path, device, verbose: bool = False):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    # Load single image [1, 3, 512, 512] in [0, 1]
    image = (
        v2.ToImage()(Image.open(image_path).convert("RGB").resize((512, 512)))
        .unsqueeze(0)
        .to(device)
    )

    # ── Stage 1: DINOv3 (no_grad) ──────────────────────────────
    image_norm = (image - mean) / std
    with torch.no_grad():
        dino_out = dino_model(image_norm)
        patch_tokens = dino_out.last_hidden_state[:, 5:, :]  # skip class + 4 pos embeds
    del image_norm

    # ── Stage 2: Mapper → pred_latents ─────────────────────────
    pred_latents = mapper(patch_tokens)

    if verbose:
        log_latents("pred_latents", pred_latents)

    # ── Stage 3: VAE encode (no_grad) → gt_latents ─────────────
    with torch.no_grad():
        gt_latents = vae.encode(image.float() * 2 - 1).latent_dist.mode()
    del image

    if verbose:
        log_latents("gt_latents", gt_latents)

    # Loss
    loss = F.mse_loss(pred_latents, gt_latents)
    return loss, pred_latents, gt_latents


# ── Main ───────────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("Overfit Experiment: DinoToVAE")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load models
    print("\n[Loading models...]")
    vae = AutoencoderKL.from_pretrained("advokat/AnimePro-FLUX", subfolder="vae").to(
        device
    )
    vae.eval()

    dino_model = AutoModel.from_pretrained(
        "facebook/dinov3-vits16-pretrain-lvd1689m"
    ).to(device)
    dino_model.eval()

    # Pick one image from dataset for overfitting
    dataset_root = (
        Path(os.environ["DATASET_ROOT"]) / "danbooru-images" / "danbooru-images"
    )
    all_paths_jpg = scan_dataset(dataset_root, pattern="**/*.jpg")
    all_paths_png = scan_dataset(dataset_root, pattern="**/*.png")
    all_paths = all_paths_jpg + all_paths_png

    test_image = all_paths[0]
    print(f"  Testing on: {test_image.name}")

    # ── Phase 1: Baseline (Linear, before training) ─────────────
    print("\n" + "=" * 60)
    print("Phase 1: Linear mapper — BEFORE training")
    print("=" * 60)

    mapper_linear = DinoToVAE_Linear(hidden_dim=384, latent_dim=16).to(device)
    _, pred_latents, _ = training_step(
        dino_model, vae, mapper_linear, test_image, device
    )
    with torch.no_grad():
        decoded = vae.decode(pred_latents).sample
    log_latents("decoded", decoded)
    save_image_batch(decoded, "/tmp/overfit_linear_before.png")
    save_raw_image((decoded[0] + 1) / 2, "/tmp/overfit_linear_before_raw.png")

    # ── Phase 2: Overfit Linear ─────────────────────────────────
    print("\n" + "=" * 60)
    print("Phase 2: Overfit Linear (500 steps, lr=1e-3)")
    print("=" * 60)

    mapper_linear = DinoToVAE_Linear(hidden_dim=384, latent_dim=16).to(device)
    optimizer = torch.optim.AdamW(mapper_linear.parameters(), lr=1e-3)

    for step in range(500):
        optimizer.zero_grad()
        loss, pred_latents, gt_latents = training_step(
            dino_model,
            vae,
            mapper_linear,
            test_image,
            device,
            verbose=(step % 100 == 0),
        )
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print(
                f"  step {step:4d}  loss={loss.item():.6f}  pred_std={pred_latents.std().item():.4f}  gt_std={gt_latents.std().item():.4f}"
            )

    with torch.no_grad():
        decoded = vae.decode(pred_latents).sample
    log_latents("decoded", decoded)
    save_image_batch(decoded, "/tmp/overfit_linear_after.png")
    save_raw_image((decoded[0] + 1) / 2, "/tmp/overfit_linear_after_raw.png")

    # ── Phase 3: Overfit MLP ────────────────────────────────────
    print("\n" + "=" * 60)
    print("Phase 3: Overfit MLP (500 steps, lr=1e-3)")
    print("=" * 60)

    mapper_mlp = DinoToVAE_MLP(hidden_dim=384, latent_dim=16).to(device)
    mlp_params = sum(p.numel() for p in mapper_mlp.parameters())
    print(f"  MLP params: {mlp_params:,}")
    optimizer = torch.optim.AdamW(mapper_mlp.parameters(), lr=1e-3)

    for step in range(500):
        optimizer.zero_grad()
        loss, pred_latents, gt_latents = training_step(
            dino_model, vae, mapper_mlp, test_image, device, verbose=(step % 100 == 0)
        )
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print(
                f"  step {step:4d}  loss={loss.item():.6f}  pred_std={pred_latents.std().item():.4f}  gt_std={gt_latents.std().item():.4f}"
            )

    with torch.no_grad():
        decoded = vae.decode(pred_latents).sample
    log_latents("decoded", decoded)
    save_image_batch(decoded, "/tmp/overfit_mlp_after.png")
    save_raw_image((decoded[0] + 1) / 2, "/tmp/overfit_mlp_after_raw.png")

    # ── Reference image ─────────────────────────────────────────
    ref_image = v2.ToImage()(Image.open(test_image).convert("RGB"))
    save_raw_image(ref_image[0], "/tmp/overfit_ref_raw.png")

    print("\nResults:")
    print("  Reference:    /tmp/overfit_ref_raw.png")
    print("  Linear before:/tmp/overfit_linear_before_raw.png")
    print("  Linear after: /tmp/overfit_linear_after_raw.png")
    print("  MLP after:    /tmp/overfit_mlp_after_raw.png")
    print("\n" + "=" * 60)
    print("Experiment complete ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()
