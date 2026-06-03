"""Full pipeline PoC: Image → DINOv3 → mapping → VAE decode → Image.

End-to-end test of the complete pipeline:
    Original image → DINOv3 ViT-S → patch tokens → DinoToVAE → latents → FLUX VAE decode → Output image

This is the ONLY trainable component in the full pipeline.
"""

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL
from dotenv import load_dotenv
from PIL import Image
from torchvision.utils import save_image
from transformers import AutoImageProcessor, AutoModel

from src.utility import scan_dataset

load_dotenv()

dataset_root = Path(os.environ["DATASET_ROOT"]) / "danbooru-images" / "danbooru-images"


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


print("=" * 60)
print("Full Pipeline PoC: DINOv3 → Mapping → VAE Decode")
print("=" * 60)

# ── Step 1: Load VAE ──────────────────────────────────────────────
print("\n=== Step 1: Load FLUX VAE ===")
vae = AutoencoderKL.from_pretrained("advokat/AnimePro-FLUX", subfolder="vae").to("cuda")
print(
    f"  ✅ latent_channels={vae.config.latent_channels}, scaling={vae.config.scaling_factor}"
)

# ── Step 2: Load DINOv3 ViT-S ─────────────────────────────────────
print("\n=== Step 2: Load DINOv3 ViT-S/16 ===")
processor = AutoImageProcessor.from_pretrained(
    "facebook/dinov3-vits16-pretrain-lvd1689m"
)
model = AutoModel.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m").to("cuda")
print(
    f"  ✅ hidden_dim={model.config.hidden_size}, patch_size={model.config.patch_size}"
)

# ── Step 3: Load test image ───────────────────────────────────────
print("\n=== Step 3: Load test image ===")
img_paths = scan_dataset(dataset_root)  # cached!

if img_paths:
    print(f"  ✅ scan_dataset found {len(img_paths)} images")
    img = Image.open(img_paths[0]).convert("RGB").resize((512, 512))
    print(f"  Using: {img_paths[0].name}")
else:
    img = Image.new("RGB", (512, 512), color=(128, 64, 192))
    img.save("/tmp/test_poc.jpg")
    print("  Using random test image")

# ── Step 4: DINOv3 feature extraction ─────────────────────────────
print("\n=== Step 4: DINOv3 feature extraction ===")
# Process 512x512 image manually (processor defaults to 224x224)
import torchvision.transforms.v2 as T2

proc = T2.Compose(
    [
        T2.ToImage(),
        T2.ToDtype(torch.float32, scale=True),  # [0, 1]
    ]
)
inputs_pil = proc(img).unsqueeze(0).to("cuda")  # [1, 3, 512, 512] in [0, 1]

with torch.no_grad():
    # Use processor with raw input (avoid automatic resize)
    # processor normalizes with ImageNet stats, we do it manually
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to("cuda")
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to("cuda")
    inputs_norm = (inputs_pil - mean) / std  # ImageNet normalization

    outputs = model(inputs_norm)
    hidden_state = outputs.last_hidden_state
    # Remove class token (0) + register tokens (1-4)
    patch_tokens = hidden_state[:, 5:, :]
print(f"  ✅ patch_tokens shape: {patch_tokens.shape}  (expected: [1, 1024, 384])")

# ── Step 5: Mapping network ───────────────────────────────────────
print("\n=== Step 5: DinoToVAE mapping ===")
mapper = DinoToVAE(hidden_dim=384, latent_dim=16).to("cuda")
print(f"  Parameters: {sum(p.numel() for p in mapper.parameters()):,}")
with torch.no_grad():
    latents = mapper(patch_tokens)
print(f"  ✅ latents shape: {latents.shape}  (expected: [1, 16, 64, 64])")

# ── Step 6: VAE decode ────────────────────────────────────────────
print("\n=== Step 6: FLUX VAE decode ===")
with torch.no_grad():
    decoded = vae.decode(latents).sample
print(f"  ✅ decoded shape: {decoded.shape}  (expected: [1, 3, 512, 512])")
print(f"     Pixel range: [{decoded.min():.4f}, {decoded.max():.4f}]")

# ── Step 7: Save results ──────────────────────────────────────────
print("\n=== Step 7: Save results ===")
save_image(decoded, "/tmp/poc_full_decoded.jpg", normalize=True, value_range=(-1, 1))
print(f"  Decoded → /tmp/poc_full_decoded.jpg")
img.save("/tmp/poc_full_original.jpg")
print(f"  Original → /tmp/poc_full_original.jpg")

print("\n" + "=" * 60)
print("Full Pipeline PoC Complete ✅")
print("=" * 60)
print("\nSummary:")
print(f"  DINOv3 ViT-S:     {patch_tokens.shape}")
print(f"  DinoToVAE params:   {sum(p.numel() for p in mapper.parameters()):,}")
print(f"  Latent output:      {latents.shape}")
print(f"  Decoded image:      {decoded.shape}")
print(f"  This is the ONLY trainable component!")
