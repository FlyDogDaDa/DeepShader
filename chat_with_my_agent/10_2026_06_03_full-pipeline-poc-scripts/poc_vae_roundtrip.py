"""PoC: VAE encode/decode test (end-to-end roundtrip).

Tests the VAE half of the pipeline. DINOv3 test is skipped until access is approved.
"""

from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as T
from torchvision.utils import save_image

print("=" * 60)
print("POC: VAE Roundtrip Test")
print("=" * 60)

# ── Step 1: Load VAE ──────────────────────────────────────────────
print("\n=== Step 1: Load FLUX VAE ===")
try:
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained("advokat/AnimePro-FLUX", subfolder="vae").to(
        "cuda"
    )
    print(f"  ✅ VAE loaded")
    print(f"     latent_channels = {vae.config.latent_channels}")
    print(f"     scaling_factor  = {vae.config.scaling_factor}")
    print(f"     shift_factor    = {vae.config.shift_factor}")
except Exception as e:
    print(f"  ❌ Failed to load VAE: {e}")
    exit(1)

# ── Step 2: Load test image ───────────────────────────────────────
print("\n=== Step 2: Load test image ===")
dataset_root = Path(
    "/mnt/nas/PC_Data/NAS PC_data/2025_02_10_old_hdd/B11223209_Vincent/Tagged_Anime_Illustration"
    "/danbooru-images/danbooru-images"
)

img_paths = sorted(dataset_root.rglob("*.jpg"))
if img_paths:
    img = Image.open(img_paths[0]).convert("RGB").resize((512, 512))
    print(f"  ✅ Using: {img_paths[0].name}")
else:
    img = Image.new("RGB", (512, 512), color=(128, 64, 192))
    img.save("/tmp/test_poc.jpg")
    print("  ⚠️  No dataset image found, using random test image")

# ── Step 3: Encode with VAE ───────────────────────────────────────
print("\n=== Step 3: Encode image → latent ===")
inputs = T.to_tensor(img).unsqueeze(0).to("cuda") * 2 - 1  # → [-1, 1]
print(f"  Input tensor shape: {inputs.shape}")

with torch.no_grad():
    encoded = vae.encode(inputs).latent_dist
    print(f"  Encoded distribution: mean={encoded.mean.shape}, std={encoded.std.shape}")

    # Sample + apply scaling (encode formula)
    latents = encoded.sample()
    latents = (latents - vae.config.shift_factor) / vae.config.scaling_factor
    print(f"  ✅ Encoded latent shape: {latents.shape}  (expected: [1, 16, 64, 64])")

    # ── Step 4: Decode back ─────────────────────────────────────────
    print("\n=== Step 4: Decode latent → image ===")
    decoded = vae.decode(latents).sample
    print(f"  ✅ Decoded image shape: {decoded.shape}")
    print(f"     Pixel range: [{decoded.min():.4f}, {decoded.max():.4f}]")

# ── Step 5: Save comparison ───────────────────────────────────────
print("\n=== Step 5: Save results ===")
save_image(
    decoded, "/tmp/poc_vae_roundtrip_decoded.jpg", normalize=True, value_range=(-1, 1)
)
save_image(decoded, "/tmp/poc_vae_roundtrip_decoded.jpg", normalize=True, value_range=(-1, 1))
print(f"  Decoded → /tmp/poc_vae_roundtrip_decoded.jpg")

# Also save the original for comparison
img.save("/tmp/poc_vae_roundtrip_original.jpg")
print(f"  Original → /tmp/poc_vae_roundtrip_original.jpg")

print("\n" + "=" * 60)
print("VAE PoC Complete ✅")
print("=" * 60)

# ── Step 6: DINOv3 (pending access) ───────────────────────────────
print("\n=== Step 6: DINOv3 Feature Extraction (SKIPPED — gated access pending) ===")
print("  ❌ Waiting for HuggingFace access approval")
print("  Required model: facebook/dinov3-vits16-pretrain-lvd1689m")
print("  Once approved, this will:")
print("    1. Load DINOv3 ViT-S/16")
print("    2. Extract patch tokens (512×512 input → 1024 tokens × 384 dim)")
print("    3. Skip class token and register tokens (first 5)")
print("    4. Output: [1, 1024, 384]")
print("  5. Pass to mapping network (to be designed)")
