"""Quick VAE test: encode an image and decode it back."""

import torch
from PIL import Image
from torchvision.utils import save_image

print("=== Step 1: Load VAE ===")
from diffusers import AutoencoderKL

vae = AutoencoderKL.from_pretrained("advokat/AnimePro-FLUX", subfolder="vae").to("cuda")
print(
    f"  VAE loaded: latent_channels={vae.config.latent_channels}, scaling={vae.config.scaling_factor}"
)

print("\n=== Step 2: Encode test image ===")
# Try dataset image first
from pathlib import Path

dataset_root = Path(
    "/mnt/nas/PC_Data/NAS PC_data/2025_02_10_old_hdd/B11223209_Vincent/Tagged_Anime_Illustration/danbooru-images/danbooru-images"
)
dataset_root = Path("/mnt/nas/PC_Data/NAS PC_data/2025_02_10_old_hdd/B11223209_Vincent/Tagged_Anime_Illustration/danbooru-images/danbooru-images")
img_paths = list(dataset_root.rglob("*.jpg"))[:1]

if img_paths:
    img = Image.open(img_paths[0]).convert("RGB").resize((512, 512))
    print(f"  Using: {img_paths[0]}")
else:
    img = Image.new("RGB", (512, 512), color=(255, 0, 128))
    img.save("/tmp/test_vae.jpg")
    print("  Using random test image")

from torchvision.transforms import functional as T

inputs = T.to_tensor(img).unsqueeze(0).to("cuda") * 2 - 1  # Normalize to [-1, 1]
print(f"  Input tensor: {inputs.shape}")

with torch.no_grad():
    latents = vae.encode(inputs).latent_dist.sample()
    # Apply scaling
    latents = (latents - vae.config.shift_factor) / vae.config.scaling_factor
    print(f"  Encoded latents: {latents.shape} (expected: [1, 16, 64, 64])")

print("\n=== Step 3: Decode back ===")
with torch.no_grad():
    decoded = vae.decode(latents).sample
    print(f"  Decoded image: {decoded.shape}")
    print(f"  Pixel range: [{decoded.min():.3f}, {decoded.max():.3f}]")

save_image(decoded, "/tmp/vae_test_decoded.jpg", normalize=True, value_range=(-1, 1))
print("  Saved to /tmp/vae_test_decoded.jpg")

print("\n=== VAE PoC Complete ===")
