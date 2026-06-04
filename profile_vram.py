#!/usr/bin/env python3
"""VRAM profiling: before vs after VAE .no_grad() fix."""

import gc

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from transformers import AutoModel

gc.collect()
torch.cuda.empty_cache()
dino = (
    AutoModel.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m").cuda().eval()
)
vae = (
    AutoencoderKL.from_pretrained("advokat/AnimePro-FLUX", subfolder="vae")
    .cuda()
    .eval()
)
print(f"Models loaded: {torch.cuda.memory_allocated() / 1024**2:.0f} MB\n")

mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()

for batch_size in [2, 4, 8, 16]:
    print(f"\n=== batch={batch_size} ===")
    images = torch.randn(batch_size, 3, 512, 512, dtype=torch.float32).cuda()

    # DINO extract (no_grad already)
    normed = (images - mean) / std
    with torch.no_grad():
        patch_tokens = dino(normed).last_hidden_state[:, 5:, :]
    print(f"  After DINO: {torch.cuda.memory_allocated() / 1024**2:.0f} MB")

    # Mapper forward (small, ~300K params)
    import torch.nn as nn

    mapper = nn.Linear(384, 16).cuda()
    pred = mapper(patch_tokens)
    pred = pred.permute(0, 2, 1).reshape(batch_size, 16, 32, 32)
    pred = F.interpolate(pred, size=64, mode="bilinear")
    print(f"  After mapper: {torch.cuda.memory_allocated() / 1024**2:.0f} MB")

    # VAE encode WITH no_grad (NEW)
    with torch.no_grad():
        gt = vae.encode(images * 2 - 1).latent_dist.mode()
    print(
        f"  After VAE encode (no_grad): {torch.cuda.memory_allocated() / 1024**2:.0f} MB"
    )

    # Loss + backward
    loss = F.mse_loss(pred, gt)
    loss.backward()
    print(f"  After backward: {torch.cuda.memory_allocated() / 1024**2:.0f} MB")

    free = torch.cuda.mem_get_info()[0]
    print(f"  Free VRAM: {free / 1024**3:.2f} GB")

    del images, patch_tokens, pred, gt, loss, normed, mapper
