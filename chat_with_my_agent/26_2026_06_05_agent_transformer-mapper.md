---
created: 2026-06-05
author: Vincent
type: draft
status: draft
tags: [model, transformer, self-attention, resnet-32-convergence]
---

# Introduce Transformer Mapper Architecture

## What

Added two Transformer-based mapper architectures to replace the per-token-only mapping in the current ResNet-32 model, addressing its convergence issues.

## Why

The ongoing training run (`runs/cached-resnet-32_20260605_132354`) showed poor convergence. The current ResNet-32 mapper processes each of the 1024 DINOv3 patch tokens independently — every token sees only its own 384-dim feature vector, with zero information about neighboring patches. For image-to-latent mapping where spatial context matters (e.g., a "sky" token needs to know about adjacent "building" tokens), this isolation severely limits expressiveness.

Self-attention solves this by letting every token aggregate information from all others, providing global context before the per-token projection.

## How

### New modules added to `src/models.py`:

- **`_TransformerBlock`** — Core Transformer encoder block: pre-norm LayerNorm → `nn.MultiheadAttention` (self-attention) → residual → post-norm LayerNorm → FFN (Linear → GELU → Linear) → residual. Standard Pre-LN Transformer architecture, batch-first operation on `[B, 1024, D]`.
- **`DinoToVAE_Transformer`** — Pure Transformer mapper: input → (optional) LayerNorm → N `_TransformerBlock` layers → (optional) LayerNorm → `Linear(D→16)` → reshape + bilinear upsample to `[B, 16, 64, 64]`. ~390K params with default 8 heads.
- **`DinoToVAE_TransformerResNet`** — Hybrid: Transformer encoder for global context → project to 512-dim → 32 `ResBlock` layers for per-token refinement → `Linear(512→16)` → reshape + upsample. ~2.8M params.

### Wiring in `src/trainer.py`:

- `TrainingConfig` extended with transformer fields: `num_heads` (8), `num_transformer_layers` (1), `mlp_ratio` (4.0), `num_resnet_layers` (32).
- `create_mapper()` now handles `"transformer"` and `"transformer_resnet"` mapper types, routing to the new classes with appropriate parameter passing.

### CLI in `train.py`:

- `--mapper` choices expanded: `"linear"` | `"mlp"` | `"resnet"` | `"transformer"` | `"transformer_resnet"`
- New flags: `--num-heads`, `--num-transformer-layers`, `--mlp-ratio`, `--num-resnet-layers`
- All passed through to `TrainingConfig` constructor.

### Active training run:

```
uv run python -m train --cache-dir ~/hdd/cache --mapper transformer \
    --hidden-channels 384 --num-transformer-layers 8 --output runs/transformer-8 \
    --batch-size 64 --cache-shards 4 --epochs 5 --warmup-steps 300 \
    --lr 0.003 --sample-indices 0,1,2,3,4
```

GPU utilization: ~147W, 81°C (approaching throttle threshold), VRAM ~11 GB / 12 GB. The 8-layer Transformer with batch 64 produces a large attention matrix (`64 × 1024 × 1024`), consuming significant memory.

## Follow-up

- Compare validation loss of `runs/transformer-8` against the stopped `runs/cached-resnet-32_20260605_132354` after 5 epochs
- Consider a `transformer_resnet` control group (1 layer attention + 32 ResNet blocks) to combine global context with per-token refinement capacity
- Monitor GPU temperature — 81°C is close to the 83-87°C throttle range; batch size 32 may be needed if it climbs further
- 8 attention layers on top of 1024-token sequence is aggressive — could be overkill; if results don't improve, try 2-4 layers

## References

- [src/models.py](../../src/models.py) — `_TransformerBlock`, `DinoToVAE_Transformer`, `DinoToVAE_TransformerResNet`
- [src/trainer.py](../../src/trainer.py) — `TrainingConfig`, `create_mapper()`
- [train.py](../../train.py) — CLI argument parsing and config construction
- [21_2026_06_05_agent_cached-oom-fix-and-resnet-mapper.md](./21_2026_06_05_agent_cached-oom-fix-and-resnet-mapper.md) — Previous ResNet mapper addition
- [25_2026_06_05_agent_decode-py-reformat.md](./25_2026_06_05_agent_decode-py-reformat.md) — Latest prior entry for today
