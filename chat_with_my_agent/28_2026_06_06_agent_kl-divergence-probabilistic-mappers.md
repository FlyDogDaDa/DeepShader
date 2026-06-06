---
created: 2026-06-06
author: Vincent
type: agent
status: draft
tags: [kl-divergence, probabilistic-mapper, distribution-alignment, encode, cache]
---

# KL Divergence Loss: Probabilistic Mappers + VAE Posterior Cache

## What

Migrated the training pipeline from deterministic per-pixel MSE reconstruction to **distribution alignment** using KL divergence between the mapper posterior and the VAE posterior. Implemented across 5 files: mappers now output `(mean, logvar)`, cache now stores `gt_logvar.pt`, and the training loop uses `MSE + β * KL` loss.

## Why

MSE is a **coordinate-wise** distance that forces the mapper to predict exact latent values — it learns a conservative "average prediction" that ignores the structure of the latent distribution. The goal is **latent distribution alignment**, not coordinate matching. KL divergence between the mapper's posterior distribution and the VAE's posterior distribution is the most direct way to achieve this.

## How

### 1. `src/models.py` — All 5 mappers became probabilistic

Every mapper now outputs `(z, mean, logvar)` instead of a single tensor:

- **New**: `_reshape_and_upsample()` — shared helper for reshaping `[B, N, C]` → `[B, C, H, W]` with bilinear interpolation
- **New**: `kl_divergence(mean_m, logvar_m, mean_v, logvar_v)` — computes KL(Nm || Nv) = 0.5 * Σ[log(var_v/var_m) + (var_m + (μ_m - μ_v)²)/var_v - 1]
- **Every mapper** (`Linear`, `MLP`, `ResNet`, `Transformer`, `TransformerResNet`):
  - `latent_project = Linear(hidden, 16)` → split into `mean_project` + `logvar_project` (each `[hidden, 16]`)
  - `forward(tokens, sample=True)` → reparameterization: `z = mean + exp(0.5 * logvar) * randn`
  - `sample=False` → `z = mean` (for evaluation / sampling)
- Params count roughly doubles: each mapper gains one extra `Linear(hidden → 16)` layer (~385–513 params)

### 2. `src/pretrains.py` — VAE wrapper returns full posterior

Added `VAEModel.encode_distribution(images) → (mean, logvar)` — wraps `self._vae.encode(images).latent_dist` to expose both `mean` and `logvar`.

### 3. `src/cache.py` — Shard cache stores logvar with backward compat

- **ShardCache**: stores `(tokens, latents, logvars|None)` — `logvars` can be `None` for backward compat
- **CachedDataset.__getitem__**: returns `(tokens, gt_mean, gt_logvar)` triple
- If `gt_logvar.pt` doesn't exist on disk (old cache), returns `zeros` of shape `[B, 16, 64, 64]`
- Shard format now: `patch_tokens.pt` + `gt_latents.pt` + `gt_logvar.pt` + `meta.json`

### 4. `src/encode.py` — Re-encode writes `gt_logvar.pt`

- `BatchShardEncoder` buffers now carry `(tokens, means, logvars)`
- Flushes 3 files per shard (was 2)
- Uses `vae.encode_distribution()` instead of `.latent_dist.mode()`
- Fixed `del gt_latents` → `del gt_mean, gt_logvar` (stale variable name)
- Added DataLoader with `num_workers=16` and `pin_memory=True` (was single-threaded, bottlenecking GPU)
- Added `--batch-size` parameter (default: 32)
- Added `--subset start,end` parameter for partial re-encode (e.g. `--subset 0,32000`)
- Fixed `validate_cache()` call: was passing `Path` directly, now wraps in `ShardCacheConfig()`
- Fixed `--device` default from `"cuda"` to `"cuda:0"`

### 5. `src/trainer.py` — KL loss in training loop

- **train_epoch_cached**: batch unpacks `(patch_tokens, gt_mean, gt_logvar)`; loss = `MSE(pred_z, gt_mean) + β * KL(pred_mean, pred_logvar, gt_mean, gt_logvar)`
- **train_epoch** (image mode): also calls `vae.encode_distribution()` for gt logvar
- **validate_epoch_cached / validate_epoch**: same KL-based loss with `sample=False`
- **_save_samples_cached**: saves `(tokens, gt_mean, gt_logvar, pred_z)` instead of `(tokens, gt_latents, pred_latents)`
- `TrainingConfig` adds `beta: float = 1e-4`
- `run_training()` passes `config.beta` to all epoch functions
- Log format extended: `loss={:.4f} mse={:.4f} kl={:.6f}`

### 6. `train.py` — Collate + CLI updates

- `_collate_fn`: returns `(tokens, means, logvars)` — 3-tuple to match `CachedDataset`
- Debug collate function updated to 3-tuple
- Added `--beta` CLI argument (default: 1e-4, stored in `TrainingConfig`)
- Config instantiation includes `beta=args.beta`

### 7. Test fixes

- `tests/test_mapper.py`: all 9 tests updated to expect 3-tuple return and use `sample=False` / `sample=True`
- `tests/test_cache.py`: mock shards now create `gt_logvar.pt`; unpacking changed to 3-tuple
- **74/74 tests passing** ✅

## Follow-up

- Run a small subset experiment: `--beta 0` (MSE-only) vs `--beta 1e-4` (KL) with identical settings to validate KL helps
- Pre-encode 32,000 images with `--force --subset 0,32000` to get `gt_logvar.pt` in cache
- Monitor whether old cache shards (with logvar=zeros) cause issues in early epochs — the KL term will be near-zero until full cache is re-encoded
- Compare `--beta 1e-4` vs `--beta 0` convergence speed and final loss
- The old cache was deleted (`rm -rf ~/hdd/cache`), so re-encoding is needed for training to resume

## References

- [src/models.py](../../src/models.py) — Probabilistic mappers with `mean_project` + `logvar_project`
- [src/pretrains.py](../../src/pretrains.py) — `encode_distribution()` method
- [src/cache.py](../../src/cache.py) — Shard cache with `gt_logvar.pt` backward compat
- [src/encode.py](../../src/encode.py) — Encode pipeline with DataLoader + `--subset` + `--batch-size`
- [src/trainer.py](../../src/trainer.py) — KL loss in epoch functions + `beta` config
- [train.py](../../train.py) — `--beta` CLI + 3-tuple collate
- [tests/test_mapper.py](../../tests/test_mapper.py) — Updated mapper tests
- [tests/test_cache.py](../../tests/test_cache.py) — Updated cache tests
