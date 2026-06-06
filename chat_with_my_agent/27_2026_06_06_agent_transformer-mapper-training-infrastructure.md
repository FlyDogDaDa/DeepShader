---
created: 2026-06-06
author: Vincent
type: agent
status: draft
tags: [training, transformer, small-subset, decode, infrastructure]
---

# Transformer Training Infra: subset, no-val, save-freq, decode layout

## What

Extended training infrastructure to support small-subset experiments, flexible checkpointing, and vertical comparison of decoded images across epochs.

## Why

The ongoing `cached-resnet-32` training showed poor convergence — the per-token-only mappers lacked spatial context. Introduced Transformer-based mappers, but with the 320K sample dataset, even 5 epochs took long. Additionally, the 16-sample debug run failed with `ZeroDivisionError` because `--batch-size 32` couldn't form a batch from 14 training samples (90/10 split).

Also needed a cleaner way to compare decoded outputs across epochs: the flat `{epoch}_idx{N}.png` layout forced scrolling through 40+ files to compare the same sample.

## How

### 1. `src/models.py` — Already had Transformer modules from prior session

- `_TransformerBlock` — Pre-LN Transformer block (MHA + FFN + residual)
- `DinoToVAE_Transformer` — Pure Transformer mapper
- `DinoToVAE_TransformerResNet` — Transformer + ResNet hybrid

### 2. CLI additions (`train.py`)

- `--subset N` — Limit to first N samples from cache (None=all). Drastically speeds up small experiments.
- `--no-val` — Use all data for training (no train/val split). Fixes ZeroDivisionError when `batch_size > train_samples`.
- `--save-freq N` — Save checkpoint & samples every N epochs (default: 5). Reduces disk I/O for long runs.

### 3. `src/trainer.py` — Config + save logic

- `TrainingConfig` extended: `subset_size`, `no_val`, `save_every`
- `_make_cached_dataloaders()`:
  - Accepts `subset_size` and `no_val`
  - `no_val=True` → `train_ratio=1.0`, `drop_last=False`, returns `(train_loader, None, dataset)`
  - `subset_size=N` → `CachedDataset` built with `subset_indices=set(range(N))`, skips reading 338 shard metadata files
- `run_training()` loop: `save_epoch = (epoch+1) % save_every == 0 or epoch == epochs-1` controls both checkpoint and sample saves. Last epoch always saves.

### 4. `src/decode.py` — Vertical comparison output layout

Changed from flat `{epoch}_idx{N}.png` to hierarchical:

```
output/
├── idx_0/
│   ├── epoch_0000.png   ← pre-training
│   ├── epoch_0001.png   ← epoch 1
│   ├── epoch_0005.png   ← epoch 5
│   └── epoch_0010.png   ← epoch 10
├── idx_1/
│   ├── epoch_0000.png
│   └── ...
```

Each `idx_N/` directory groups all epoch versions of the same sample for easy vertical comparison. Uses `mkdir(parents=True, exist_ok=True)` for each index directory.

### Active training commands

Transformer-4 on subset (fast experiment):
```bash
uv run python -m train --cache-dir ~/hdd/cache --mapper transformer \
    --hidden-channels 384 --num-transformer-layers 4 --output runs/transformer-4 \
    --batch-size 32 --cache-shards 32 --subset 32000 --epochs 50 \
    --warmup-steps 300 --lr 0.003 --sample-indices 0,1,2,3,4 --no-val
```

Decode after training:
```bash
uv run python -m src.decode --samples runs/transformer-4/samples \
    --output-dir ~/hdd/decode-results/transformer-4
```

## Follow-up

- Monitor training loss of `runs/transformer-4` vs previous `runs/cached-resnet-32_20260605_132354`
- GPU VRAM: 32000 samples + batch=32 + 4-layer Transformer + attention[32×1024×1024] → check if OOM
- If `--save-freq=5` produces too many checkpoints on large subset, increase to 10
- Consider comparing `transformer_resnet` (attention + deep ResNet refinement) vs pure Transformer

## References

- [src/models.py](../../src/models.py) — `_TransformerBlock`, `DinoToVAE_Transformer`, `DinoToVAE_TransformerResNet`
- [src/trainer.py](../../src/trainer.py) — `TrainingConfig`, `_make_cached_dataloaders()`, `run_training()` save logic
- [train.py](../../train.py) — CLI: `--subset`, `--no-val`, `--save-freq`
- [src/decode.py](../../src/decode.py) — Vertical comparison output layout
- [26_2026_06_05_agent_transformer-mapper.md](../chat_with_my_agent/26_2026_06_05_agent_transformer-mapper.md) — Prior entry: Transformer mapper introduction
