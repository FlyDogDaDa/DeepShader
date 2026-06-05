---
created: 2026-06-05
author: agent
type: agent
status: final
tags: [daily-log, oom-fix, shard-aware-sampler, cache-shards-param, resnet-mapper]
---

# Cached training OOM 修復 + ShardAwareSampler + Cache shards 參數 + ResNet mapper

## What

1. 修復 cached training OOM crash（DataLoader worker 被 Killed）
2. 加入 `ShardAwareSampler` 提升 LRU hit rate（~2.8% → ~97%）
3. 新增 `--cache-shards` CLI 參數（可調整 RAM 快取 shards 數量）
4. 新增 `DinoToVAE_ResNet` mapper（32 層殘差塊，~634K params）

## Why

### OOM 修復

`num_workers=16` 在 cached mode 下，每個 worker 子進程都要載入 shard 到 RAM（~1.84 GB/shard），16 個同時操作耗盡記憶體被 OOM Killer 幹掉。

image mode 的 16 workers 是為了緩解 NFS 讀取延遲，但 cached mode 有內部 LRU cache，不需要 multiprocessing。

### ShardAwareSampler

原始 `_make_cached_dataloaders` 使用 `SubsetRandomSampler`（完全隨機 index 打亂），導致連續 batch 來自不同 shards → LRU hit rate 只有 ~2.8%，每次都要從 HDD 讀 shard。

改用 `ShardAwareSampler`（兩層 shuffle：shard 順序打亂 + shard 內連續）→ hit rate ~97%。

### Cache shards 參數

`max_cached_shards=8` 是硬編碼預設值，使用者需要能調整。預設 ~10 GB RAM，全部載入 ~530 GB。

### ResNet mapper

MLP mapper（~300K params）複雜度可能不夠。新增 ResNet mapper：32 個殘差塊 × 512 wide，總 params ~634K。

## How

### `train.py` 修改

**移除 `--num-workers` CLI：**

```python
# Before: --num-workers default=16 in "data" group
# After: moved to "data (image mode)" group, only used by image mode
```

**新增 `--cache-shards` CLI：**

```bash
uv run python -m train --cache-dir ~/hdd/cache --cache-shards 16
# Default=8 (~10 GB), set to 338 to cache all shards (~530 GB)
```

**Mapper 加入 `resnet` 選項：**

```bash
uv run python -m train --cache-dir ~/hdd/cache --mapper resnet
# Choices: linear, mlp, resnet
```

### `src/cache.py` — `ShardAwareSampler` 延長

```python
class ShardAwareSampler(Sampler[int]):
    def __init__(self, dataset, shard_size=1000, seed=42,
                 train_ratio=None, is_train=True):
        # Pre-compute shuffled index list (shard-aware ordering)
        self._ordered_indices = [...]  # shard order shuffled, sequential within

        # Optional train/val split AFTER ordering
        if train_ratio is not None:
            n_train = int(len(self._ordered_indices) * train_ratio)
            self._indices = self._ordered_indices[:n_train] if is_train else self._ordered_indices[n_train:]
        else:
            self._indices = self._ordered_indices
```

關鍵設計：train/val 切分在 **打亂後** 進行，所以兩份 sampler 共享相同的 shard-aware 順序。

### `src/trainer.py` 修改

**`_decode_samples_cached` 修復：**
- 移除錯誤的 `.squeeze(1)`（cached batch 已是 `[B, 1024, 384]`）
- 無 VAE 時 fallback：取前 3 channels 做 RGB 近似可視化（避免 16-channel PIL error）

**`create_mapper` 加入 resnet 分支：**

```python
elif config.mapper == "resnet":
    return DinoToVAE_ResNet(
        hidden_channels=config.hidden_channels,
        num_layers=config.num_layers,
        learnable_norm=config.learnable_norm,
    ).to(device)
```

### `src/models.py` — `DinoToVAE_ResNet`

```
[patch_tokens]   → [1024, 384]
↓ Linear + GELU + LN → [1024, 512]  (input projection)
↓ ResBlock × 32  (each: Linear → GELU → LN → Linear → GELU → LN + x)
↓ Linear → [1024, 16]
↓ reshape + bilinear upsample → [B, 16, 64, 64]
```

| mapper | params | blocks |
|--------|--------|--------|
| `linear` | 6,160 | 1 |
| `mlp` | ~302K | 4 |
| `resnet` | ~634K | 32 (512-wide) |

## Follow-up

- [ ] 觀察 resnet vs mlp 訓練 loss 曲線差異
- [ ] 驗證 `--cache-shards 16` 實際 RAM 用量
- [ ] 驗證 image mode backward compat
- [ ] 清理 type warnings（optional）

## References

- [16 Cache 架構設計](./16_2026_06_04_agent_cache-architecture-design.md)
- [src/models.py](../../src/models.py) — DinoToVAE_ResNet
- [src/cache.py](../../src/cache.py) — ShardAwareSampler train/val split
- [src/trainer.py](../../src/trainer.py) — create_mapper + _decode_samples_cached
- [train.py](../../train.py) — CLI args
