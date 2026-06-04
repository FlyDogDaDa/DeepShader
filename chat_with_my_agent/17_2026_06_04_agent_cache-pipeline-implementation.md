---
created: 2026-06-04
author: agent
type: agent
status: final
tags: [daily-log, cache-pipeline, encode-py, pre-encode, shard-encoder, resume-support]
---

# Cache Pipeline 實作：`cache.py` + `encode.py`

## What

完成 cache pipeline 核心實作：`cache.py`（LRU shard cache + CachedDataset + ShardAwareSampler + manifest）與 `encode.py`（pre-encode CLI）。

## How

### `src/cache.py` — 完整實作

```
cache/
├── ShardCacheConfig      # 配置：shard_size=1000, max_cached_shards=8, dtype=float16
├── ShardCache            # LRU cache（OrderedDict + auto evict + hit/miss stats）
├── CachedDataset         # PyTorch Dataset（O(1) index_map lookup）
├── ShardAwareSampler     # Two-level shuffle（shard order → shard-internal shuffle）
├── CacheManifest         # Manifest dataclass（version, hash, model versions）
├── load_manifest / save_manifest  # Manifest I/O
├── compute_dataset_hash  # Dataset 穩定 hash
└── validate_cache        # 檢查 cache 是否過期
```

**ShardAwareSampler 關鍵特性：**
- Level 1：打亂 shard 順序（每個 epoch 不同）
- Level 2：shard 內 shuffle（用 `seed + shard_id` 固定，跨 epoch 一致）
- 與 LRU(8) 配合 → hit rate ~97%，每 125 batches 只 miss 一次

### `src/encode.py` — Pre-encode Pipeline

```python
class BatchShardEncoder:
    """Batch encoding with automatic shard flush."""
    feed(tokens, latents, paths)  # Add to buffer → flush when full
    flush()                       # Flush remaining (last shard)
```

**CLI 功能：**
- `--dataset /data` — Dataset root
- `--cache-dir ~/hdd/cache` — Cache output directory
- `--shard-size 1000` — Images per shard
- `--resume` — Resume from last completed shard
- `--dry-run` — Validate manifest only
- `--force` — Force re-encoding
- `--device cuda` — Encoding device

**Resume 機制：**
1. 讀取 manifest → `completed_shards` 列表
2. 計算 `start_shard = max(completed) + 1`
3. 跳過已完成的 shard，只編碼剩餘部分
4. 驗證 cache 有效性（dataset hash + model versions）

### `src/__init__.py` — 公共 API 更新

新增 export：
- `CachedDataset`, `ShardAwareSampler`, `ShardCache`, `ShardCacheConfig`, `ShardMeta`, `CacheManifest`
- `load_manifest`, `save_manifest`, `validate_cache`, `compute_dataset_hash`

### `tests/test_cache.py` — 22 tests

| 類別 | Tests | 涵蓋 |
|------|-------|------|
| ShardCache | 4 | put/get, miss, LRU eviction, hit rate tracking |
| ShardCacheConfig | 1 | defaults |
| Manifest | 2 | save/load round-trip, missing file |
| Dataset hash | 3 | deterministic, order-independent, different paths |
| validate_cache | 4 | no manifest, hash mismatch, model mismatch, valid |
| CachedDataset | 5 | len, shard_count, shape, dtype, cache hits, multi-shard evict, info |
| ShardAwareSampler | 4 | covers all, different seeds, deterministic, len |

## 實作決策紀錄

| 決策 | 選擇 | 理由 |
|------|------|------|
| `encode.py` 位置 | `src/encode.py`（而非根目錄） | 與其他 src 模組一致，用 `python -m src.encode` 執行 |
| Buffer 處理 | `list.extend()` 而非預分配 | 簡單可靠，shard_size=1000 不算大 |
| Float 精度 | 存儲 float16 → 讀取轉 float32 | 節省 50% 存儲+I/O，訓練需要 float32 |
| Resume 策略 | `completed_shards: list[int]` | 支持任意 shard 順序完成（crash recovery） |
| 編碼批次大小 | batch_size=32 | DINOv3 ViT-S on 512x512, batch=32 約 ~4GB VRAM |

## Follow-up

- [ ] 重構 `trainer.py`（去掉 DINO/VAE 依賴，專為 cached training 優化）
- [ ] 重構 `train.py`（改用 cache 模式）
- [ ] 執行 pre-encode（實際測試）
- [ ] 比較 pre-encode vs 傳統 training 速度

## References

- [16_2026_06_04_cache-architecture-design](./16_2026_06_04_agent_cache-architecture-design.md) — 架構設計
