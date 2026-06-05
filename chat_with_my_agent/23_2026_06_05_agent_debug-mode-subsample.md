---
created: 2026-06-05
author: agent
type: agent
status: final
tags: [daily-log, debug-mode, subsample, cached-dataset, shard-aware-sampler]
---

# `train.py` debug mode 優化：只讀 4 個樣本，不掃完整資料集

## What

修改 `train.py` 的 `--debug` 行為，使其只使用極少量樣本（4 個）即可運行，不再遍歷完整資料集。

## Why

舊版 debug 模式有兩個問題：

1. **image mode**：仍會掃描全部 337K 張圖片（`scan_dataset`），只是 subsample 到 40 張
2. **cached mode**：`ShardAwareSampler` 會預先建出完整的 337K index list，`CachedDataset._build_index_map()` 會讀取全部 338 個 shard 的 `meta.json`

對於 debug 測試來說，這些開銷完全沒必要。debug 應該只用 1-2 個樣本快速驗證即可。

## How

### `src/cache.py`：`CachedDataset` 新增 `subset_indices` 參數

```python
def __init__(
    self,
    config: ShardCacheConfig,
    subset_indices: set[int] | None = None,
):
```

- 當 `subset_indices` 為 `None` 時，行為與之前完全相同（backward compatible）
- 當提供 `subset_indices` 時，`_build_index_map()` 只會建立對應的索引 entry，並跳過不含目標索引的 shard metadata 讀取
- 338 個 shard 的 metadata 讀取 → 最多 ~1 次（索引 0~3 都在 shard 0）

### `train.py`：cached mode debug 路徑重構

```python
if args.debug:
    debug_dataset = CachedDataset(
        config_, subset_indices=set(debug_indices)
    )
    train_loader = DataLoader(
        debug_dataset,
        batch_size=config.batch_size,
        sampler=RandomSampler(debug_dataset, replacement=False),
        ...
    )
```

- 建立 `CachedDataset` 時傳入 `subset_indices={0,1,2,3}`，只建 4 個 index entry
- 用 `RandomSampler` 替代 `ShardAwareSampler`，避免 337K index list
- `val_loader` 同樣使用 debug dataset + `SequentialSampler`

### `train.py`：image mode debug subsample 調整

- 從 40 張 → 4 張 (`all_paths[:4]`)
- 維持掃描 dataset 的邏輯（`scan_dataset` 很快，主要是目錄遍歷）

### 測試驗證

```bash
uv run pytest tests/test_cache.py -x -v   # 25 passed
uv run pytest tests/ -x -q                 # 74 passed
```

## Follow-up

- [ ] 實際執行 `python -m train --cache-dir ~/hdd/cache --debug` 驗證 debug mode 能否正常跑完
- [ ] 確認 debug dataset 的 `len()` 確實回傳 4（`_index_map` 只有 4 個 entry）
- [ ] 觀察 debug mode 的訓練速度是否如預期顯著提升

## References

- [src/cache.py](../../src/cache.py) — `CachedDataset.__init__`, `_build_index_map`
- [train.py](../../train.py) — `main()`, cached mode debug branch
- [22_2026_06_05_agent_validation-removed-and-decode-py](./22_2026_06_05_agent_validation-removed-and-decode-py.md) — 上一筆記錄
- [18_2026_06_04_agent_trainer-and-train-py-double-mode](./18_2026_06_04_agent_trainer-and-train-py-double-mode.md) — cached mode 設計
