---
created: 2026-06-04
author: agent
type: agent
status: final
tags: [daily-log, trainer-reroute, train-py-two-modes, dual-mode-training]
---

# `trainer.py` 雙模式重構 + `train.py` CLI 改版

## What

完成訓練管線雙模式重構：`trainer.py` 新增 cached epoch helper，`train.py` 支援 `--cache-dir` 參數進入 cached mode。

## Why

舊版 `trainer.py` 綁死 DINOv3 + VAE — 每個 batch 都要即時編碼。
`train.py` 只能跑 image mode。

重構後：
- `dino=None, vae=None` → cached mode（只訓練 mapper，不用 DINO/VAE）
- `dino=vae` → image mode（舊行為，backward compatible）

## How

### `src/trainer.py` 修改

新增函數（不改變既有 API）：

- `train_epoch_cached()` — 吃 `(patch_tokens, gt_latents)` batch，不需要 DINO/VAE
- `validate_epoch_cached()` — 同上，validation
- `_decode_samples_cached()` — 從 cached loader 解碼 sample 圖片
- `run_training()` — 加入 `use_cached = dino is None and vae is None` 自動偵測
  - 偵測到 cached mode 自動切換函數，不需要改 call site

舊的 `_decode_samples()` / `train_epoch()` / `validate_epoch()` 完全保留，image mode 呼叫不受影響。

### `train.py` 修改

CLI 新增 `--cache-dir` + `--no-cache`：

| 用法 | mode | 會載入 DINO/VAE |
|------|------|----------------|
| `python -m train` | image（舊） | ✅ |
| `python -m train --cache-dir ~/hdd/cache` | cached | ❌ |
| `python -m train --no-cache` | image（強制） | ✅ |

Cached mode DataLoaders 建立流程：
1. `CachedDataset(config)` — 讀 manifest + 建 index map
2. `SubsetRandomSampler(train_indices)` — 打亂訓練索引
3. `SequentialSampler(val_indices)` — 驗證不 shuffle
4. `_collate_fn` — `(tokens, latents) tuples → stacked batches`

### 檔案修改一覽

| 檔案 | 變更 |
|------|------|
| `src/trainer.py` | +120 lines：cached epoch helpers + `run_training()` 自動偵測 |
| `src/__init__.py` | +4 exports：`train_epoch_cached`, `validate_epoch_cached` |
| `train.py` | rewrite：mode selection + cached dataloader factory |
| `src/encode.py` | pre-encode pipeline（前一階段完成） |

## Status

### 已完成 ✅
- `src/cache.py` — 完整實作
- `src/encode.py` — pre-encode CLI
- `src/trainer.py` — 雙模式自動偵測
- `train.py` — 雙模式 CLI
- `src/__init__.py` — API exports
- `tests/test_cache.py` — 22 tests

### Bugfix 🔧
- `train.py` L259 — `_make_cached_dataloaders_v2` → `_make_cached_dataloaders`（函數不存在，cached mode 會拋 `NameError`）
- 補上遺漏的 `seed=42` 參數

### 待處理 ⏳
- `trainer.py` / `train.py` 剩餘 type warnings（`DataLoader` 泛型、`Any` 型別） — 不影響執行，可後續清理
- 實測 cached training loop 能否正常跑完

## Follow-up

- [ ] 清理 type warnings（可选，不影響功能）
- [ ] 執行 `python -m train --cache-dir ~/hdd/cache --debug` 驗證 cached mode
- [ ] 執行 `python -m train --debug` 驗證 image mode（backward compat）
- [ ] 跑 pre-encode（如果有 dataset 和 cache 空間）

## References

- [16 Cache 架構設計](./16_2026_06_04_agent_cache-architecture-design.md)
- [17 Cache Pipeline 實作](./17_2026_06_04_agent_cache-pipeline-implementation.md)
- [src/cache.py](../../src/cache.py)
- [src/trainer.py](../../src/trainer.py)
- [src/encode.py](../../src/encode.py)
- [train.py](../../train.py)
