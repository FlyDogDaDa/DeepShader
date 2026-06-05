---
created: 2026-06-05
author: agent
type: agent
status: final
tags: [daily-log, validation-removed, sample-indices, decode-py, latent-storage]
---

# 移除 Validation + 固定採樣 + 獨立 Decode 工具

## What

1. **移除 Validation loss** — 從 `TrainingConfig` 和 `run_training` 中刪除 validation loop
2. **固定採樣索引** — 新增 `--sample-indices` CLI 參數，每次 epoch 保存固定索引的 `.pt` 檔
3. **新建 `src/decode.py`** — 獨立 VAE 解碼工具，可 CPU 解碼
4. **樣本格式從 epoch-based 改為 index-based**

## Why

### 移除 Validation

- Cached mode 不載入 VAE，validation loss 無法用 VAE 評估品質
- Validation epoch 在 cached mode 下只是重複計算 MSE on latents，資訊量有限
- 訓練監控靠 training loss + 固定索引樣本即可

### 固定採樣索引

舊的 epoch-based 採樣（`epoch_0001/sample_000.png`）受 DataLoader shuffle 影響，跨 epoch 無法對比。

改用固定索引採樣：
- 每個 index 對應固定的 shard + offset
- 跨 epoch / 跨模型嚴格可對比
- 只需讀取 dataset[idx]，不增加 GPU 負擔（每個 index 一個 batch）

### 獨立 Decode

Training loop 不需要載入 VAE（省 VRAM）。解碼是**推理階段**任務：
- 訓練時只存 `.pt`（tokens, gt_latents, pred_latents）
- 訓練完後用 `src/decode.py` 載入 VAE 解碼成 `.png`
- 支援 `--device cpu`，不需 GPU 也能解碼

## How

### `src/trainer.py` 修改

**TrainingConfig 變更：**

```python
# Removed:
val_freq: int = 5
sample_images: int = 4

# Added:
sample_indices: list[int] = field(default_factory=list)  # 固定採樣索引
```

**`_save_samples_cached`（原 `_decode_samples_cached`）：**

```python
def _save_samples_cached(mapper, dataset, indices, device, out_dir) -> None:
    """Save sample data for fixed indices (no validation, no VAE)."""
    for idx in indices:
        tokens, gt_latents = dataset[idx]
        pred_latents = mapper(tokens)
        # Save as .pt files
        idx_dir = samples_dir / f"idx_{idx:05d}"
        torch.save(tokens.cpu(), idx_dir / "tokens.pt")
        torch.save(gt_latents.cpu(), idx_dir / "gt_latents.pt")
        torch.save(pred_latents.cpu(), idx_dir / "pred_latents.pt")
```

**`run_training` 樣本保存邏輯：**

```python
# Validation loop removed entirely
# Instead, save samples at each epoch end:
if config.sample_indices:
    _save_samples_cached(
        mapper, train_loader.dataset, config.sample_indices, device, out,
    )
```

### `train.py` CLI 修改

**移除：** `--val-freq`
**新增：** `--sample-indices`（字串解析為 list[int]）

```bash
# 保存索引 0, 2, 4, 8 的樣本
uv run python -m train --cache-dir ~/hdd/cache --sample-indices "0,2,4,8"
```

### `src/decode.py`（新檔案）

```bash
# 批量解碼
uv run python -m src.decode --samples runs/exp/samples

# 解碼單一檔案
uv run python -m src.decode --latent runs/exp/samples/idx_00000/pred_latents.pt

# CPU 解碼
uv run python -m src.decode --samples runs/exp/samples --device cpu
```

### 目錄結構

```
runs/exp/
├── checkpoints/
│   ├── epoch_0001.pt
│   ├── epoch_0005.pt
│   └── final.pt
├── samples/
│   ├── idx_00000/
│   │   ├── tokens.pt        # [1, 1024, 384]
│   │   ├── gt_latents.pt    # [1, 16, 64, 64]
│   │   └── pred_latents.pt  # [1, 16, 64, 64]
│   ├── idx_00002/
│   │   ├── tokens.pt
│   │   ├── gt_latents.pt
│   │   └── pred_latents.pt
│   └── ...
└── logs/train.log
```

### 檔案變更一覽

| 檔案 | 變更 |
|------|------|
| `src/trainer.py` | TrainingConfig 移除 val_freq/sample_images，新增 sample_indices；`_save_samples_cached` 取代 `_decode_samples_cached`；移除 validation loop |
| `train.py` | 移除 `--val-freq`，新增 `--sample-indices` |
| `src/decode.py` | 新檔案：VAE decode 工具 |

## Follow-up

- [ ] 實際測試 `--sample-indices 0,2,4,8` 能否正確寫入 `.pt`
- [ ] 測試 `src/decode.py` 解碼 pipeline
- [ ] 使用新設定重跑訓練

## References

- [21 OOM 修復 + ShardAwareSampler](./21_2026_06_05_agent_cached-oom-fix-and-resnet-mapper.md)
- [src/decode.py](../../src/decode.py) — 新解碼工具
- [src/trainer.py](../../src/trainer.py) — `_save_samples_cached` + run_training
- [train.py](../../train.py) — `--sample-indices` CLI
