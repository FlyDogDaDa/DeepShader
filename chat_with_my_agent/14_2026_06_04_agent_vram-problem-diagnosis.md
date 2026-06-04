---
created: 2026-06-04
author: agent
type: agent
status: final
tags: [daily-log, vram-problem, gradient-graph, training-memory, dataset-resize]
---

# VRAM Problem Diagnosis + Training Pipeline Fix

## What

1. **修正 dataset transform** — 加入 `Resize(512) + CenterCrop(512)` 解決圖片尺寸不一致問題
2. **修正 VAE encode gradient leak** — 將 `vae.encode()` 包進 `torch.no_grad()`，VRAM 佔用下降 87–93%
3. **增加 training logging frequency** — 每 100 steps 寫入 log，而非僅 epoch 結束才 log
4. **VRAM profiling 完成** — 用自寫 profile 腳本精確測量各階段 VRAM 佔用

## Why

### 圖片尺寸不一致

Dataset 混雜了 123×160、512×650、512×512 等多種尺寸圖片。`default_transform()` 原本只有 augmentation 沒有 resize，DataLoader collate 時 tensor 尺寸不一致導致 `RuntimeError: stack expects each tensor to be equal size`。

### VRAM 爆量的真正原因

Mapper 模型本身非常小（MLP ~302K params = 1.2 MB），但訓練時 batch=2 就佔了 6.6 GB。VRAM profiling 顯示：

| Batch | 舊 code（VAE with grad） | 新 code（VAE no_grad） | 節省 |
|-------|------------------------|----------------------|------|
| 1 | 3,458 MB | 439 MB | **87%** |
| 2 | 6,682 MB | 456 MB | **93%** |

VAE encode 如果在 gradient context 下執行，PyTorch 會為 FLUX VAE encoder 的 **所有 intermediate activations** 建立 gradient graph。FLUX VAE encoder 很深（多個 down_block + resnet），batch=2 時保住了每一層的 activation tensors，導致 VRAM 爆表。

### Logging 頻率低

1 epoch = 143,768 steps（287K 圖片 / batch=2），但原本只在 epoch 結束才 log 一次，中間完全沒有進度回饋。

## How

### 檔案修改

**`src/dataset.py`** — `default_transform()` 加入 resize + crop：
```python
return T.Compose([
    T.Resize(512),
    T.CenterCrop(512),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
])
```

**`src/trainer.py`** — 三處修改：
1. `TrainingConfig` 新增 `log_freq` 欄位
2. `train_epoch()` 加入 `with torch.no_grad():` 包 VAE encode
3. `train_epoch()` 每 `log_freq` 次 step 寫 log（含 step #、batch #、loss、lr）

**`train.py`** — CLI 新增 `--log-freq` 參數

### Profiling 腳本

寫了 `profile_vram.py` 精確測量 pipeline 各階段的 VRAM 使用：
- DINO extract（no_grad 本來就安全）
- Mapper forward
- VAE encode（before/after 對比）
- Loss backward

## Follow-up

- **實際訓練 VRAM 測試** — 用 `batch_size=16` 跑了一輪，VRAM 使用 11,455 / 12,282 MiB (93%)，效果不錯但沒有太多頭
- **正式訓練** — 用 `batch_size=8` 應該可以安全跑（剩 ~1 GB free），平衡速度與穩定性
- **DINOv3 memory** — DINOv3 ViT-S forward 也保住了中間 activation，未來可考慮 gradient checkpointing 或改用更小的模型（如果適用）
- **checkpoint 驗證** — 訓練開始後確認 checkpoint 可以正常 resume

## 2026-06-04 後續更新

### 訓練管線更新

1. **`__init__.py` 清理** — 移除不存在的 `ImageLoader` export（已改名為 `ImageDataset`）
2. **`train.py` 清理** — 移除未使用的 `--max-images` 和 `--in-memory` CLI 參數
3. **`train_epoch()` 增加 `epoch` 參數** — 修正 step counting 缺少的 bug
4. **step 1 強制寫 log** — 訓練一開始就顯示進度，確認 pipeline 正常啟動
5. **DataLoader `prefetch_factor` 加大** — 從 2 改到 20，提升 NAS dataset 讀取效率
6. **CLI 新增 `--prefetch` 參數** — 可自訂 prefetch 大小

### VRAM 分析結論

Mapper 模型本身極小（302K params ≈ 1.2 MB），VRAM 爆量的真正原因是 **VAE encode 在 gradient context 下執行**，PyTorch 為 FLUX VAE encoder 的整個 forward pass 保留所有 intermediate activations。加上 `torch.no_grad()` 後 VRAM 下降 87–93%。

### Training 進度 (Step 1–5,000)

```
step 1,000   loss=2.1132  lr=0.001000
step 2,000   loss=1.9582  lr=0.001000  ← 最低點
step 3,000   loss=2.0591  lr=0.001000
step 4,000   loss=1.9888  lr=0.001000
step 5,000   loss=2.0210  lr=0.001000  ← plateau
```

**分析**：loss 停在 2.0 左右，中等偏弱。原因可能是：
- mapper 容量不足（302K params 太薄）
- latent space 的映射本身困難（overfit 實驗顯示即使 500 steps 過 fit 仍過曝）
- 純 MSE loss 對 VAE latent space 不夠敏感

## References

- [src/dataset.py](../../src/dataset.py) — transform pipeline
- [src/trainer.py](../../src/trainer.py) — training loop + logging
- [train.py](../../train.py) — CLI interface
- [profile_vram.py](../../profile_vram.py) — VRAM profiling tool
- [13 Training Pipeline Restructure](./13_2026_06_03_agent_training-pipeline-restructure.md)
