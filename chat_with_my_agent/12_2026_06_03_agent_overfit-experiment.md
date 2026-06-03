---
created: 2026-06-03
author: agent
type: agent
status: final
tags: [daily-log, overfit-experiment, training, mlp, latent, decode]
---

# Overfit Experiment: Linear vs MLP Mapper

## What

執行 single-image overfit 實驗，測試 DinoToVAE pipeline 在極端條件下能否學到任何有效映射。比較 pure linear mapper 與 MLP mapper 的訓練結果。

## Why

mini-batch training PoC 的 decoded 結果是灰階雜訊，可能原因：
1. mapper 太弱（pure linear，6K params）
2. DINOv3 feature → VAE latent 是根本上不可學的空間變換
3. pipeline 架構有 bug

Overfit 實驗可以回答：如果 mapper 有無限 capacity 且看到完美目標，能不能讓 decoded 圖片看起來對？如果過 fit 成功 → 問題在 mapper capacity 或訓練策略。如果過 fit 仍然雜訊 → 根本問題在 feature-space mismatch。

## How

### 實驗設計

在 `overfit_experiment.py` 中對單一 image 做 500 steps overfit，分三個階段：

**Phase 1: Linear mapper — BEFORE training** (random init baseline)

**Phase 2: Overfit Linear** — `nn.Linear(384→16)`，500 steps，lr=1e-3
- 6,160 params，最簡單的 baseline

**Phase 3: Overfit MLP** — 4-layer MLP（hidden=256），500 steps，lr=1e-3
- 302,096 params，non-linear 映射

每個 phase 在 step 0/100/200/300/400 印出 pred_latents 統計與 loss。

### 訓練流程

```
Input image [1, 3, 512, 512]
    ↓ DINOv3 .no_grad()
patch_tokens [1, 1024, 384]
    ↓ mapper
pred_latents [1, 16, 64, 64]
    ↓ VAE.decode .no_grad()
decoded [1, 3, 512, 512]

Loss: MSE(pred_latents, vae.encode(gt).mode())
      → backward → mapper only
```

### 結果摘要

| 階段 | Init loss | Step 500 loss | pred_std → gt_std |
|------|-----------|---------------|-------------------|
| Linear | 42.36 | 1.02 | 0.96 → 5.78 |
| MLP | 40.18 | 0.56 | 1.18 → 5.78 |

### 視覺結果

| 圖片 | 結果 |
|------|------|
| overfit_ref_raw.png | 原始圖片：綠色百葉窗，窗外有建築 |
| overfit_linear_before_raw.png | Linear before：全黑（random init） |
| overfit_linear_after_raw.png | Linear after：有輪廓但過曝 |
| overfit_mlp_after_raw.png | MLP after：輪廓更清晰，但仍過曝 |

decoded 過曝（mean ≈ +0.73）是因為 VAE decoder 對 extreme latent values 會 saturate，但這不影響 latent-space training loss。

### 修正記錄

1. **`v2.ToImage()` 回傳 3D tensor** — 需要 `.unsqueeze(0)` 變成 4D `[1, 3, H, W]`
2. **DINOv3 patch tokens 切片** — 用 `[:, 5:, :]` 跳過 class token + 4 pos embeds，與 `train_poc.py` 一致
3. **`load_dotenv()` 有 import 但沒呼叫** — 加上 `load_dotenv()` 並用 `src/utility.scan_dataset`
4. **MLP mapper dimension mismatch** — global context 需要先 `project` 到 hidden_channels 再 concat，否則 fuse layer 的輸入維度不匹配
5. **Output 過吵** — 將 log 改為 `verbose=(step % 100 == 0)`，每 100 步印一次

### 檔案整理

`train_poc.py` 和 `overfit_experiment.py` 移至專屬資料夾：
- `11_2026_06_03_training-poc-scripts/train_poc.py`
- `11_2026_06_03_training-poc-scripts/overfit_experiment.py`

## Follow-up

- 用 MLP mapper 做更完整的訓練（更多資料 + 更多 epochs）
- 檢查 VAE decode 過曝問題：可能是 training 時 VAE latent 值域太廣，需要 normalization 或不同的 latent space
- 完整訓練 pipeline 設計（dataset, dataloader, scheduler）

## References

- [10 Full Pipeline PoC](./10_2026_06_03_agent_full-pipeline-poc-complete.md)
- [11 Mini-batch Training PoC](./11_2026_06_03_agent_mini-batch-training-poc.md)
- [12 Training PoC Scripts](./12_2026_06_03_training-poc-scripts/)
