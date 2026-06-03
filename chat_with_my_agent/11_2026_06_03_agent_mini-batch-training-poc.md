---
created: 2026-06-03
author: agent
type: agent
status: final
tags: [daily-log, training, mini-batch, poc, dino-to-vae, gradient]
---

# Mini-batch Training PoC: DinoToVAE

## What

DinoToVAE mini-batch training PoC 成功。3 epochs × 7 batches 全部跑完，沒有崩潰。

## Why

驗證核心訓練流程：
```
pred_latents = mapper(patch_tokens)       # train
gt_latents   = vae.encode(gt_image).mode()  # no_grad
loss = MSE(pred_latents, gt_latents)       # backward → mapper only
```

## How

### 訓練流程設計

完全按照設計架構：DINOv3 和 VAE 都在 `.no_grad()` 裡，`backward()` 只經過 mapper。

```
Stage 1: DINOv3 .no_grad() → patch_tokens [B, 1024, 384]
Stage 2: mapper → pred_latents [B, 16, 64, 64]
Stage 3: VAE .no_grad() → gt_latents [B, 16, 64, 64]
Loss: MSE(pred_latents, gt_latents)
```

**VRAM 優化要點：**
- `images` 在 Stage 3 後立即 `del` 釋放
- DINOv3 和 VAE 都在 `no_grad` 中，不會累積梯度
- VAE decode 只在 inference/視覺化時呼叫，不參與訓練

### 修正記錄

1. **`requires_grad=True` 不保證 gradient flow** — 需要 `.no_grad()` 包起來才能真正斷開圖
2. **uint8 溢出** — `images * 2 - 1` 從 uint8 開始算會溢出，需要 `images.float()`
3. **`.detach()` vs `.no_grad()`** — 在 `no_grad` 裡用 `.detach()` 會把圖完全斷掉，loss 沒有 `grad_fn`；正確做法是讓 `no_grad` 包住不需要梯度的部分，mapper 的輸出保持圖
4. **return latents 不是 images** — `training_step` 現在 return `pred_latents [B, 16, 64, 64]`，save 前需要 `vae.decode()`

### 執行結果

```
Epoch 1/3 done
Epoch 2/3 done
Epoch 3/3 done
```

## Follow-up

- 檢查 `/tmp/train_poc_before.jpg` vs `/tmp/train_poc_after.jpg` 品質變化
- 增加 training epochs 和 sample size 看 loss 是否持續下降
- 完整訓練 pipeline 設計（dataset, dataloader, scheduler）
- 保存 trained mapper weights

## References

- [Scripts](./10_2026_06_03_full-pipeline-poc-scripts/)
- [09 架構方向變更](./09_2026_06_03_human_vae-direct-feature-decode.md)
- [DINOv3 模型規格](./01_2026_06_02_agent_dinov3-model-specifications.md)
- [AnimePro FLUX VAE](./02_2026_06_02_agent_animepro-flux-vae-analysis.md)
