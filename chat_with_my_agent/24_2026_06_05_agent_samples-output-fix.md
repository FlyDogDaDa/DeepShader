---
created: 2026-06-05
author: agent
type: agent
status: final
tags: [daily-log, sample-output, trainer-fix]
---

# 修正 sample 輸出：訓練前存 + 每 epoch 存 + 正確檔案結構

## What

修正 `_save_samples_cached` 的行為，使 `--sample-indices` 設定時正確輸出預測結果到 `samples/` 目錄。

## Why

舊版有兩個問題：

1. **訓練前沒有存初始樣本**：只在每個 epoch 結束時存，缺少 `epoch=0` 的隨機權重基線
2. **檔案結構不符預期**：舊版將每個 sample 存到 `samples/idx_00000/tokens.pt` 等獨立目錄，不是一個 epoch 一個檔案

正確需求是一個 epoch 一個檔案：`samples/sample_0000.pt`（訓練前）、`samples/sample_0001.pt`（epoch 1 後）、`samples/sample_0002.pt`（epoch 2 後）。

## How

### `src/trainer.py`：`_save_samples_cached` 重構

簽名從 `prefix: str` 改為 `epoch: int = 0`。

輸出結構改為一個 epoch 一個檔案，每個檔案是一個 dict：
- key：`idx_0`, `idx_1`, `idx_2`, `idx_3`（對應指定的 sample indices）
- value：`{"tokens": ..., "gt_latents": ..., "pred_latents": ...}`

### `src/trainer.py`：新增訓練前初始樣本保存

在 `run_training` 的 training loop 之前，若 `config.sample_indices` 非空即呼叫 `_save_samples_cached(epoch=0)`。

### `src/trainer.py`：epoch 結束時保存

training loop 內每輪結束呼叫 `_save_samples_cached(epoch=epoch + 1)`。

## Validation

```bash
uv run pytest tests/ -x -q    # 74 passed
```

實際執行驗證：

```bash
python -m train --cache-dir ~/hdd/cache --sample-indices 0,1,2,3 --debug --output runs/debug_sample
```

輸出確認：

```
[train] Saved samples at epoch 0 to runs/.../samples/sample_0000.pt   ← 訓練前
[INFO] Epoch 1/2  loss=...
[train] Saved samples at epoch 1 to runs/.../samples/sample_0001.pt   ← epoch 1 後
[INFO] Epoch 2/2  loss=...
[train] Saved samples at epoch 2 to runs/.../samples/sample_0002.pt   ← epoch 2 後
```

檔案結構：

```
samples/
├── sample_0000.pt  (269 MB, epoch 0 隨機權重)
├── sample_0001.pt  (269 MB, epoch 1 結束後)
└── sample_0002.pt  (269 MB, epoch 2 結束後)
```

各 epoch 的 pred_latents MSE 不為零（0.91~1.31），確認 model 確實在學習。

## References

- [src/trainer.py](../../src/trainer.py) — `_save_samples_cached`, `run_training`
- [23_2026_06_05_agent_debug-mode-subsample](./23_2026_06_05_agent_debug-mode-subsample.md) — debug mode 優化
