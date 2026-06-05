---
created: 2026-06-05
author: agent
type: agent
status: final
tags: [daily-log, decode-py-reformat, samples-to-images]
---

# `decode.py` 改版：支援新樣本格式 → PNG 解碼

## What

重寫 `src/decode.py` 以支援新的 sample 輸出格式，將 `sample_*.pt` 解碼成可視 PNG 圖片。

## Why

舊版 `decode.py` 找的是 `samples/idx_XXXXX/pred_latents.pt` 分目錄格式，但新版 trainer 輸出改為 `samples/sample_{epoch}.pt`（單一檔案包含所有 idx 的 dict）。舊格式已不匹配，無法解碼。

## How

### `src/decode.py` 重構

**輸入格式改變：**
- 舊：`rglob("pred_latents.pt")` 搜尋巢狀目錄結構
- 新：`glob("sample_*.pt")` 讀取扁平結構

**輸出結構改變：**
- 舊：`samples/idx_00000_decoded.png`
- 新：`samples/sample_0000_idx0.png`（每個 epoch × 每個 idx 一張圖）

**內部結構：**
- `sample_*.pt` 是一個 dict：`{"idx_0": {tokens, gt_latents, pred_latents}, "idx_1": ...}`
- decode 時遍歷 dict 的每一個 `idx_*` key，取出 `pred_latents` 解碼成圖片
- 輸出命名 `{stem}_idx{idx_num}.{fmt}`，例如 `sample_0000_idx0.png`

**API：**
- `--samples runs/<run>/samples` — 批次解碼所有 `sample_*.pt`
- `--latent runs/<run>/samples/sample_0001.pt` — 解單一檔案
- `--device cpu` / `cuda` — 選擇解碼裝置
- `--format png` / `jpg` — 輸出格式

### 測試驗證

```bash
uv run python -m src.decode --samples runs/debug_sample_20260605_131403/samples --device cpu
# Found 3 samples to decode
# Decoded 12 images to .../samples  (4 idx × 3 epoch)
```

輸出確認：
```
sample_0000_idx0.png  ~  sample_0000_idx3.png   ← epoch 0 隨機權重
sample_0001_idx0.png  ~  sample_0001_idx3.png   ← epoch 1 學習後
sample_0002_idx0.png  ~  sample_0002_idx3.png   ← epoch 2 進一步學習
```

## Follow-up

- [ ] 驗證 cached-resnet 訓練跑完後的樣本解碼（目前只有 `sample_0000.pt`）
- [ ] 解碼結果視覺化比對各 epoch 的圖片品質變化

## References

- [src/decode.py](../../src/decode.py) — 完整的 decode 邏輯
- [src/trainer.py](../../src/trainer.py) — `_save_samples_cached` 寫入格式
- [24_2026_06_05_agent_samples-output-fix](./24_2026_06_05_agent_samples-output-fix.md) — sample 輸出格式變更
