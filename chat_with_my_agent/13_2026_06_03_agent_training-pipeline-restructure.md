---
created: 2026-06-03
author: agent
type: agent
status: final
tags: [daily-log, project-structure, test-driven, training-pipeline, cli, debug-run, vram]
---

# Training Pipeline: Project Restructure + Debug Run

## What

完成訓練管線的重構與實作，包含新的專案結構、測試套件、以及 `train.py` 訓練入口。Debug run 成功驗證完整流程。

## Why

之前的 PoC 腳本分散在不同資料夾，無法直接用於正式訓練。需要一個統一的、可重現的訓練框架：

1. 模型（`models.py`）：Linear + MLP mapper，含可學習 normalization
2. 預訓練模型載入（`pretrains.py`）：VAE encode/decode + DINOv3 extract 封裝
3. 訓練管線（`trainer.py`）：config、checkpoint、log、sample decode
4. 訓練入口（`train.py`）：CLI 介面，`--output` + `--debug` 參數
5. 測試覆蓋：20 個新測試 + 29 個既有測試 = 49/49 通過

## How

### 專案結構重整

```
src/
├── __init__.py          # 公共 API 匯出（20+ symbols）
├── pretrains.py         # VAEModel / DINOModel wrapper
├── models.py            # DinoToVAE_Linear / DinoToVAE_MLP
├── trainer.py           # TrainingConfig + train/validate epoch + checkpoint + logging
├── dataset.py           # 既有（未修改）
└── utility.py           # 既有（未修改）

train.py                 # 訓練入口 CLI
tests/
├── test_mapper.py       # 9 tests
├── test_pretrains.py    # 7 tests
└── test_trainer.py      # 4 tests
```

### 關鍵決策

| 決策 | 原因 |
|------|------|
| `output_dir` 取代 `checkpoint_dir` | 統一所有輸出一個根目錄（config、checkpoints、samples、logs） |
| normalization 用 `nn.LayerNorm` 自動學 | 比固定 z-score 更靈活 |
| 測試先行（TDD） | 確保每個 module 行為正確 |
| `_decode_samples` 用 `vae.decode()` 直接回傳 Tensor | diffusers 新版 API 回傳不是 named tuple |

### Debug Run 結果

```
batch=2, epochs=2, 40 images (debug subsample)

Epoch 1/2  loss=8.5872  val=6.4065
Epoch 2/2  loss=4.6795  val=4.8193

loss 下降 46%（兩個 epoch 的 mini-batch 結果）
```

### VRAM 監控

```
GPU 1: 8933 MiB / 12282 MiB (~73%)  ← batch=2
GPU 0: 4 MiB (idle)
```

### 修正記錄

1. **`TrainingConfig` 缺少 `val_ratio`** — 新增欄位並從 `train.py` 正確使用
2. **`src/__init__.py` 漏導 `create_dataloaders`** — 補上 `src/dataset` 匯入
3. **`VAEModel.decode()` 回傳 Tensor 非 named tuple** — 移除 `.sample` 屬性存取
4. **Debug 模式 batch=8 會 OOM** — debug mode 改為 batch=2
5. **cudnn.so.9 缺失** — 強迫重新安裝 torch + nvidia-cudnn-cu13
6. **`load_dotenv()` 未呼叫** — 在 `train.py` main 開頭加上

### 輸出目錄結構

```
runs/debug-test_20260603_141539/
├── config.yaml           # YAML 完整訓練配置
├── checkpoints/
│   ├── epoch_0001.pt     # 階段 checkpoint
│   ├── epoch_0002.pt
│   └── final.pt          # 最終模型
├── samples/
│   ├── epoch_0001/       # 解碼 sample 圖片（4張）
│   │   ├── sample_000.png
│   │   ├── sample_001.png
│   │   ├── sample_002.png
│   │   └── sample_003.png
│   └── epoch_0002/
│       ├── sample_000.png
│       ├── sample_001.png
│       ├── sample_002.png
│       └── sample_003.png
└── logs/
    └── train.log         # 訓練 log（含 debug shape 輸出）
```

## Follow-up

- **加大 batch_size** — GPU 1 還有 ~3GB，可嘗試 batch=4 或 6
- **正式訓練** — 用 `--output` 指定名稱，不用 `--debug`
- **檢查解碼圖片品質** — 觀察 `samples/` 下的圖片
- **多 GPU** — 可用 `CUDA_VISIBLE_DEVICES=0` 或 `1` 切換
- **VRAM profile** — 在完整訓練時持續監控

## References

- [src/__init__.py](../../src/__init__.py) — 公共 API 匯出
- [src/pretrains.py](../../src/pretrains.py) — VAE/DINO wrapper
- [src/models.py](../../src/models.py) — Mapper 模型
- [src/trainer.py](../../src/trainer.py) — 訓練管線
- [train.py](../../train.py) — 訓練入口 CLI
- [tests/test_mapper.py](../../tests/test_mapper.py) — mapper 測試
- [tests/test_pretrains.py](../../tests/test_pretrains.py) — pretrain 測試
- [tests/test_trainer.py](../../tests/test_trainer.py) — trainer 測試
- [pyproject.toml](../../pyproject.toml) — 新增 pyyaml 依賴
