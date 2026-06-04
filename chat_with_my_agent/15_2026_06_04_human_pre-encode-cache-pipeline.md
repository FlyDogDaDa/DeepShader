---
created: 2026-06-04
author: FlyDogDaDa
type: human
status: final
tags: [daily-log, training-architecture, pre-encode-cache, pipeline-redesign]
---

# 大重構訓練流程：Pre-Encode Cache 兩階段架構

## What

決定對訓練管線進行大重構，將當前「每個 batch 重複編碼」的設計替換為「先編碼、後訓練」的兩階段架構。

## Why

### 當前瓶頸分析

審視現有訓練流程：

```
每個 batch 都要：
  1. 從 NAS 讀 512×512 JPG    ← 慢 I/O
  2. 跑 DINOv3 (21M params)    ← 編碼器不訓練，白跑
  3. 跑 VAE.encode (12M params) ← 編碼器不訓練，白跑
  4. mapper forward/backward   ← 實際需要的，302K params, ~1ms
```

287K images × 100 epochs = **2870 萬次** DINOv3 + VAE.encode 重複運算。這些編碼器完全不更新，卻每個 batch 都重新計算。

VRAM 瓶頸雖然嚴重（限制 batch size 無法加大），但**根本瓶頸是 I/O + 重複運算**。

## How

### 新架構設計（兩階段）

**Phase 0: Pre-encode（跑一次，未來所有訓練共用）**

```
Dataset (JPG) ──► DINOv3 + VAE.encode ──► shard cache on disk
```

| 參數 | 數值 |
|------|------|
| 每個 shard 圖片數 | 1000 |
| 每張 cache 大小 | patch_tokens(1.58 GB) + gt_latents(0.26 GB) = 1.84 GB |
| shard 總數 | 287K / 1000 ≈ 288 shards |
| 總 cache 容量 | ~530 GB（存放在 ~/hdd） |
| 編碼時間 | 預估 ~2 小時 |

**Phase 1: Training（每次訓練，毫秒級 batch）**

```
shard_reader ──► unpack tensors ──► mapper forward/backward
```

| 優勢 | 說明 |
|------|------|
| NAS I/O 壓力歸零 | cache 在本地 HDD |
| DINOv3/VAE 不再重複 | 只在 pre-encode 跑一次 |
| VRAM 暴降 | 只放 mapper + 當前 batch |
| 可加大 batch size | 沒有 VAE/DINOv3 的 VRAM 負擔 |
| 可加 gradient accumulation | effective batch 不受 VRAM 限制 |
| step time ~10ms | 只有 302K params 的 mapper |

### 檔案結構

```
DeepShader/
├── encode.py              ← 獨立腳本：跑一次，產出 cache
├── train.py               ← 獨立腳本：用 cache 訓練
└── src/
    ├── models.py          ← 不變（mapper 架構）
    ├── cache.py           ← 新增：pre-encode pipeline + cached dataloader
    ├── trainer.py         ← 重構（去掉 DINO/VAE 依賴，專注 mapper training）
    ├── pretrains.py       ← 僅 encode 階段用
    └── utility.py         ← 保留（scan_dataset 等）
```

### 為什麼要分開？

| 考量 | 合併在 train.py (encode 子指令) | 獨立成 encode.py |
|------|--------------------------------|------------------|
| 依賴 | train.py 需 import pretrains.py | encode.py 專責，不混用 trainer |
| 執行時機 | 每次都跑（荒謬）或忘跑（浪費時間） | 明確的一次性任務 |
| CLI 複雜度 | 需要 subcommand / flag 分支 | 各自乾淨 |
| 可重複性 | 改 hyperparams 要重跑 encode | cache 永久，換 hyperparams 直接練 |

### 新的 `src/` 結構

```
src/
├── models.py           # 不變（DinoToVAE_Linear/MLP）
├── cache.py            # 新增：pre-encode pipeline + shard reader
├── trainer.py          # 重構：專為 cached training 設計（去掉 DINO/VAE 依賴）
├── pretrains.py        # 僅給 pre-encode 階段用
└── utility.py          # 保留（scan_dataset 等）
```

### 新的 CLI 介面

```bash
# 步驟 1: 編碼（跑一次）
uv run python encode.py --dataset /data --cache-dir ~/hdd/cache

# 步驟 2: 訓練（可多次，共用同一份 cache）
uv run python train.py --cache-dir ~/hdd/cache --epochs 100

# 步驟 3: 驗證（可選）
uv run python train.py --cache-dir ~/hdd/cache --val-size 0.1
```

### 設計原則

- **先編碼、後訓練**：把所有時間花在刀刃上（mapper training）
- **Cache 永久存在**：同一份 dataset 可做多種實驗（不同 mapper、不同 hyperparams）
- **不綁死過往設計**：借鑑現有架構但重新設計，不被舊程式束縛
- **可擴展**：未來可換 DINOv3 模型（S→B→L）而不重編整個 dataset

## Follow-up

- [ ] 設計 `cache.py` 的 pre-encode pipeline（DINOv3 + VAE.encode → disk shard）
- [ ] 設計 shard format（torch.save / mmapable）
- [ ] 設計 `CachedDataset` / `ShardReader` 讀取 shard files
- [ ] 獨立 `encode.py` CLI 腳本（pre-encode pipeline 入口）
- [ ] 重構 `trainer.py`（去掉 DINO/VAE，專為 cached training 優化）
- [ ] 重構 `train.py`（改用 cache，移除 VAE/DINOv3 載入邏輯）
- [ ] 執行 pre-encode（預估 ~2 小時）
- [ ] 比較 pre-encode vs 傳統 training 的速度差異

## References

- [14 VRAM Problem Diagnosis](./14_2026_06_04_agent_vram-problem-diagnosis.md) — 當前瓶頸分析
- [13 Training Pipeline Restructure](./13_2026_06_03_agent_training-pipeline-restructure.md) — 當前架構
- [src/trainer.py](../../src/trainer.py) — 當前訓練管線
- [src/dataset.py](../../src/dataset.py) — 當前 dataset 設計
- [src/pretrains.py](../../src/pretrains.py) — DINOv3/VAE wrapper
