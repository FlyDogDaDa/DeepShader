# DeepShader Chat History Index

> **專案**：DeepShader — 使用 DINOv3 特徵 + FLUX VAE 解碼的 Anime 圖片生成系統  
> **時間跨度**：2026-06-02 ~ 2026-06-04（3 天）  
> **總記錄數**：20 筆（15 agent 筆記 + 5 human 筆記 + 3 script 資料夾）

---

## 📌 專案核心方向（當前）

```
圖片 → DINOv3 → 特徵 → 輕量映射（Mapper） → FLUX VAE Decoder → 圖片
```

**關鍵決策**：跳過 Diffusion/Flow Model，直接將 DINOv3 patch tokens 映射到 FLAX VAE latent space。  
**需訓練的唯一模型**：`DinoToVAE` mapper（~300K params，遠小於 DINOv3 的 21M）。

---

## 🗺️ 發展時間線

### Phase 0：基礎建設與研究（06-02）

| # | 日期 | 標題 | 類型 | 摘要 |
|---|------|------|------|------|
| 00 | 06-02 | Skill & Project Setup | Agent | 建立 daily-log skill、Git 設定、`uv init` 專案初始化 |
| 01 | 06-02 | DINOv3 模型規格研究 | Agent | 完整分析 10 個 DINOv3 模型（ViT-S~7B + ConvNeXt T~L），patch=16，2.182B 訓練數據 |
| 02 | 06-02 | AnimePro FLUX VAE 設定檔分析 | Agent | 詳解 VAE 配置：downsample=8×、latent=16ch、sample_size=1024、decode formula=`x/0.3611+0.1159` |

### Phase 1：資料集與基礎管線（06-03 上午）

| # | 日期 | 標題 | 類型 | 摘要 |
|---|------|------|------|------|
| 03 | 06-03 | Tagged Anime Illustrations Dataset 發現 | Agent | 找到 NAS 上的 337K 張 512×512 JPG + metadata（Danbooru2017 + MoeImouto） |
| 04 | 06-03 | Dataset Object with PyTorch Transforms | Agent | 建立 `src/dataset.py`、`src/utility.py`、ImageDataset + default_transform |
| 05 | 06-03 | Dataset Size & RAM Check | Agent | RAM 32GB vs 994GB 展開大小 → 不可全載，DataLoader lazy-load 是唯一解 |
| 06 | 06-03 | NAS NFS Read Speed Benchmark | Agent | IOPS=85、sequential=9MB/s、建議 `num_workers=8~16` |
| 07 | 06-03 | Test Suite for Dataset Module | Agent | pytest 26 tests 全通過，修正 3 個 bug（cache break、prefetch_factor、shuffle） |
| 08 | 06-03 | scan_dataset Cache | Agent | `~/.cache/deepshader/` mtime-based 快取，避免重複目錄掃描 |

### Phase 2：架構轉向與 PoC 驗證（06-03 下午）

| # | 日期 | 標題 | 類型 | 摘要 |
|---|------|------|------|------|
| 09 | 06-03 | **架構方向變更**：DINO→VAE 直接映射 | Human | 決定跳過 Diffusion/Flow，改用輕量 mapper + VAE decode |
| 10 | 06-03 | Full Pipeline PoC 成功 | Agent | `圖片→DINOv3→mapper→VAE→圖片` 端到端通過，mapper 僅 6,160 params |
| 11 | 06-03 | Mini-batch Training PoC | Agent | 3 epochs × 7 batches 跑完，gradient flow 驗證通過 |
| 12 | 06-03 | Overfit Experiment：Linear vs MLP | Agent | Linear→loss 1.02/MLP→loss 0.56，MLP 可學但有過曝問題 |
| 13 | 06-03 | Training Pipeline Restructure | Agent | 專案重整：`models.py`+`trainer.py`+`train.py`，debug run loss 降 46% |

### Phase 3：VRAM 優化與兩階段架構（06-04）

| # | 日期 | 標題 | 類型 | 摘要 |
|---|------|------|------|------|
| 14 | 06-04 | VRAM Problem Diagnosis | Agent | **關鍵發現**：VAE encode 無 `no_grad` 佔用 93% VRAM。修復後降 87-93% |
| 15 | 06-04 | **大重構決定**：Pre-Encode Cache 兩階段 | Human | 每 batch 重跑 DINOv3+VAE.encode = 2870 萬次白費。改為「先編碼→後訓練」 |
| 16 | 06-04 | Cache 架構設計 | Agent | Shard 存儲格式、LRU cache、**ShardAwareSampler**（hit rate 2.8%→97.2%） |
| 17 | 06-04 | Cache Pipeline 實作 | Agent | `src/cache.py`+`src/encode.py` 完成，22 tests，resume 機制，float16 存儲 |
| 18 | 06-04 | Trainer & Train.py 雙模式重構 | Agent | `trainer.py` 自動偵測 cached/image mode，`train.py` CLI 加 `--cache-dir` |
| 19 | 06-04 | Encode.py Buffer Refactor + Fix | Agent | memory leak 修復、tqdm 進度條、`--batch-size` 參數、`--resume` bug fix |

### 📂 Script 資料夾

| 資料夾 | 內容 |
|--------|------|
| `10_2026_06_03_full-pipeline-poc-scripts/` | `poc_full_pipeline.py`, `poc_vae_roundtrip.py`, `poc_vae_only.py`, `test_mapping_network.py` |
| `11_2026_06_03_training-poc-scripts/` | `train_poc.py` |
| `12_2026_06_03_training-poc-scripts/` | `overfit_experiment.py` |

---

## 🔑 關鍵技術要點

### 模型規格

| 組件 | 規格 |
|------|------|
| **DINOv3** | ViT-S/16，21M params，hidden=384，patch tokens=1024（512×512 輸入） |
| **Mapper** | DinoToVAE_Linear(6K) / DinoToVAE_MLP(302K)，輸出 [B, 16, 64, 64] |
| **FLUX VAE** | AutoencoderKL，downsample=8×，latent=16ch，decode=`x/0.3611+0.1159` |
| **Dataset** | 337,038 張 512×512 JPG，NAS NFS 儲存 |

### 兩階段架構

```
Phase 0: Pre-encode（跑一次，永久共用）
  Dataset JPG → DINOv3 + VAE.encode → shard cache on disk (~530 GB, 288 shards)

Phase 1: Training（每次訓練，毫秒級 batch）
  shard_reader → unpack tensors → mapper forward/backward（~10ms/step）
```

### Cache 系統

| 組件 | 說明 |
|------|------|
| `ShardCache` | LRU cache，8 shards 容量，hit rate ~97% |
| `ShardAwareSampler` | 兩層 shuffle：shard 順序→shard 內順序 |
| `CachedDataset` | O(1) index map 查找，自動載入 shard |
| `CacheManifest` | 版本控制、dataset hash、模型版本追蹤 |
| `encode.py` | CLI pre-encode 工具，支援 resume、tqdm、batch-size 調整 |

---

## 🔧 已知待處理事項

| 狀態 | 項目 | 來源 |
|------|------|------|
| ⏳ | 執行 pre-encode（預估 ~2 小時） | [15](./15_2026_06_04_human_pre-encode-cache-pipeline.md) |
| ⏳ | 實測 cached training loop | [18](./18_2026_06_04_agent_trainer-and-train-py-double-mode.md) |
| ⏳ | 驗證 image mode backward compat | [18](./18_2026_06_04_agent_trainer-and-train-py-double-mode.md) |
| ⏳ | 清理 type warnings（不影響功能） | [18](./18_2026_06_04_agent_trainer-and-train-py-double-mode.md) |
| ⏳ | 檢查 decoded 圖片品質 | [10](./10_2026_06_03_agent_full-pipeline-poc-complete.md) |

---

## 📁 文件列表（按時間排序）

| # | 日期 | 檔案 | 快速連結 |
|---|------|------|----------|
| 00 | 06-02 | `00_agent_skill-and-project-setup.md` | [↗](./00_2026_06_02_agent_skill-and-project-setup.md) |
| 01 | 06-02 | `01_agent_dinov3-model-specifications.md` | [↗](./01_2026_06_02_agent_dinov3-model-specifications.md) |
| 02 | 06-02 | `02_agent_animepro-flux-vae-analysis.md` | [↗](./02_2026_06_02_agent_animepro-flux-vae-analysis.md) |
| 03 | 06-03 | `03_agent_tagged-anime-illustrations-dataset-lookup.md` | [↗](./03_2026_06_03_agent_tagged-anime-illustrations-dataset-lookup.md) |
| 04 | 06-03 | `04_agent_dataset-object-pytorch-transforms.md` | [↗](./04_2026_06_03_agent_dataset-object-pytorch-transforms.md) |
| 05 | 06-03 | `05_agent_dataset-size-and-ram-check.md` | [↗](./05_2026_06_03_agent_dataset-size-and-ram-check.md) |
| 06 | 06-03 | `06_agent_nas-speed-test.md` | [↗](./06_2026_06_03_agent_nas-speed-test.md) |
| 07 | 06-03 | `07_agent_test-suite.md` | [↗](./07_2026_06_03_agent_test-suite.md) |
| 08 | 06-03 | `08_agent_scan-dataset-cache.md` | [↗](./08_2026_06_03_agent_scan-dataset-cache.md) |
| 09 | 06-03 | `09_human_vae-direct-feature-decode.md` | [↗](./09_2026_06_03_human_vae-direct-feature-decode.md) |
| 10 | 06-03 | `10_agent_full-pipeline-poc-complete.md` | [↗](./10_2026_06_03_agent_full-pipeline-poc-complete.md) |
| 11 | 06-03 | `11_agent_mini-batch-training-poc.md` | [↗](./11_2026_06_03_agent_mini-batch-training-poc.md) |
| 12 | 06-03 | `12_agent_overfit-experiment.md` | [↗](./12_2026_06_03_agent_overfit-experiment.md) |
| 13 | 06-03 | `13_agent_training-pipeline-restructure.md` | [↗](./13_2026_06_03_agent_training-pipeline-restructure.md) |
| 14 | 06-04 | `14_agent_vram-problem-diagnosis.md` | [↗](./14_2026_06_04_agent_vram-problem-diagnosis.md) |
| 15 | 06-04 | `15_human_pre-encode-cache-pipeline.md` | [↗](./15_2026_06_04_human_pre-encode-cache-pipeline.md) |
| 16 | 06-04 | `16_agent_cache-architecture-design.md` | [↗](./16_2026_06_04_agent_cache-architecture-design.md) |
| 17 | 06-04 | `17_agent_cache-pipeline-implementation.md` | [↗](./17_2026_06_04_agent_cache-pipeline-implementation.md) |
| 18 | 06-04 | `18_agent_trainer-and-train-py-double-mode.md` | [↗](./18_2026_06_04_agent_trainer-and-train-py-double-mode.md) |
| 19 | 06-04 | `19_agent_encode-py-refactor-and-fix.md` | [↗](./19_2026_06_04_agent_encode-py-refactor-and-fix.md) |

---

## 📊 進度統計

| 指標 | 數值 |
|------|------|
| 總筆記數 | 20（15 agent + 5 human） |
| 測試覆蓋 | 74 tests passing |
| Mapper 模型大小 | 6K（Linear）/ 302K（MLP） params |
| Dataset 規模 | 337,038 images × 512×512 × 3 channels |
| Cache 預估容量 | ~530 GB（288 shards × 1.84 GB） |
| 預計訓練速度提升 | 從 2870 萬次重複編碼 → 僅讀取 shard I/O |
