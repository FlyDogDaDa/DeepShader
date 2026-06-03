---
created: 2026-06-03
author: FlyDogDaDa
type: human
status: final
tags: [daily-log, architecture, dino-v3, vae, latent, pipeline]
---

# 架構方向變更：直接用特徵→VAE還原，跳過 Diffusion/Flow Model

## What

修正技術方向：不再使用 Diffusion / Flow Model 生成，改為直接將 DINOv3 特徵映射到 latent space，再由 FLUX VAE 解碼回圖片。

## Why

看完 agent 的 DINOv3 規格報告後，我想到 DINOv3 提取的特徵已經足夠豐富，可能足以直接還原像素內容。

因此決定不用 Diffusion / Flow Model 來「腦補」，直接使用特徵 + VAE 解碼即可。

這個決定有兩層理由：

1. **技術上**：DINOv3 是用 2.182B 張圖訓練出來的，其 embedding 本身已經包含豐富的語意與風格資訊。如果能直接映射到 latent space，就省去了 Diffusion 去無為有所需的龐大計算量。

2. **效能上**：最終用途是 Vtuber 皮或動畫渲染，每幀都要解碼。跳過 Diffusion/Flow Model 的迭代去噪（6~50 steps × Transformer 推論），直接 `latent → decode` 一次過，FPS 會高很多。

## How

### 核心流程（變更後）

```
圖片 → DINOv3 → 特徵 → 輕量映射 → FLUX VAE Decoder → 圖片
```

唯一需要訓練的是 **特徵↔Latent 映射網路**（輕量 MLP / Linear 層），將 DINOv3 embedding 投影到 FLUX latent space（16 channels, 128×128）。

### 架構要點

1. **DINOv3 特徵提取**：使用訓練好的模型（ViT-S/B/L）提取圖片特徵。
   - **用 patch tokens，不用 class token**（196 個 tokens 是 224×224 輸入；512×512 輸入時 patch grid 為 32×32 = 1024 tokens）
   - 特徵維度：S=384, B=768, L=1024

2. **輕量映射**：將 DINOv3 patch tokens 映射到 FLUX VAE latent space。
   - spatial：patch grid × 2 = latent grid（32×32 → 64×64，對應 512×512 輸入）
   - channel：384/768/1024 → **16**（VAE 的 latent_channels）
   - 特徵降到 16 channels 應該綽綽有餘

3. **VAE 解碼**：使用 FLUX VAE 的 `AutoencoderKL` decoder，解碼公式：`x = vae.decode((latents / 0.3611) + 0.1159)`。

4. **訓練數據**：tagged-anime-illustrations（337K 張 512×512 JPG），無需額外標註。

### 排除項目

- `image quality filter (black border detection)` — 不實作
- `transform augmentation behavior` 的額外測試 — 不實作
- **Diffusion / Flow Model** — 整體排除

## Follow-up

- **✅ VAE roundtrip 已驗證**：`[1, 3, 512, 512]` → encode → `[1, 16, 64, 64]` → decode → `[1, 3, 512, 512]` 正常
- **⏳ DINOv3 gated access 申請中**：`facebook/dinov3-vits16-pretrain-lvd1689m`
- **✅ DINOv3 訪問已通過！**
- 完整 PoC（DINOv3 → 隨機投影 → VAE decode）等 DINOv3 通過後執行
- 選擇 DINOv3 模型（S/B/L 取決於特徵維度與計算成本）
- 設計特徵↔Latent 映射網路的架構（patch grid → latent grid spatial 2x upsampling + channel 降維 384→16）
- 評估硬體的 VRAM 是否足以執行這個 pipeline
- 效能評估：完整 pipeline FPS 估算（DINOv3 + 映射 + VAE decode vs Diffusion pipeline）

## References

- [01 DINOv3 模型規格](./01_2026_06_02_agent_dinov3-model-specifications.md)
- [02 AnimePro FLUX VAE 設定檔分析](./02_2026_06_02_agent_animepro-flux-vae-analysis.md)
- [03 Tagged Anime Illustrations dataset lookup](./03_2026_06_03_agent_tagged-anime-illustrations-dataset-lookup.md)
- [DINOv3 official](https://github.com/facebookresearch/dinov3)
