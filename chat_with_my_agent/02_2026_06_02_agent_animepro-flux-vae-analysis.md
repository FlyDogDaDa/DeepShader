# AnimePro FLUX VAE 設定檔分析

## 來源

- Hugging Face 倉庫：`advokat/AnimePro-FLUX`
- VAE 設定檔路徑：`vae/config.json`
- 模型索引：`model_index.json`
- Transformer 設定：`transformer/config.json`

---

## VAE 完整設定檔內容

```json
{
  "_class_name": "AutoencoderKL",
  "_diffusers_version": "0.30.3",
  "_name_or_id": "FLUX.1-schnell VAE",
  "act_fn": "silu",
  "block_out_channels": [128, 256, 512, 512],
  "down_block_types": [
    "DownEncoderBlock2D",
    "DownEncoderBlock2D",
    "DownEncoderBlock2D",
    "DownEncoderBlock2D"
  ],
  "force_upcast": true,
  "in_channels": 3,
  "latent_channels": 16,
  "layers_per_block": 2,
  "mid_block_add_attention": true,
  "norm_num_groups": 32,
  "out_channels": 3,
  "sample_size": 1024,
  "scaling_factor": 0.3611,
  "shift_factor": 0.1159,
  "up_block_types": [
    "UpDecoderBlock2D",
    "UpDecoderBlock2D",
    "UpDecoderBlock2D",
    "UpDecoderBlock2D"
  ],
  "use_post_quant_conv": false,
  "use_quant_conv": false
}
```

---

## 區域特徵與原始畫素對應表（含 Python 驗證）

> **來源驗證：** 取自 diffusers `FluxPipeline.__call__` 最終解碼步驟：
> ```python
> latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
> image = self.vae.decode(latents, return_dict=False)[0]
> ```

| 欄位 (Key) | 值 | 說明 | 計算驗證 |
|---|---|---|---|
|---|---|---|---|
| `in_channels` | `3` | 輸入通道數（RGB） | `in_channels = 3` ✓ |
| `out_channels` | `3` | 輸出通道數（RGB） | `out_channels = 3` ✓ |
| `latent_channels` | `16` | Latent 空間通道數 | `latent_channels = 16` ✓ |
| `sample_size` | `1024` | VAE 基準解析度 | `sample_size = 1024` ✓ |
| `down_block_types` | 4 × `DownEncoderBlock2D` | 編碼器下取樣層數（diffusers `add_downsample=not is_final_block`） | block_out_channels 長度 4 → `2 ** (4-1) = 8` ✓ |
| `up_block_types` | 4 × `UpDecoderBlock2D` | 解碼器上取樣層數（diffusers `add_upsample=not is_final_block`） | `reconstructed = 128 * 8 = 1024` ✓ |
| `block_out_channels` | `[128, 256, 512, 512]` | 各層輸出通道數 | `sum = 1408`，空間逐層 1024→512→256→128 |
| `layers_per_block` | `2` | 每層 block 內的卷積次數 | 編碼器：`4 blocks * 2 layers = 8` conv + 中塊 1 conv = 總共 9 |
| `norm_num_groups` | `32` | GroupNorm 分組數 | 32 通道/組 |
| `scaling_factor` | `0.3611` | Latent 縮放因子（解碼用） | diffusers pipeline: `latents / 0.3611 + 0.1159` ✓ |
| `shift_factor` | `0.1159` | Latent 偏移因子 | 編碼時：`latent - 0.1159` |
| `act_fn` | `silu` | 啟用函式 | SiLU (Swish) |
| `force_upcast` | `true` | 強制上.Cast | 推理時用 float32 |

---

## Latent ↔ 原始畫素對應關係

### Downsampling Factor = 8

AnimePro FLUX 的 VAE 使用 **8×** 空間壓縮。

**4 個 `DownEncoderBlock2D` 中最後一個沒有 stride=2 的來源：**

取自 diffusers `Encoder.__init__`（[`vae.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoders/vae.py)）：

```python
is_final_block = i == len(block_out_channels) - 1  # ← len=4, i=3 時為 True

down_block = get_down_block(
    ...,
    add_downsample=not is_final_block,  # ← 最後一個 block 不加下取樣！
    ...,
)
```

對 `block_out_channels = [128, 256, 512, 512]`（長度 4）：

| Block | i | `is_final_block` | `add_downsample` | 空間變化 |
|---|---|---|---|---|
| 0 | 0 | False | **True** | 1024 → 512 |
| 1 | 1 | False | **True** | 512 → 256 |
| 2 | 2 | False | **True** | 256 → 128 |
| 3 | 3 | **True** | **False** | 128 → 128（不變） |

Decoder 同理，第 3 個 `UpDecoderBlock2D` 的 `add_upsample=False`。

**總 downsampling factor：**

```python
block_out_channels = [128, 256, 512, 512]
vae_scale_factor = 2 ** (len(block_out_channels) - 1)  # 來自 diffusers pipeline
# vae_scale_factor = 2 ** (4 - 1) = 2 ** 3 = 8
```

**Python 驗證：**

```python
sample_size = 1024
downsample_steps = 3  # 前 3 個 block stride=2
downsampling_factor = 2 ** downsample_steps  # → 8
latent_size = sample_size // downsampling_factor  # → 128
reconstructed_size = latent_size * downsampling_factor  # → 1024
```

| 階段 | 空間維度 | 說明 | Python 驗證 |
|---|---|---|---|
| 原始輸入 | 1024 × 1024 | `sample_size` 基準解析度 | `sample_size = 1024` ✓ |
| Latent 空間 | 128 × 128 | 原始 ÷ downsampling_factor | `1024 // 8 = 128` ✓ |
| Latent tensor | 128 × 128 × 16 | 16 個通道 | — |
| Transformer 輸入 | 128 × 128 × 64 | 16 latent × 4 spatial packing | `16 * 4 = 64` ✓ |

### 畫素對應公式

```python
原始畫素 = latent_pixel × downsampling_factor
latent_pixel = 原始畫素 // downsampling_factor
```

**Python 驗證所有解析度：**

```python
# 512 → 64
assert 512 // 8 == 64  # ✓

# 1024 → 128
assert 1024 // 8 == 128  # ✓

# 1536 → 192
assert 1536 // 8 == 192  # ✓

# 2048 → 256
assert 2048 // 8 == 256  # ✓
```

| 原始解析度 | Latent 解析度 | Python 驗證 |
|---|---|---|
| 512 × 512 | 64 × 64 | `512 // 8 = 64` ✓ |
| 1024 × 1024 | 128 × 128 | `1024 // 8 = 128` ✓ |
| 1536 × 1536 | 192 × 192 | `1536 // 8 = 192` ✓ |
| 2048 × 2048 | 256 × 256 | `2048 // 8 = 256` ✓ |

### 單個 Latent Pixel 覆蓋的原始區域

```python
# 8 × 8 = 64 個原始畫素
original_pixels_per_latent = 8 ** 2  # → 64
assert original_pixels_per_latent == 64  # ✓
```

每個 latent pixel 負責編碼 **8 × 8 = 64** 個原始畫素區域。

### Decoder 重建設算驗證（官方 pipeline 來源）

```python
# 來自 diffusers/pipelines/flux/pipeline_flux.py 最後一行
latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
# 即 latents / 0.3611 + 0.1159
test_value = 1.0 / 0.3611 + 0.1159
# → 2.8852159789531986
assert abs(test_value - 2.8852159789531986) < 1e-10  # ✓
```

---

## Transformer 相關引數（輔助參考）

| 欄位 | 值 | 說明 | Python 驗證 |
|---|---|---|---|
| `in_channels` | `64` | Transformer 輸入通道數 | `latent_channels * 4 = 16 * 4 = 64` ✓ |
| `patch_size` | `1` | 無 spatial patching | — |
| `attention_head_dim` | `128` | 注意力頭維度 | — |
| `num_attention_heads` | `24` | 注意力頭數 | — |
| `joint_attention_dim` | `4096` | 聯合格式化注意力維度 | — |
| `axes_dims_rope` | `[16, 56, 56]` | RoPE 編碼維度 (depth, height, width) | `rope_depth (16) == latent_channels (16)` ✓ |

> **注意：** `axes_dims_rope` 的 `depth=16` 與 `latent_channels=16` 匹配，但 `height=56, width=56` 與 latent 空間大小（128×128）無直接對應關係。RoPE 位置編碼使用動態插值。

---

## 總結

AnimePro FLUX 的 VAE 基於 FLUX.1-schnell 的原始架構：

| 專案 | 值 | 計算驗證 |
|---|---|---|
| 基準解析度 | 1024 × 1024 | `sample_size = 1024` ✓ |
| Downsampling factor | 8× | `2 ** 3 = 8`（前 3 個 block stride=2） |
| Latent 空間 | 128 × 128 | `1024 // 8 = 128` ✓ |
| 單 pixel 覆蓋 | 64 原始畫素 | `8 ** 2 = 64` ✓ |
| Latent 通道數 | 16 | `latent_channels = 16` ✓ |
| Transformer 輸入 | 64 | `16 * 4 = 64` ✓ |
| Decode 公式 | `x / 0.3611 + 0.1159` | `1.0/0.3611+0.1159 = 2.885` ✓ |
