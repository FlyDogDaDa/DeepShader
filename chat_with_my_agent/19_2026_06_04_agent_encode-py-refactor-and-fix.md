---
created: 2026-06-04
author: agent
type: agent
status: final
tags: [daily-log, encode-py, memory-leak, buffer-refactor, tqdm, batch-size-param]
---

# `encode.py` Buffer Refactor + Memory Leak Fix + Progress Display

## What

1. **Buffer 重構為 Tensor-based** — `BatchShardEncoder` 的 `_buffer_tokens` / `_buffer_latents` 從 `list[torch.Tensor]` 改為單一 `torch.Tensor | None`，用 `torch.cat` 追加 + slice 分割，取代 `tolist()` + `torch.tensor()` 的無效轉換。
2. **修復記憶體持續上升 bug** — `feed()` 內將 GPU tensor 先 `.to("cpu")` 存入 buffer（buffer 現在是純 RAM），並在 `main()` 的 loop 內 `del` + `torch.cuda.empty_cache()` 釋放 GPU references。
3. **加入 `--batch-size` CLI 參數**（default=16）— 取代硬編碼 `batch_size=2`，讓使用者可依 VRAM 調整。
4. **加入 `tqdm` 進度條** — 取代每 10 shard 的 print，提供批次級別的進度條 + postfix（shard #、已處理圖片數、每 shard 耗時）。
5. **修復 `--resume` 忽略 missing manifest bug** — manifest 遺失時原來會跳過全部（`start_shard = shard_count`），改為從 0 開始。
6. **修復 `pbar.total` 錯誤被 `len()` 包裹** — `pbar.total` 已經是 `int`，不需要 `len()`。
7. **清理測試產生的 shard_00000~00005** — 這些是用 `--shard-size 10` 測試建立的，格式不兼容正式 1000。
8. **修復一個 type warning 在 test** — `test_validate_cache_valid` 使用硬編碼 hash `"abc"` 導致 `validate_cache` 計算出的 hash 不匹配，改為使用 `compute_dataset_hash()` 的結果。

## Why

### 記憶體問題

之前 buffer 用 Python list 儲存 tensor，每次 `feed()` 直接 `extend()` 放入 CUDA tensor，導致 buffer 本身就在 GPU 上累積。加上 loop 內沒有 `del` 原始輸出，VRAM 持續上升直到 OOM。

**Buffer 重構理由：**
- `list[torch.Tensor]` + 每次 `torch.tensor(t, dtype=...)` 轉換 — 非常低效，要經過 Python-list → float → tensor 的二次拷貝
- 單一 tensor 做 buffer：`torch.cat` 追加 + slice 切 shard — O(1) 的記憶體管理

**記憶體釋放理由：**
- `feed()` 內 `.to("cpu")` 確保 buffer 只佔 RAM
- `del patch_tokens / gt_latents` 釋放 GPU tensor references
- `torch.cuda.empty_cache()` 釋放 PyTorch 的 unallocated cache

### 為什麼要用 tqdm 取代 print

- `print` 無法動態更新（每 10 shard 才顯示一次，中間 125 batches 完全沒回饋）
- `tqdm` 提供平滑進度條 + 批次進度 + postfix metrics，長時運行更安心

### `--batch-size` 的必要性

VRAM 是 GPU 專屬限制（~12GB），使用者可以依自己硬體調整：
- batch=2：最安全，幾乎不會爆 VRAM
- batch=16：在 12GB GPU 上剛好的平衡點
- 更大 / 更小可自由調

## How

### `src/encode.py` 主要修改

**Buffer 結構重構：**

```python
# Before (list, GPU-leaking)
self._buffer_tokens: list[torch.Tensor] = []
# feed(): self._buffer_tokens.extend([tokens])  # tokens still on GPU

# After (single tensor, CPU-bound buffer)
self._buffer_tokens: torch.Tensor | None = None
# feed():
#   tokens = tokens.to("cpu", non_blocking=True)  # move to RAM
#   self._buffer_tokens = torch.cat([buffer, tokens], dim=0)  # CPU cat
# _flush_shard(): tokens = buffer[:shard_size].to(torch.float16)  # slice + fp16
#   buffer = buffer[shard_size:]  # remainder stays in buffer
```

**記憶體管理：**

```python
encoder.feed(patch_tokens, gt_latents, batch_paths_str)

# Drop GPU references immediately after feeding to CPU buffer
del patch_tokens
del gt_latents
torch.cuda.empty_cache()
```

**tqdm 進度條：**

```python
pbar = tqdm(range(batch_count), desc="Encoding", unit="batches")
for batch_idx in pbar:
    # ... encode ...
    pbar.set_postfix({
        "shards": shard_num,
        "img": shard_num * args.shard_size,
        "rate": f"{elapsed / max(shard_num, 1):.1f}s/shard",
    })
```

**CLI 新增 `--batch-size`：**

```bash
uv run python -m src.encode --dataset ... --cache-dir ... --batch-size 16 --device cuda
uv run python -m src.encode --dataset ... --cache-dir ... --batch-size 16 --resume --device cuda
```

**Manifest missing fix：**

```python
# Before: if manifest missing and resume not set, start_shard = shard_count (skip all)
# After: if manifest missing, start_shard = 0 (start fresh)
```

### `tests/test_cache.py` 修復

```python
# Before: manifest hash hardcoded as "abc"
manifest = CacheManifest(dataset_hash="abc", ...)
result = validate_cache(..., "dino-x", "vae-y")  # returns False (hash mismatch)

# After: use actual computed hash
actual_hash = compute_dataset_hash(paths)
manifest = CacheManifest(dataset_hash=actual_hash, ...)
result = validate_cache(..., paths, "dino-x", "vae-y")  # returns True
```

### `trainer.py` 修復

**Cached epoch helpers 移除錯誤的 `.squeeze(1)`：**

```python
# Before: squeeze(1) turns [B, 1024, 384] into [B, 384] — wrong!
patch_tokens = patch_tokens.squeeze(1).to(device)

# After: use as-is (already [B, 1024, 384] from _collate_fn)
patch_tokens = patch_tokens.to(device)
```

### 測試結果

- `pytest tests/` — 74 tests all passing (1 test fixed)

## Follow-up

- [ ] 明天檢查 pre-encode 進度（背景跑了 `batch=16`，21065 batches × 1.3s/batch ≈ 4.5 小時）
- [ ] pre-encode 完成後測試 cached training：`python -m train --cache-dir ~/hdd/cache --debug`
- [ ] 測試 image mode backward compat
- [ ] 清理 type warnings（optional, doesn't affect functionality）

## References

- [src/encode.py](../../src/encode.py) — pre-encode pipeline
- [src/trainer.py](../../src/trainer.py) — training loop
- [src/__init__.py](../../src/__init__.py) — public API
- [tests/test_cache.py](../../tests/test_cache.py) — cache tests
- [18_2026_06_04_agent_trainer-and-train-py-double-mode](./18_2026_06_04_agent_trainer-and-train-py-double-mode.md) — previous entry
