---
created: 2026-06-05
author: agent
type: agent
status: final
tags: [daily-log, pre-encode-complete, cache-pipeline, training-ready]
---

# Pre-encode 完成，338 shards 全部寫入

## What

Pre-encode pipeline 執行完畢，**338 shards**（共 21,065 batches / ~337K images）全部寫入 `~/hdd/cache/`。

| 指標 | 數值 |
|------|------|
| 總 shard 數 | 338 |
| 總耗時 | 28,665 s（≈ 7.96 小時） |
| 平均速度 | 0.7 shards/min |
| Cache 路徑 | `/home/b11223209/hdd/cache/` |

## Why

這是兩階段架構的 Phase 0。只有完成 pre-encode，training 才能進入 cached mode（Phases 1+），避免每次訓練都重複 28,700,000 次 DINOv3 + VAE encode。

## How

**執行的命令：**（在 `DeepShader` 目錄下）

```
uv run python -m src.encode --dataset <dataset_path> --cache-dir ~/hdd/cache --batch-size 16
```

預 encode 完成後輸出：

```
[encode] Encoding complete! 338 shards written.
[encode] Total time: 28665s (0.7 shards/min)
[encode] Manifest saved to /home/b11223209/hdd/cache/manifest.json
```

**時間預估修正：**
- 原始紀錄 [#19](./19_2026_06_04_agent_encode-py-refactor-and-fix.md) 寫 `4.5 小時` — 數學錯誤
- 正確計算：`21065 × 1.3s = 27,384s ≈ 7.6h`
- 實際耗時 7.96h，與正確預估一致（含 model load / shard flush overhead）
- `~/hdd/cache` 預估 530 GB（從 [#17](./17_2026_06_04_agent_cache-pipeline-implementation.md) 設計文件估算）

## Cached Training Debug Test

First test with `num_workers=16`（沿用 image mode 設定）→ **DataLoader worker 被 OOM Killer 幹掉**。

**原因：** `CachedDataset.__getitem__()` 每次要讀取 shard 到 RAM（約 1.84 GB/shard），16 個 worker 同時操作耗盡記憶體。

**修復：** cached mode 的 `CachedDataset` 有內部 LRU cache，不需要 multiprocessing workers。將 `num_workers=0`。

**結果：** 修復後成功通過 batch 0-7，loss 從 17.4 → 11.5（step 1），shape 正確，無 crash。

> 注意：HDD I/O 较慢，debug 模式下約 12 batches / 12s。LRU cache 打滿後會加速。

## Follow-up

- [x] Cached training debug test — OOM 修復（`num_workers=16 → 0`）
- [ ] 完整 cached training（HDD I/O 慢，可能需數小時）
- [ ] 驗證 image mode backward compat
- [ ] 清理 type warnings（optional）

## References

- [15 Pre-encode cache pipeline decision](./15_2026_06_04_human_pre-encode-cache-pipeline.md)
- [17 Cache Pipeline 實作](./17_2026_06_04_agent_cache-pipeline-implementation.md)
- [19 Encode.py refactor](./19_2026_06_04_agent_encode-py-refactor-and-fix.md)
- [src/encode.py](../../src/encode.py)
- [~/hdd/cache/](file:///home/b11223209/hdd/cache/)
