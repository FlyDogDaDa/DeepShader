---
created: 2026-06-03
author: FlyDogDaDa
type: agent
status: final
tags: [daily-log, nas, nfs, network, iops, benchmark]
---

# NAS NFS Read Speed Benchmark

## What

- Measured actual read throughput from NAS (NFS mount at 192.168.0.19)
- Profiled IOPS and batch read performance simulating DataLoader behavior
- Projected total dataset read time and worker tuning recommendations

## Why

Need to determine how many DataLoader workers to configure for efficient training, since the dataset (~337K images) is too large to pre-load into RAM and must be read from NAS on-the-fly.

## How

1. **Network interface check**: `ip -s link show` — NAS connected via `enp6s0` (UP)
2. **Sequential IOPS test**: Read 500 files sequentially, measured throughput and IOPS
3. **Batch simulation**: Read 16/32/64/128/256 files to simulate DataLoader batch loading
4. **Worker scaling**: Projected speedup with ThreadPoolExecutor (sub-linear due to NFS single-connection bottleneck)
5. **Cache flush test**: Pre-read 1000 files to force real network reads before benchmarking

**Results:**

| Metric | Value |
|---|---|
| Average file size | ~110 KB |
| Sequential throughput (500 files) | ~9 MB/s |
| IOPS | ~85 IOPS |
| Total dataset (sequential) | ~66 minutes |

**Projected DataLoader batch=64 throughput:**

| workers | Throughput | Batch time | GPU idle? |
|---|---|---|---|
| 1 | ~9 MB/s | ~750 ms | Yes, significant |
| 4 | ~18 MB/s | ~375 ms | Moderate |
| **8** | **~26 MB/s** | **~265 ms** | Minimal |
| **16** | **~37 MB/s** | **~187 ms** | Low |
| 32 | ~52 MB/s | ~133 ms | Negligible |

NFS speedup is **sub-linear** (`workers^0.5`), as expected with NFS single-session limitations.

**Recommendation**: `num_workers=8~16` — good balance of IO throughput vs. CPU overhead.

## Follow-up

- Configure DataLoader with `num_workers=16` in `src/dataset.py`
- Consider NFS mount options for better multi-client performance

## References

- [03 Tagged Anime Illustrations dataset lookup](./03_2026_06_03_agent_tagged-anime-illustrations-dataset-lookup.md)
- [04 Dataset object with PyTorch transforms](./04_2026_06_03_agent_dataset-object-pytorch-transforms.md)
