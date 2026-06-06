---
created: 2026-06-06
author: Vincent
type: agent
status: draft
tags: [encode, resume, cache, manifest, bug-diagnosis]
---

# encode.py `--resume` Bug Diagnosis

## What

Diagnosed why `--resume` in `src/encode.py` fails with "Cache is INVALID (dataset/model changed)" and why `Shards: 32/32` appears instead of `338/338`. Found that `--resume` is effectively broken when the old cache was created with `--subset`.

## Why

User wants to use the idle GPU `cuda:1` to re-encode the full dataset (337,038 images → 338 shards). The training pipeline was already using a subset (32,000 images, 32 shards) encoded with `--subset 0,32000`. Running `--resume` to continue from the old cache triggers a dataset hash mismatch.

## How

### Root Cause Analysis

**Symptom 1**: `Shards: 32/32` instead of `338/338`

Line 337 of `src/encode.py` prints the old manifest's numbers:

```python
print(f"[encode]   Shards: {manifest.shard_count}/{manifest.total_shards}")
```

The old manifest (from the `--subset 0,32000` run) recorded `shard_count=32, total_shards=32`.

**Symptom 2**: `Cache is INVALID (dataset/model changed)`

Line 345 calls `validate_cache()`, which recomputes the dataset hash from the **current** image paths (full dataset) and compares it with the manifest's stored hash (computed from `paths[0:32000]`). They differ → returns `False` → exits.

### Why Resume Fails

The resume logic at line 361:

```python
start_shard = max(completed) + 1 if completed else shard_count
```

This mathematically works: `start_shard = 32`, `start_idx = 32000`, `remaining_paths = paths[32000:]`. But the `validate_cache()` check (lines 345–354) runs **before** this and rejects the cache because the dataset hash changed (full dataset ≠ subset dataset).

### Manifest Overwrite Bug

Lines 473–490 write the manifest at completion:

```python
manifest = CacheManifest(
    ...
    total_shards=shard_count,        # current computed value
    shard_count=shard_count,         # same (not actual completed count!)
    completed_shards=list(range(shard_count)),  # assumes ALL done!
)
```

If the process is interrupted mid-flight (OOM, Ctrl+C, crash), the manifest is **not** updated. But if the process completes, the manifest is always overwritten with "all done" — there's no incremental tracking of completed shards. This means:

1. A completed `--subset 0,32000` run produces `shard_count=32, total_shards=32, completed_shards=[0..31]`
2. The dataset hash is stored from that subset run
3. A subsequent full-dataset `--resume` fails the hash check

**Note**: During training, the cache is working fine because training uses `CachedDataset` which reads shard files directly (not via `encode.py` resume logic). So training is unaffected — this bug only affects the pre-encode resume.

### Potential Fix (deferred)

Options:
1. **Store `total_images` in manifest** and compare image count instead of hash (subset runs would have `total_images < current_total_images`, triggering full re-encode with warning)
2. **Incremental manifest updates**: write `completed_shards` after each shard flush (not just at completion), and don't overwrite `dataset_hash` if the dataset is the same — but handle subset vs full correctly
3. **Force flag required**: if `shard_count > manifest.total_shards`, require `--force` explicitly instead of `--resume`

### Decision

Deferred. Training is running on the existing 32-shard cache. Will revisit full-dataset re-encode with `--force` (full re-encode) when ready.

## Follow-up

- [ ] When ready for full-dataset encoding: `--force` to do a fresh full re-encode (takes ~8 hours)
- [ ] Implement incremental manifest updates so `--resume` works after crashes
- [ ] Store `total_images` in manifest for subset/full detection

## References

- [src/encode.py](../../src/encode.py) — `main()` lines 330–370 (cache check + resume), lines 473–490 (manifest write)
- [src/cache.py](../../src/cache.py) — `validate_cache()`, `CacheManifest`
- [train.py](../../train.py) — training pipeline (unaffected, uses `CachedDataset` directly)
