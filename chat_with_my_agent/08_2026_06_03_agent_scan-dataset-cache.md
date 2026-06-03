---
created: 2026-06-03
author: agent
type: agent
status: final
tags: [performance, caching, dataset]
---

# scan_dataset cache

## What

Added disk caching to `scan_dataset` to avoid repeated directory scans.

## Why

Scanning large Danbooru dataset directories with `rglob` is slow. Cache avoids redundant filesystem traversal when the directory hasn't changed.

## How

- `src/utility.py`: `scan_dataset(root, pattern)` → `list[Path]` via `rglob()`
- Cache stored in `~/.cache/deepshader/<dirname>.json` as `{mtime, paths}`
- Cache invalidated when root directory `st_mtime` changes

## Follow-up

- N/A

## References

- [src/utility.py](../src/utility.py)
- [main.py](../main.py)
