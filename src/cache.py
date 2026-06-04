"""Pre-encode cache pipeline: shard storage, LRU cache, and shard-aware sampler.

Provides a two-stage training architecture:

    Phase 0: Pre-encode (run once)
        Images → DINOv3 + VAE.encode → shard cache on disk

    Phase 1: Training (use cache)
        CachedDataset + ShardAwareSampler → fast mini-batch training

Shard storage format:

    cache/
    ├── manifest.json              # version, hash, model versions
    ├── shard_00000/
    │   ├── patch_tokens.pt        # [N, 1024, 384] float16
    │   ├── gt_latents.pt          # [N, 16, 64, 64] float16
    │   └── meta.json              # image paths, shard info
    └── ...

LRU shard cache for fast sequential access during training:

    CachedDataset.__getitem__(idx)
        → O(1) index_map lookup → {shard_id, offset}
        → ShardCache.get(shard_id) → cache hit → slice & return
        → cache miss → load shard from HDD → LRU evict oldest → slice & return

Shard-aware shuffle for high cache hit rate:

    ShardAwareSampler
        Level 1: shuffle shard order
        Level 2: shuffle within each shard
        → sequential shard access → LRU hit rate ~97%

Usage:

    # Training
    config = ShardCacheConfig(cache_dir=Path("~/hdd/cache"))
    dataset = CachedDataset(config)
    sampler = ShardAwareSampler(dataset, seed=42)
    loader = DataLoader(dataset, batch_size=8, sampler=sampler)

From [16_2026_06_04_cache-architecture-design](./../chat_with_my_agent/16_2026_06_04_agent_cache-architecture-design.md).
"""

from __future__ import annotations

import json
import random
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, Sampler

# ── Data classes ────────────────────────────────────────────────────


@dataclass
class ShardCacheConfig:
    """Configuration for the pre-encode cache pipeline."""

    shard_size: int = 1000  # images per shard
    cache_dir: Path = field(default_factory=lambda: Path.home() / "hdd" / "cache")
    max_cached_shards: int = 8  # number of shards to keep in RAM
    dtype: torch.dtype = torch.float16  # storage precision for tensors


@dataclass
class ShardMeta:
    """Metadata for a single shard, stored in meta.json."""

    shard_id: int
    n_images: int
    image_paths: list[str] = field(default_factory=list)  # for debugging


@dataclass
class CacheManifest:
    """Global manifest tracking cache version and consistency."""

    version: int = 1
    dataset_root: str = ""
    dataset_hash: str = ""
    model_versions: dict[str, str] = field(default_factory=dict)
    total_shards: int = 0
    shard_count: int = 0
    created_at: str = ""
    completed_shards: list[int] = field(default_factory=list)  # for resume


# ── Manifest helpers ────────────────────────────────────────────────


def _manifest_path(cache_dir: Path) -> Path:
    return cache_dir / "manifest.json"


def save_manifest(config: ShardCacheConfig, manifest: CacheManifest) -> None:
    """Write manifest.json to cache_dir."""
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    data = asdict(manifest)
    data["created_at"] = datetime.now(timezone.utc).isoformat()
    with open(_manifest_path(config.cache_dir), "w") as f:
        json.dump(data, f, indent=2)


def load_manifest(cache_dir: Path) -> CacheManifest:
    """Read manifest.json and return CacheManifest."""
    path = _manifest_path(cache_dir)
    if not path.exists():
        raise FileNotFoundError(f"Cache manifest not found: {path}")
    data = json.loads(path.read_text())
    return CacheManifest(
        **{k: v for k, v in data.items() if k in CacheManifest.__dataclass_fields__}
    )


def compute_dataset_hash(paths: list[Path]) -> str:
    """Compute a SHA-256 hash of all image paths (sorted, for stability)."""
    import hashlib

    sorted_paths = sorted(p.resolve().name + str(p.parent.name) for p in paths)
    content = "\n".join(sorted_paths)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def validate_cache(
    config: ShardCacheConfig,
    paths: list[Path],
    dino_model: str,
    vae_model: str,
) -> bool:
    """Check if existing cache is consistent with current dataset/models.

    Returns True if cache is valid and usable.
    Returns False if cache is stale and needs re-encoding.
    """
    try:
        manifest = load_manifest(config.cache_dir)
    except FileNotFoundError:
        return False

    current_hash = compute_dataset_hash(paths)
    if manifest.dataset_hash != current_hash:
        return False

    if manifest.model_versions.get("dino") != dino_model:
        return False

    if manifest.model_versions.get("vae") != vae_model:
        return False

    return True


# ── LRU Shard Cache ────────────────────────────────────────────────


class ShardCache:
    """LRU cache for shard tensors using OrderedDict.

    Designed for per-Dataset-instance use (one per DataLoader worker).
    Automatically evicts the least-recently-used shard when capacity is exceeded.

    Usage:
        cache = ShardCache(max_shards=8)
        cache.put(0, tokens, latents)
        result = cache.get(0)  # → (tokens, latents) or None
    """

    def __init__(self, max_shards: int = 8):
        self._cache: OrderedDict[int, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self.max_shards = max_shards
        self.hits = 0
        self.misses = 0

    def get(self, shard_id: int) -> tuple[torch.Tensor, torch.Tensor] | None:  # noqa: ANN201
        """Get cached shard tensors, or None on miss.

        On hit, moves entry to end (most recently used).
        """
        if shard_id in self._cache:
            self._cache.move_to_end(shard_id)
            self.hits += 1
            return self._cache[shard_id]
        self.misses += 1
        return None

    def put(self, shard_id: int, tokens: torch.Tensor, latents: torch.Tensor) -> None:
        """Insert shard into cache, evicting LRU if at capacity."""
        if shard_id in self._cache:
            self._cache.move_to_end(shard_id)

        self._cache[shard_id] = (tokens, latents)

        # Evict LRU entries (at the front)
        while len(self._cache) > self.max_shards:
            evict_id, (evict_tokens, evict_latents) = self._cache.popitem(last=False)
            del evict_tokens, evict_latents

    @property
    def size(self) -> int:
        return len(self._cache)

    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "size": self.size,
            "max": self.max_shards,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{self.hits / max(total, 1) * 100:.1f}%",
        }


# ── CachedDataset ──────────────────────────────────────────────────


class CachedDataset(Dataset[Any]):  # noqa: D101
    """PyTorch Dataset backed by pre-encoded shard files.

    Index mapping: {global_idx → (shard_id, local_offset)}
    Lookup: O(1) dict lookup → LRU cache → tensor slice

    Shuffle is handled by ShardAwareSampler, so __getitem__ is purely
    O(1) lookup + cache check.

    Usage:
        config = ShardCacheConfig(cache_dir=Path("cache"))
        dataset = CachedDataset(config)
        sampler = ShardAwareSampler(dataset, shard_size=1000, seed=42)
        loader = DataLoader(dataset, batch_size=8, sampler=sampler)
    """

    def __init__(self, config: ShardCacheConfig):
        self.config = config
        self.shard_size = config.shard_size
        self._cache = ShardCache(config.max_cached_shards)
        self._build_index_map()

    def _build_index_map(self) -> None:
        """Build {global_idx → (shard_id, offset)} lookup table.

        Reads shard directories sorted by name to infer shard_id.
        Does NOT shuffle — shuffle is handled by the Sampler.
        """
        cache_dir = self.config.cache_dir
        manifest = load_manifest(cache_dir)
        self.shard_count = manifest.total_shards

        self._index_map: dict[int, tuple[int, int]] = {}
        global_idx = 0

        for shard_id in range(self.shard_count):
            shard_dir = cache_dir / f"shard_{shard_id:05d}"
            meta_path = shard_dir / "meta.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                n = meta["n_images"]
                for offset in range(n):
                    self._index_map[global_idx] = (shard_id, offset)
                    global_idx += 1

    def __len__(self) -> int:
        return len(self._index_map)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: D102
        """Get a single sample (patch_tokens, gt_latents).

        Checks LRU cache first, then loads from disk on miss.
        Returns float32 tensors for training.
        """
        shard_id, offset = self._index_map[idx]

        # Cache hit
        cached = self._cache.get(shard_id)
        if cached is not None:
            tokens, latents = cached
            tokens = tokens[offset : offset + 1].to(torch.float32)
            latents = latents[offset : offset + 1].to(torch.float32)
            return tokens, latents

        # Cache miss — load entire shard from HDD
        shard_dir = self.config.cache_dir / f"shard_{shard_id:05d}"
        tokens_path = shard_dir / "patch_tokens.pt"
        latents_path = shard_dir / "gt_latents.pt"

        tokens = torch.load(tokens_path, weights_only=True).to(torch.float32)
        latents = torch.load(latents_path, weights_only=True).to(torch.float32)

        # Cache the whole shard for subsequent samples
        self._cache.put(shard_id, tokens, latents)

        # Return single sample
        tokens = tokens[offset : offset + 1]
        latents = latents[offset : offset + 1]
        return tokens, latents

    @property
    def cache(self) -> ShardCache:
        return self._cache

    def shard_stats(self) -> dict[str, Any]:
        return self._cache.stats()

    def info(self) -> dict[str, Any]:
        return {
            "total_samples": len(self),
            "shard_count": self.shard_count,
            "shard_size": self.shard_size,
            "cache": self.shard_stats(),
        }


# ── ShardAwareSampler ──────────────────────────────────────────────


class ShardAwareSampler(Sampler[int]):  # noqa: D101
    """Two-level shuffle sampler for high LRU cache hit rate.

    Level 1: shuffle shard order
    Level 2: shuffle within each shard (deterministic per epoch)

    With 288 shards and LRU(8), random shuffle gives hit rate ~2.8%,
    while shard-aware shuffle gives ~97.2% (8/288 = 1 miss per 125 batches).

    Usage:
        sampler = ShardAwareSampler(dataset, shard_size=1000, seed=42)
        loader = DataLoader(dataset, batch_size=8, sampler=sampler)
        # ⚠️ DO NOT set shuffle=True when using this sampler
    """

    def __init__(
        self,
        dataset: CachedDataset,
        shard_size: int = 1000,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.shard_size = shard_size
        self.seed = seed

    def __iter__(self) -> Iterator[int]:  # noqa: D102
        rng = random.Random(self.seed)

        # Level 1: shuffle shard order
        shard_ids = list(range(self.dataset.shard_count))
        rng.shuffle(shard_ids)

        # Level 2: yield samples from each shard (shuffled within)
        for shard_id in shard_ids:
            shard_rng = random.Random(self.seed + shard_id)
            offsets = list(range(self.shard_size))
            shard_rng.shuffle(offsets)

            for offset in offsets:
                global_idx = shard_id * self.shard_size + offset
                if global_idx in self.dataset._index_map:
                    yield global_idx

    def __len__(self) -> int:
        return len(self.dataset)
