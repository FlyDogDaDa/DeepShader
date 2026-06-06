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
    │   ├── gt_latents.pt          # [N, 16, 64, 64] float16 — mean (μ_gt)
    │   ├── gt_logvar.pt           # [N, 16, 64, 64] float16 — log-variance (log σ²_gt)
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
from tqdm import tqdm

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

    Cached data: (tokens, latents, logvars) where logvars may be None
    for backward compatibility with caches that don't have gt_logvar.pt.

    Usage:
        cache = ShardCache(max_shards=8)
        cache.put(0, tokens, latents, logvars)
        result = cache.get(0)  # → (tokens, latents, logvars) or None
    """

    def __init__(self, max_shards: int = 8):
        self._cache: OrderedDict[
            int, tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]
        ] = OrderedDict()
        self.max_shards = max_shards
        self.hits = 0
        self.misses = 0

    def get(
        self, shard_id: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None:  # noqa: ANN201
        """Get cached shard tensors, or None on miss.

        On hit, moves entry to end (most recently used).
        Returns (tokens, latents, logvars) where logvars may be None
        for backward compatibility with old caches.
        """
        if shard_id in self._cache:
            self._cache.move_to_end(shard_id)
            self.hits += 1
            return self._cache[shard_id]
        self.misses += 1
        return None

    def put(
        self,
        shard_id: int,
        tokens: torch.Tensor,
        latents: torch.Tensor,
        logvars: torch.Tensor | None = None,
    ) -> None:
        """Insert shard into cache, evicting LRU if at capacity."""
        if shard_id in self._cache:
            self._cache.move_to_end(shard_id)

        self._cache[shard_id] = (tokens, latents, logvars)

        # Evict LRU entries (at the front)
        while len(self._cache) > self.max_shards:
            evict_id, (evict_tokens, evict_latents, evict_logvars) = (
                self._cache.popitem(last=False)
            )
            del evict_tokens, evict_latents, evict_logvars

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

    Returns (patch_tokens, gt_mean, gt_logvar) triple.
    If gt_logvar.pt doesn't exist (old cache), returns zeros for logvar.

    Usage:
        config = ShardCacheConfig(cache_dir=Path("cache"))
        dataset = CachedDataset(config)
        sampler = ShardAwareSampler(dataset, shard_size=1000, seed=42)
        loader = DataLoader(dataset, batch_size=8, sampler=sampler)
    """

    def __init__(
        self,
        config: ShardCacheConfig,
        subset_indices: set[int] | None = None,
    ):
        self.config = config
        self.shard_size = config.shard_size
        self._cache = ShardCache(config.max_cached_shards)
        self._build_index_map(subset_indices)

    def _build_index_map(self, subset_indices: set[int] | None = None) -> None:
        """Build {global_idx → (shard_id, local_offset)} lookup table.

        When ``subset_indices`` is provided, only builds entries for those
        specific indices — useful for debug mode to skip reading all 338
        shard metadata files.
        """
        cache_dir = self.config.cache_dir
        manifest = load_manifest(cache_dir)
        self.shard_count = manifest.total_shards

        self._index_map: dict[int, tuple[int, int]] = {}
        global_idx = 0

        if subset_indices is not None:
            # Fast path: only build index entries for requested indices.
            # Skip reading shard metadata files that can't contain needed
            # indices, cutting 338 metadata reads to ~1.
            needed = set(subset_indices)
            for shard_id in range(self.shard_count):
                if not needed:
                    break
                shard_dir = cache_dir / f"shard_{shard_id:05d}"
                meta_path = shard_dir / "meta.json"
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text())
                    n = meta["n_images"]
                    shard_end = global_idx + n
                    # Only read shard if needed indices fall in its range
                    if needed & set(range(global_idx, shard_end)):
                        for offset in range(n):
                            idx = global_idx + offset
                            if idx in needed:
                                self._index_map[idx] = (shard_id, offset)
                                needed.discard(idx)
                            if not needed:
                                break
        else:
            # Normal path: build full index map
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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # noqa: D102
        """Get a single sample (patch_tokens, gt_mean, gt_logvar).

        Checks LRU cache first, then loads from disk on miss.
        Returns float32 tensors for training.

        If gt_logvar.pt doesn't exist (backward compat with old cache),
        returns zeros for the logvar tensor.
        """
        shard_id, offset = self._index_map[idx]

        # Cache hit
        cached = self._cache.get(shard_id)
        if cached is not None:
            tokens, latents, logvars = cached
            tokens = tokens[offset : offset + 1].to(torch.float32)
            latents = latents[offset : offset + 1].to(torch.float32)
            if logvars is not None:
                logvars = logvars[offset : offset + 1].to(torch.float32)
            else:
                # Backward compat: old cache without logvar → assume unit variance
                logvars = torch.zeros(latents.shape, dtype=torch.float32)
            return tokens, latents, logvars

        # Cache miss — load entire shard from HDD
        shard_dir = self.config.cache_dir / f"shard_{shard_id:05d}"
        tokens_path = shard_dir / "patch_tokens.pt"
        latents_path = shard_dir / "gt_latents.pt"
        logvars_path = shard_dir / "gt_logvar.pt"

        tokens = torch.load(tokens_path, weights_only=True).to(torch.float32)
        latents = torch.load(latents_path, weights_only=True).to(torch.float32)

        # Backward compat: logvar file may not exist
        logvars: torch.Tensor | None
        if logvars_path.exists():
            logvars = torch.load(logvars_path, weights_only=True).to(torch.float32)
        else:
            logvars = None  # old cache, will be filled with zeros in trainer

        # Cache the whole shard for subsequent samples
        self._cache.put(shard_id, tokens, latents, logvars)

        # Return single sample
        tokens = tokens[offset : offset + 1]
        latents = latents[offset : offset + 1]
        if logvars is not None:
            logvars = logvars[offset : offset + 1]
        else:
            logvars = torch.zeros(latents.shape, dtype=torch.float32)
        return tokens, latents, logvars

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


# ── InMemoryDataset ────────────────────────────────────────────────


class InMemoryDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Fully in-memory dataset for when all shards fit in RAM.

    Specify ``num_shards`` to load that many shards (starting from shard 0)
    into RAM at construction time.  All shard tensors are concatenated into
    three large tensors keyed by a flat ``_index_map`` so ``__getitem__``
    becomes a simple slice — no LRU cache, no shard-aware sampler needed.

    Benefits:
        * ``__getitem__`` is O(1) tensor indexing (no dict lookups, no disk I/O)
        * Standard ``DataLoader(shuffle=True)`` works perfectly — no ``ShardAwareSampler``
        * Complex samplers can be dropped entirely

    Usage::

        config = ShardCacheConfig(cache_dir=Path("~/hdd/cache"))
        dataset = InMemoryDataset(config, num_shards=32)  # ~32 K samples, ~50 GB
        loader = DataLoader(dataset, batch_size=32, shuffle=True, pin_memory=True)

    """

    def __init__(
        self,
        config: ShardCacheConfig,
        num_shards: int,
    ) -> None:
        self.config = config
        self.num_shards = num_shards

        # Load manifest to discover shard sizes
        cache_dir = config.cache_dir
        manifest = load_manifest(cache_dir)
        self.shard_count = manifest.total_shards

        # Phase 1: read metadata, build flat index map, track shard counts & logvar presence
        #   _index_map[i] = (shard_id, offset_within_shard)
        self._index_map: list[tuple[int, int]] = []
        shard_info: list[dict[str, Any]] = []  # {"n": int, "has_logvar": bool}
        total_samples = 0
        for shard_id in tqdm(range(num_shards), desc="Building index map"):
            shard_dir = cache_dir / f"shard_{shard_id:05d}"
            meta_path = shard_dir / "meta.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                n = meta["n_images"]
                has_logvar = (shard_dir / "gt_logvar.pt").exists()
                shard_info.append({"n": n, "has_logvar": has_logvar})
                for offset in range(n):
                    self._index_map.append((shard_id, offset))
                total_samples += n
            else:
                raise FileNotFoundError(
                    f"Shard {shard_id} not found at {shard_dir} "
                    f"(manifest says {manifest.total_shards} shards)"
                )

        self._total_samples = total_samples

        # Phase 2: load all shard tensors into RAM
        # Keep original dtype (float16) to minimize RAM usage.
        # Conversion to float32 happens in the collate function per batch.
        # Pre-allocate tensors then fill in-place to avoid torch.cat doubling memory.

        # Load first shard to get shape templates (released immediately after)
        first_shard_dir = cache_dir / "shard_00000"
        first_tokens = torch.load(
            first_shard_dir / "patch_tokens.pt", weights_only=True
        )
        first_latents = torch.load(first_shard_dir / "gt_latents.pt", weights_only=True)
        tokens_shape = first_tokens.shape
        latents_shape = first_latents.shape
        del first_tokens, first_latents

        # Pre-allocate tensors (float16 — saves 50% RAM vs float32)
        self._tokens: torch.Tensor = torch.empty(
            (total_samples, *tokens_shape[1:]), dtype=torch.float16, device="cpu"
        )
        self._latents: torch.Tensor = torch.empty(
            (total_samples, *latents_shape[1:]), dtype=torch.float16, device="cpu"
        )

        # Pre-allocate logvars with total_samples to keep indexing uniform.
        # Missing logvar shards get zero-filled (negligible memory: ~32KB).
        self._logvars: torch.Tensor = torch.zeros(
            (total_samples, *latents_shape[1:]), dtype=torch.float16, device="cpu"
        )

        # Fill in-place: load one shard, copy to pre-allocated tensor, delete shard.
        # Peak memory = final_tensor + one_shard (NOT 2x like torch.cat).
        idx_offset = 0

        for shard_id in tqdm(range(num_shards), desc="Loading shard tensors"):
            shard_dir = cache_dir / f"shard_{shard_id:05d}"
            n = shard_info[shard_id]["n"]

            tokens = torch.load(
                shard_dir / "patch_tokens.pt", weights_only=True
            )  # float16
            latents = torch.load(
                shard_dir / "gt_latents.pt", weights_only=True
            )  # float16

            # In-place copy into pre-allocated buffers
            self._tokens[idx_offset : idx_offset + n] = tokens
            self._latents[idx_offset : idx_offset + n] = latents

            # Handle logvars
            logvars_path = shard_dir / "gt_logvar.pt"
            if logvars_path.exists():
                logvars = torch.load(logvars_path, weights_only=True)  # float16
                self._logvars[idx_offset : idx_offset + n] = logvars
                del logvars

            # Delete shard tensors immediately to free memory
            del tokens
            del latents

            idx_offset += n

    def __len__(self) -> int:
        return self._total_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (patch_tokens, gt_mean, gt_logvar) for a single sample."""
        tokens = self._tokens[idx : idx + 1]  # [1, ...]
        latents = self._latents[idx : idx + 1]
        logvars = self._logvars[idx : idx + 1]
        return tokens, latents, logvars

    def info(self) -> dict[str, Any]:
        """Return dataset metadata (mode, samples, shard count, tensor shapes)."""
        return {
            "mode": "in_memory",
            "total_samples": self._total_samples,
            "num_shards": self.num_shards,
            "tokens_shape": list(self._tokens.shape),
            "latents_shape": list(self._latents.shape),
        }


# ── ShardAwareSampler ──────────────────────────────────────────────


class ShardAwareSampler(Sampler[int]):  # noqa: D101
    """Two-level shuffle sampler for high LRU cache hit rate.

    Level 1: shuffle shard order
    Level 2: shuffle within each shard (deterministic per epoch)

    With 288 shards and LRU(8), random shuffle gives hit rate ~2.8%,
    while shard-aware shuffle gives ~97.2% (8/288 = 1 miss per 125 batches).

    Supports optional train/val split via ``train_ratio``.
    When split is used, both train and val iterators inherit the same
    shard-aware access pattern — only the sample assignment differs.

    Usage:
        sampler = ShardAwareSampler(dataset, shard_size=1000, seed=42)
        loader = DataLoader(dataset, batch_size=8, sampler=sampler)
        # ── or with train/val split ──
        train_sampler = ShardAwareSampler(
            dataset, train_ratio=0.9, seed=42, is_train=True,
        )
        val_sampler = ShardAwareSampler(
            dataset, train_ratio=0.9, seed=42, is_train=False,
        )
        # ⚠️ DO NOT set shuffle=True when using this sampler
    """

    def __init__(
        self,
        dataset: CachedDataset,
        shard_size: int = 1000,
        seed: int = 42,
        train_ratio: float | None = None,
        is_train: bool = True,
    ):
        self.dataset = dataset
        self.shard_size = shard_size
        self.seed = seed
        self.train_ratio = train_ratio
        self.is_train = is_train

        # Pre-compute the shuffled sample list so train/val split
        # preserves the shard-aware sequential access pattern.
        rng = random.Random(self.seed)
        shard_ids = list(range(self.dataset.shard_count))
        rng.shuffle(shard_ids)

        self._ordered_indices: list[int] = []
        for shard_id in shard_ids:
            shard_rng = random.Random(self.seed + shard_id)
            offsets = list(range(self.shard_size))
            shard_rng.shuffle(offsets)
            for offset in offsets:
                global_idx = shard_id * self.shard_size + offset
                if global_idx in self.dataset._index_map:
                    self._ordered_indices.append(global_idx)

        # ── Optional train/val split ─────────────────────────
        # Split AFTER the shard-aware ordering so both train and val
        # inherit the same sequential shard-access pattern.
        if self.train_ratio is not None:
            n_train = int(len(self._ordered_indices) * self.train_ratio)
            if self.is_train:
                self._indices = self._ordered_indices[:n_train]
            else:
                self._indices = self._ordered_indices[n_train:]
        else:
            self._indices = self._ordered_indices

    def __iter__(self) -> Iterator[int]:  # noqa: D102
        yield from self._indices

    def __len__(self) -> int:
        return len(self._indices)
