"""Tests for src/cache.py: ShardCache, CachedDataset, ShardAwareSampler."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from src.cache import (
    CachedDataset,
    CacheManifest,
    ShardAwareSampler,
    ShardCache,
    ShardCacheConfig,
    ShardMeta,
    compute_dataset_hash,
    load_manifest,
    save_manifest,
    validate_cache,
)

# ── ShardCache tests ────────────────────────────────────────────────


def test_shard_cache_basic_put_get():
    """Basic put + get works."""
    cache = ShardCache(max_shards=2)
    tokens = torch.randn(10, 5)
    latents = torch.randn(10, 3)
    cache.put(0, tokens, latents)
    result = cache.get(0)
    assert result is not None
    assert torch.equal(result[0], tokens)
    assert torch.equal(result[1], latents)


def test_shard_cache_miss():
    """get returns None for uncached shard."""
    cache = ShardCache(max_shards=2)
    result = cache.get(999)
    assert result is None


def test_shard_cache_lru_eviction():
    """LRU eviction removes oldest shard at capacity."""
    cache = ShardCache(max_shards=2)
    cache.put(0, torch.randn(5, 2), torch.randn(5, 1))
    cache.put(1, torch.randn(5, 2), torch.randn(5, 1))
    # cache full at 2 shards
    assert cache.size == 2

    # accessing 0 makes it MRU
    cache.get(0)
    # inserting new one evicts shard 1 (LRU)
    cache.put(2, torch.randn(5, 2), torch.randn(5, 1))
    assert cache.size == 2
    assert cache.get(1) is None  # evicted
    assert cache.get(0) is not None  # still there
    assert cache.get(2) is not None  # just inserted


def test_shard_cache_hit_rate_tracking():
    """Hits and misses are counted correctly."""
    cache = ShardCache(max_shards=2)
    cache.put(0, torch.randn(5, 2), torch.randn(5, 1))
    cache.get(0)  # hit
    cache.get(999)  # miss
    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate"] == "50.0%"


# ── ShardCacheConfig tests ─────────────────────────────────────────


def test_config_defaults():
    """ShardCacheConfig uses sensible defaults."""
    config = ShardCacheConfig()
    assert config.shard_size == 1000
    assert config.max_cached_shards == 8
    assert config.dtype == torch.float16


# ── Manifest tests ─────────────────────────────────────────────────


def test_save_and_load_manifest(tmp_path: Path):
    """save_manifest + load_manifest round-trips correctly."""
    config = ShardCacheConfig(cache_dir=tmp_path / "cache")
    manifest = CacheManifest(
        version=1,
        dataset_root="/data/test",
        dataset_hash="abc123",
        model_versions={"dino": "test-dino", "vae": "test-vae"},
        total_shards=10,
        shard_count=10,
    )
    save_manifest(config, manifest)

    loaded = load_manifest(config.cache_dir)
    assert loaded.version == 1
    assert loaded.dataset_root == "/data/test"
    assert loaded.dataset_hash == "abc123"
    assert loaded.total_shards == 10
    assert loaded.model_versions == {"dino": "test-dino", "vae": "test-vae"}


def test_load_manifest_missing(tmp_path: Path):
    """load_manifest raises FileNotFoundError for non-existent cache."""
    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path / "nonexistent")


# ── compute_dataset_hash tests ──────────────────────────────────────


def test_dataset_hash_deterministic():
    """Same paths produce same hash."""
    paths = [Path("/data/img_001.jpg"), Path("/data/img_002.jpg")]
    h1 = compute_dataset_hash(paths)
    h2 = compute_dataset_hash(paths)
    assert h1 == h2


def test_dataset_hash_order_independent():
    """Hash is the same regardless of input order."""
    paths = [Path("/data/img_001.jpg"), Path("/data/img_002.jpg")]
    h1 = compute_dataset_hash(paths)
    h2 = compute_dataset_hash(list(reversed(paths)))
    assert h1 == h2


def test_dataset_hash_different_paths():
    """Different paths produce different hashes."""
    paths_a = [Path("/data/a/img.jpg")]
    paths_b = [Path("/data/b/img.jpg")]
    assert compute_dataset_hash(paths_a) != compute_dataset_hash(paths_b)


# ── validate_cache tests ────────────────────────────────────────────


def test_validate_cache_no_manifest(tmp_path: Path):
    """validate_cache returns False when no manifest exists."""
    config = ShardCacheConfig(cache_dir=tmp_path / "cache")
    result = validate_cache(config, [Path("/fake/img.jpg")], "dino-x", "vae-y")
    assert result is False


def test_validate_cache_hash_mismatch(tmp_path: Path):
    """validate_cache returns False when dataset hash differs."""
    config = ShardCacheConfig(cache_dir=tmp_path / "cache")
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = CacheManifest(dataset_hash="stale_hash", total_shards=1, shard_count=1)
    save_manifest(config, manifest)

    result = validate_cache(config, [Path("/fake/img.jpg")], "dino-x", "vae-y")
    assert result is False


def test_validate_cache_model_mismatch(tmp_path: Path):
    """validate_cache returns False when model version differs."""
    config = ShardCacheConfig(cache_dir=tmp_path / "cache")
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = CacheManifest(
        dataset_hash="abc",
        total_shards=1,
        shard_count=1,
        model_versions={"dino": "old-model", "vae": "test-vae"},
    )
    save_manifest(config, manifest)

    result = validate_cache(config, [Path("/fake/img.jpg")], "new-model", "test-vae")
    assert result is False


def test_validate_cache_valid(tmp_path: Path):
    """validate_cache returns True when everything matches."""
    config = ShardCacheConfig(cache_dir=tmp_path / "cache")
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    # Use the actual computed hash so validate_cache will match
    paths = [Path("/fake/img.jpg")]
    actual_hash = compute_dataset_hash(paths)
    manifest = CacheManifest(
        dataset_hash=actual_hash,
        total_shards=1,
        shard_count=1,
        model_versions={"dino": "dino-x", "vae": "vae-y"},
    )
    save_manifest(config, manifest)

    result = validate_cache(config, paths, "dino-x", "vae-y")
    assert result is True


# ── CachedDataset tests (mocked shard files) ───────────────────────


@pytest.fixture
def mock_shard_structure(tmp_path: Path):
    """Create a minimal mock cache with 3 shards (10 images each)."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # manifest
    manifest = CacheManifest(
        version=1,
        dataset_root="/data/test",
        dataset_hash="abc123",
        model_versions={"dino": "d", "vae": "v"},
        total_shards=3,
        shard_count=3,
    )
    save_manifest(ShardCacheConfig(cache_dir=cache_dir), manifest)

    # shard files
    for shard_id in range(3):
        shard_dir = cache_dir / f"shard_{shard_id:05d}"
        shard_dir.mkdir()

        # Create mock tensor files
        n = 10 if shard_id < 2 else 8  # last shard has 8 images
        tokens = torch.randn(n, 1024, 384, dtype=torch.float16)
        latents = torch.randn(n, 16, 64, 64, dtype=torch.float16)
        torch.save(tokens, shard_dir / "patch_tokens.pt")
        torch.save(latents, shard_dir / "gt_latents.pt")

        meta = ShardMeta(shard_id=shard_id, n_images=n)
        with open(shard_dir / "meta.json", "w") as f:
            json.dump(asdict(meta), f)

    return cache_dir


def test_cached_dataset_len(mock_shard_structure: Path):
    """CachedDataset.__len__ returns total sample count."""
    config = ShardCacheConfig(cache_dir=mock_shard_structure)
    dataset = CachedDataset(config)
    assert len(dataset) == 28  # 10 + 10 + 8


def test_cached_dataset_shard_count(mock_shard_structure: Path):
    """CachedDataset knows its shard count."""
    config = ShardCacheConfig(cache_dir=mock_shard_structure)
    dataset = CachedDataset(config)
    assert dataset.shard_count == 3


def test_cached_dataset_getitem_shape(mock_shard_structure: Path):
    """__getitem__ returns single samples with correct shapes."""
    config = ShardCacheConfig(cache_dir=mock_shard_structure)
    dataset = CachedDataset(config)

    tokens, latents = dataset[0]
    assert tokens.shape == (1, 1024, 384)
    assert latents.shape == (1, 16, 64, 64)

    # Different shard
    tokens, latents = dataset[15]
    assert tokens.shape == (1, 1024, 384)
    assert latents.shape == (1, 16, 64, 64)


def test_cached_dataset_getitem_dtype(mock_shard_structure: Path):
    """__getitem__ returns float32 tensors (training precision)."""
    config = ShardCacheConfig(cache_dir=mock_shard_structure)
    dataset = CachedDataset(config)

    tokens, latents = dataset[0]
    assert tokens.dtype == torch.float32
    assert latents.dtype == torch.float32


def test_cached_dataset_cache_hits(mock_shard_structure: Path):
    """Sequential access in same shard produces cache hits."""
    config = ShardCacheConfig(cache_dir=mock_shard_structure)
    dataset = CachedDataset(config)

    # Access samples 0, 1, 2 (all in shard_0)
    dataset[0]
    dataset[1]
    dataset[2]

    stats = dataset.shard_stats()
    assert stats["hits"] == 2  # first access is miss, then 2 hits
    assert stats["misses"] == 1
    assert stats["size"] == 1  # only shard_0 loaded


def test_cached_dataset_multi_shard(mock_shard_structure: Path):
    """Accessing across shards loads and evicts correctly."""
    config = ShardCacheConfig(cache_dir=mock_shard_structure, max_cached_shards=2)
    dataset = CachedDataset(config)

    # Each shard has 10 samples
    dataset[0]  # shard_0, miss → load
    dataset[10]  # shard_1, miss → evict shard_0 (LRU), load shard_1
    dataset[20]  # shard_2, miss → evict shard_1 (LRU), load shard_2

    stats = dataset.shard_stats()
    assert stats["misses"] == 3
    assert stats["size"] == 2  # max_cached_shards


def test_cached_dataset_info(mock_shard_structure: Path):
    """CachedDataset.info() returns summary dict."""
    config = ShardCacheConfig(cache_dir=mock_shard_structure)
    dataset = CachedDataset(config)
    info = dataset.info()
    assert info["total_samples"] == 28
    assert info["shard_count"] == 3
    assert info["shard_size"] == 1000


# ── ShardAwareSampler tests ────────────────────────────────────────


def test_sampler_covers_all_indices(mock_shard_structure: Path):
    """ShardAwareSampler yields all indices exactly once."""
    config = ShardCacheConfig(cache_dir=mock_shard_structure)
    dataset = CachedDataset(config)
    sampler = ShardAwareSampler(dataset, shard_size=1000, seed=42)

    indices = list(sampler)
    assert len(indices) == len(dataset)
    assert set(indices) == set(range(len(dataset)))


def test_sampler_shuffles():
    """ShardAwareSampler produces different orders with different seeds."""
    mock_dataset = MagicMock(spec=CachedDataset)
    mock_dataset.shard_count = 5
    mock_dataset._index_map = {i: (i // 100, i % 100) for i in range(500)}
    mock_dataset.__len__ = MagicMock(return_value=500)

    iter1 = list(ShardAwareSampler(mock_dataset, seed=42))
    iter2 = list(ShardAwareSampler(mock_dataset, seed=123))

    assert iter1 != iter2, "Different seeds should produce different orders"


def test_sampler_deterministic():
    """Same seed produces the same order."""
    mock_dataset = MagicMock(spec=CachedDataset)
    mock_dataset.shard_count = 5
    mock_dataset._index_map = {i: (i // 100, i % 100) for i in range(500)}
    mock_dataset.__len__ = MagicMock(return_value=500)

    iter1 = list(ShardAwareSampler(mock_dataset, seed=42))
    iter2 = list(ShardAwareSampler(mock_dataset, seed=42))

    assert iter1 == iter2


def test_sampler_len(mock_shard_structure: Path):
    """ShardAwareSampler.__len__ returns dataset length."""
    config = ShardCacheConfig(cache_dir=mock_shard_structure)
    dataset = CachedDataset(config)
    sampler = ShardAwareSampler(dataset, shard_size=1000, seed=42)
    assert len(sampler) == len(dataset)
