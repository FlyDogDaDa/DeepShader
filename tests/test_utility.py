"""Tests for src/utility.py."""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import pytest

from src.utility import scan_dataset, train_val_test_split

# ── scan_dataset ──────────────────────────────────────────────────


def test_scan_dataset_returns_paths(tmp_dataset_structure: Path) -> None:
    """scan_dataset() returns all .jpg files under the root."""
    paths = scan_dataset(tmp_dataset_structure, use_cache=False)
    assert len(paths) == 30


def test_scan_dataset_filter_by_subdir(tmp_dataset_structure: Path) -> None:
    """scan_dataset() with glob pattern returns matching files only."""
    sub = tmp_dataset_structure / "0000"
    paths = scan_dataset(sub, pattern="*.jpg", use_cache=False)
    assert len(paths) == 10


def test_scan_dataset_no_match_returns_empty(tmp_path: Path) -> None:
    """scan_dataset() on an empty dir returns []."""
    paths = scan_dataset(tmp_path, use_cache=False)
    assert paths == []


def test_scan_dataset_pattern(tmp_dataset_structure: Path) -> None:
    """scan_dataset() accepts custom glob patterns."""
    paths = scan_dataset(
        tmp_dataset_structure, pattern="0000/**/*.jpg", use_cache=False
    )
    assert len(paths) == 10


def test_scan_dataset_cache_hit(tmp_dataset_structure: Path) -> None:
    """Second call with cache returns the same result without rescanning."""
    cache_dir = Path.cwd() / ".cache"
    cache_files = list(cache_dir.glob("*.pkl"))
    n_before = len(cache_files)

    first = scan_dataset(tmp_dataset_structure)
    cache_files = list(cache_dir.glob("*.pkl"))
    assert len(cache_files) == n_before + 1
    assert len(first) == 30

    second = scan_dataset(tmp_dataset_structure)
    assert first == second

    # Verify cache file was loaded (same object from pickle)
    with cache_files[-1].open("rb") as f:
        cached = pickle.load(f)
    assert cached == first


def test_scan_dataset_cache_different_patterns(tmp_dataset_structure: Path) -> None:
    """Different patterns produce different cache files."""
    cache_dir = Path.cwd() / ".cache"
    n_before = len(list(cache_dir.glob("*.pkl")))

    scan_dataset(tmp_dataset_structure, pattern="*.jpg", use_cache=True)
    scan_dataset(tmp_dataset_structure, pattern="0000/**/*.jpg", use_cache=True)
    cache_files = list(cache_dir.glob("*.pkl"))
    assert len(cache_files) == n_before + 2


def test_scan_dataset_cache_no_match_returns_empty(tmp_path: Path) -> None:
    """scan_dataset() on an empty dir returns [] (cache empty list)."""
    cache_dir = Path.cwd() / ".cache"
    n_before = len(list(cache_dir.glob("*.pkl")))

    paths = scan_dataset(tmp_path, use_cache=True)
    assert paths == []
    cache_files = list(cache_dir.glob("*.pkl"))
    assert len(cache_files) == n_before + 1


# ── train_val_test_split ──────────────────────────────────────────


def test_split_proportions(artificial_paths: list[Path]) -> None:
    """Split ratios match expected values (±5 % tolerance)."""
    n = len(artificial_paths)
    train, val, test = train_val_test_split(
        artificial_paths, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42
    )
    assert len(train) == pytest.approx(n * 0.8, abs=5), f"train={len(train)}"
    assert len(val) == pytest.approx(n * 0.1, abs=5), f"val={len(val)}"
    assert len(test) == pytest.approx(n * 0.1, abs=5), f"test={len(test)}"


def test_split_sum(artificial_paths: list[Path]) -> None:
    """train + val + test should equal total."""
    train, val, test = train_val_test_split(
        artificial_paths, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42
    )
    assert len(train) + len(val) + len(test) == len(artificial_paths)


def test_split_no_overlap(artificial_paths: list[Path]) -> None:
    """Sets should be disjoint."""
    train, val, test = train_val_test_split(
        artificial_paths, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42
    )
    train_set = set(train)
    val_set = set(val)
    test_set = set(test)
    assert train_set & val_set == set()
    assert train_set & test_set == set()
    assert val_set & test_set == set()


def test_split_deterministic(artificial_paths: list[Path]) -> None:
    """Same seed → same split."""
    a = train_val_test_split(artificial_paths, seed=42)
    b = train_val_test_split(artificial_paths, seed=42)
    assert a == b


def test_split_different_seed(artificial_paths: list[Path]) -> None:
    """Different seed → different split."""
    a = train_val_test_split(artificial_paths, seed=42)
    b = train_val_test_split(artificial_paths, seed=123)
    assert a != b


def test_split_empty_path() -> None:
    """Empty input → empty output."""
    train, val, test = train_val_test_split([], seed=42)
    assert train == val == test == []


def test_split_all_to_train(artificial_paths: list[Path]) -> None:
    """Extreme ratio: all 100 % to train."""
    train, val, test = train_val_test_split(
        artificial_paths, train_ratio=1.0, val_ratio=0, test_ratio=0, seed=42
    )
    assert len(train) == 100
    assert val == []
    assert test == []
