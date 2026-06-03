"""Pytest configuration for DeepShader tests."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pytest
from PIL import Image

from src.utility import scan_dataset, train_val_test_split

# ── Cache cleanup ─────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clean_cache() -> None:
    """Remove cached scan results before each test."""
    cache = Path.cwd() / ".cache"
    if cache.exists():
        shutil.rmtree(cache)


def _make_jpg(path: Path, size: tuple[int, int] = (512, 512)) -> Path:
    """Create a test JPEG image at *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color=(128, 64, 200))
    img.save(path, "JPEG")
    return path


@pytest.fixture()
def tmp_jpg(tmp_path: Path) -> Path:
    """A single temporary JPEG file (512×512)."""
    jpg = _make_jpg(tmp_path / "test.jpg")
    yield jpg


@pytest.fixture()
def tmp_jpgs(tmp_path: Path) -> Path:
    """A temporary directory containing 5 test JPEGs."""
    for i in range(5):
        _make_jpg(tmp_path / f"img_{i:03d}.jpg")
    return tmp_path


@pytest.fixture()
def tmp_dataset_structure(tmp_path: Path) -> Path:
    """A directory tree mimicking danbooru-images/{dir}/{file}.jpg."""
    base = tmp_path / "danbooru-images"
    base.mkdir()
    for subdir in ("0000", "0001", "0002"):
        d = base / subdir
        d.mkdir()
        for i in range(10):
            _make_jpg(d / f"{i:07d}.jpg")
    return base


@pytest.fixture()
def small_paths(tmp_dataset_structure: Path) -> list[Path]:
    """Paths from the temp dataset (10 × 3 = 30 paths)."""
    paths = scan_dataset(tmp_dataset_structure, use_cache=False)
    assert len(paths) == 30, f"Expected 30 paths, got {len(paths)}"
    return paths


@pytest.fixture()
def artificial_paths() -> list[Path]:
    """100 mock Path objects (no real files) for logic-only tests."""
    return [Path(f"/fake/images/img_{i:05d}.jpg") for i in range(100)]


@pytest.fixture()
def one_path() -> Path:
    """Single fake path."""
    return Path("/fake/images/one.jpg")


@pytest.fixture(scope="session")
def sample_jpg(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One persistent sample image reused across tests."""
    tmp = tmp_path_factory.mktemp("sample")
    return _make_jpg(tmp / "sample.jpg")
