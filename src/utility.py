from __future__ import annotations

import hashlib
import pickle
import random
from pathlib import Path


def scan_dataset(
    root: Path,
    pattern: str = "**/*.jpg",
    use_cache: bool = True,
) -> list[Path]:
    """Scan a directory tree and return matching image paths.

    Cache is stored in `<cwd>/.cache/<hash>.pkl` and persists across runs.

    Args:
        root: Root directory to search.
        pattern: Glob pattern (default: all .jpg files recursively).
        use_cache: If ``True`` (default), use disk cache. Set ``False`` to
                   force a fresh scan (useful in tests or during development).

    Returns:
        List of absolute Path objects matching the pattern.
    """
    cache_dir = Path.cwd() / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Hash root + pattern → stable cache filename
    sig = hashlib.sha256(f"{root.resolve()}:::{pattern}".encode()).hexdigest()[:16]
    cache_file = cache_dir / f"{sig}.pkl"

    if use_cache and cache_file.exists():
        with cache_file.open("rb") as f:
            return list(pickle.load(f))

    paths = list(Path(root).rglob(pattern))

    with cache_file.open("wb") as f:
        pickle.dump(paths, f)

    return paths


def train_val_test_split(
    paths: list[Path],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[Path], list[Path], list[Path]]:
    """Split a list of paths into train / val / test subsets.

    Args:
        paths: All image paths to split.
        train_ratio: Fraction for training (default 0.8).
        val_ratio: Fraction for validation (default 0.1).
        test_ratio: Fraction for testing (default 0.1).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_paths, val_paths, test_paths).
    """
    rng = random.Random(seed)
    shuffled = list(paths)
    rng.shuffle(shuffled)

    n = len(shuffled)
    if n == 0:
        return [], [], []

    total = train_ratio + val_ratio + test_ratio
    train_end = int(n * train_ratio / total)
    val_end = int(n * (train_ratio + val_ratio) / total)

    train_end = min(train_end, n)
    val_end = max(train_end, min(val_end, n))

    train_paths = shuffled[:train_end]
    val_paths = shuffled[train_end:val_end]
    test_paths = shuffled[val_end:]

    return train_paths, val_paths, test_paths
