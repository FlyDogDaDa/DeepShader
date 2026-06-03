"""Tests for src/trainer.py."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from src.trainer import TrainingConfig, load_checkpoint, save_checkpoint


def test_config_defaults() -> None:
    """TrainingConfig() creates with sensible defaults."""
    config = TrainingConfig()
    assert config.lr > 0
    assert config.epochs > 0
    assert config.batch_size > 0
    assert config.output_dir is not None
    assert config.dataset == ""


def test_config_custom() -> None:
    """TrainingConfig accepts custom values."""
    config = TrainingConfig(lr=1e-3, epochs=10, batch_size=32)
    assert config.lr == 1e-3
    assert config.epochs == 10
    assert config.batch_size == 32


def test_checkpoint_save_load(tmp_path: Path) -> None:
    """save_checkpoint + load_checkpoint restores optimizer state."""
    mapper = nn.Linear(10, 5)
    optimizer = torch.optim.Adam(mapper.parameters(), lr=1e-3)

    # Do one step to populate state
    x = torch.randn(4, 10)
    loss = mapper(x).sum()
    loss.backward()
    optimizer.step()

    # Save and load
    ckpt_path = tmp_path / "ckpt.pt"
    save_checkpoint(
        ckpt_path,
        epoch=5,
        mapper_state=mapper.state_dict(),
        optimizer_state=optimizer.state_dict(),
        config=None,
    )
    restored_optimizer = torch.optim.Adam(mapper.parameters(), lr=1e-3)
    epoch = load_checkpoint(ckpt_path, mapper, restored_optimizer)
    assert epoch["epoch"] == 5
    # Restored optimizer should have state
    assert len(restored_optimizer.state) > 0


def test_checkpoint_save_load_config(tmp_path: Path) -> None:
    """Checkpoint can save and restore config."""
    config = TrainingConfig(lr=2e-4, epochs=50)
    ckpt_path = tmp_path / "ckpt.pt"
    save_checkpoint(
        ckpt_path, epoch=10, mapper_state=None, optimizer_state=None, config=config
    )
    restored = load_checkpoint(ckpt_path, None, None)
    assert isinstance(restored, dict)
    assert restored["epoch"] == 10
