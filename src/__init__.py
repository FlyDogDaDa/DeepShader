"""DeepShader — DINOv3 → FLUX VAE mapping pipeline."""

from __future__ import annotations

from src.dataset import (
    create_dataloaders,
    default_transform,
    load_image,
)
from src.models import DinoToVAE_Linear, DinoToVAE_MLP
from src.pretrains import DINO_MEAN, DINO_STD, DINOModel, VAEModel, load_dino, load_vae
from src.trainer import (
    TrainingConfig,
    create_mapper,
    load_checkpoint,
    run_training,
    save_checkpoint,
    train_epoch,
    validate_epoch,
)
from src.utility import scan_dataset

__all__ = [
    # dataset
    "create_dataloaders",
    "default_transform",
    "load_image",
    # pretrains
    "load_vae",
    "load_dino",
    "VAEModel",
    "DINOModel",
    "DINO_MEAN",
    "DINO_STD",
    # models
    "DinoToVAE_Linear",
    "DinoToVAE_MLP",
    # trainer
    "TrainingConfig",
    "train_epoch",
    "validate_epoch",
    "create_mapper",
    "run_training",
    "save_checkpoint",
    "load_checkpoint",
    # utility
    "scan_dataset",
]
