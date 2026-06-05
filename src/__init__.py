"""DeepShader — DINOv3 → FLUX VAE mapping pipeline."""

from __future__ import annotations

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
from src.dataset import (
    create_dataloaders,
    default_transform,
)
from src.models import (
    DinoToVAE_Linear,
    DinoToVAE_MLP,
    DinoToVAE_ResNet,
    DinoToVAE_Transformer,
    DinoToVAE_TransformerResNet,
)
from src.pretrains import DINO_MEAN, DINO_STD, DINOModel, VAEModel, load_dino, load_vae
from src.trainer import (
    TrainingConfig,
    create_mapper,
    load_checkpoint,
    run_training,
    save_checkpoint,
    train_epoch,
    train_epoch_cached,
    validate_epoch,
    validate_epoch_cached,
)
from src.utility import scan_dataset

__all__ = [
    # cache
    "CachedDataset",
    "ShardAwareSampler",
    "ShardCache",
    "ShardCacheConfig",
    "ShardMeta",
    "CacheManifest",
    "load_manifest",
    "save_manifest",
    "validate_cache",
    "compute_dataset_hash",
    # dataset
    "create_dataloaders",
    "default_transform",
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
    "DinoToVAE_ResNet",
    "DinoToVAE_Transformer",
    "DinoToVAE_TransformerResNet",
    # trainer
    "TrainingConfig",
    "train_epoch",
    "train_epoch_cached",
    "validate_epoch",
    "validate_epoch_cached",
    "create_mapper",
    "run_training",
    "save_checkpoint",
    "load_checkpoint",
    # utility
    "scan_dataset",
]
