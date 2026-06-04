"""Training and validation loop for DinoToVAE.

Provides:
    * ``TrainingConfig`` — all hyper-parameters as a dataclass
    * ``train_epoch()`` / ``validate_epoch()`` — per-epoch helpers (image mode)
    * ``train_epoch_cached()`` / ``validate_epoch_cached()`` — cached mode
    * ``save_checkpoint()`` / ``load_checkpoint()`` — state persistence
    * ``run_training()`` — one-shot full training call (auto-detects mode)

Two training modes:
    Image mode (default): images → DINOv3 → mapper → VAE → latents
    Cached mode:          patch_tokens + gt_latents from shard cache
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from src.models import DinoToVAE_Linear, DinoToVAE_MLP
from src.pretrains import DINOModel, VAEModel, load_dino, load_vae

# ── Logger setup ─────────────────────────────────────────────────


def setup_logging(output_dir: str) -> logging.Logger:
    """Setup logging to file and stdout."""
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "train.log"

    logger = logging.getLogger("deepshader")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # avoid duplicate handlers

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


# ── Config ─────────────────────────────────────────────────────────


@dataclass
class TrainingConfig:
    """Training hyper-parameters with sensible defaults."""

    lr: float = 1e-3
    epochs: int = 100
    batch_size: int = 8
    num_workers: int = 16
    warmup_steps: int = 1000
    max_steps: int = 0  # 0 = run all epochs
    log_freq: int = 100  # log every N training steps
    val_freq: int = 5  # validate every N epochs
    val_ratio: float = 0.1  # fraction for validation set
    sample_images: int = 4  # decode N samples every val step
    debug: bool = False  # print shapes for debugging
    output_dir: str = "runs/default"  # root output directory
    seed: int = 42
    mapper: str = "mlp"  # "linear" | "mlp"
    hidden_channels: int = 256
    num_layers: int = 4
    learnable_norm: bool = True
    dataset: str = ""  # dataset path (for reference)


# ── Config helpers ───────────────────────────────────────────────


def save_config(config: TrainingConfig, out_dir: Path) -> None:
    """Save training config as YAML."""
    import yaml

    cfg_path = out_dir / "config.yaml"
    with cfg_path.open("w") as f:
        yaml.dump(asdict(config), f, default_flow_style=False)


# ── Model factory ─────────────────────────────────────────────────


def create_mapper(config: TrainingConfig, device: torch.device) -> nn.Module:
    """Instantiate the mapper network."""
    if config.mapper == "linear":
        return DinoToVAE_Linear().to(device)
    else:
        return DinoToVAE_MLP(
            hidden_channels=config.hidden_channels,
            num_layers=config.num_layers,
            learnable_norm=config.learnable_norm,
        ).to(device)


# ── Epoch helpers ─────────────────────────────────────────────────


def _decode_samples(
    mapper,
    loader,
    dino,
    vae,
    device,
    out_dir,
    num_samples=4,
) -> None:
    """Decode samples and save as images."""
    from torchvision.utils import save_image

    mapper.eval()
    samples_saved = 0
    with torch.no_grad():
        for images in loader:
            if samples_saved >= num_samples:
                break
            images = images.to(device)
            patch_tokens = dino.extract(images)
            pred_latents = mapper(patch_tokens)
            decoded = vae.decode(pred_latents)  # returns Tensor directly
            for i in range(min(decoded.shape[0], num_samples - samples_saved)):
                save_image(
                    decoded[i],
                    out_dir / f"sample_{samples_saved:03d}.png",
                    normalize=True,
                    value_range=(-1, 1),
                )
                samples_saved += 1
                if samples_saved >= num_samples:
                    break
    mapper.train()


def _decode_samples_cached(
    mapper: nn.Module,
    loader: DataLoader,
    vae: VAEModel | None,
    device: torch.device,
    out_dir: Path,
    num_samples: int = 4,
) -> None:
    """Decode samples from cached loader (patch_tokens + gt_latents)."""
    from torchvision.utils import save_image

    mapper.eval()
    samples_saved = 0
    with torch.no_grad():
        for tokens, _ in loader:
            if samples_saved >= num_samples:
                break
            tokens = tokens.squeeze(1).to(device)
            pred_latents = mapper(tokens)
            if vae is not None:
                decoded = vae.decode(pred_latents)
            else:
                decoded = pred_latents
            for i in range(min(decoded.shape[0], num_samples - samples_saved)):
                if vae is not None:
                    save_image(
                        decoded[i],
                        out_dir / f"sample_{samples_saved:03d}.png",
                        normalize=True,
                        value_range=(-1, 1),
                    )
                else:
                    save_image(
                        pred_latents[i], out_dir / f"sample_{samples_saved:03d}.png"
                    )
                samples_saved += 1
                if samples_saved >= num_samples:
                    break
    mapper.train()


def train_epoch(
    mapper: nn.Module,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    loader: DataLoader,
    dino: DINOModel,
    vae: VAEModel,
    device: torch.device,
    debug: bool = False,
    log_freq: int = 100,
    logger: logging.Logger | None = None,
    epoch: int = 0,
) -> float:
    """Run one training epoch. Returns mean loss."""
    mapper.train()
    total_loss = 0.0
    n_samples = 0

    for batch_idx, images in enumerate(loader):
        images = images.to(device)  # [B, 3, 512, 512] in [0, 1]
        optimizer.zero_grad()

        if debug:
            print(f"[debug] batch {batch_idx}: images={images.shape}")

        # DINOv3 → patch tokens
        patch_tokens = dino.extract(images)  # [B, 1024, 384]

        if debug:
            print(f"[debug] patch_tokens={patch_tokens.shape}")

        # Mapper → pred latents
        pred_latents = mapper(patch_tokens)  # [B, 16, 64, 64]

        if debug:
            print(f"[debug] pred_latents={pred_latents.shape}")

        # VAE → gt latents
        with torch.no_grad():
            gt_latents = vae.encode(images * 2 - 1)  # [B, 16, 64, 64]

        if debug:
            print(f"[debug] gt_latents={gt_latents.shape}")

        # Loss & backward
        loss = F.mse_loss(pred_latents, gt_latents)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * images.shape[0]
        n_samples += images.shape[0]

        # Log every N steps, and always on step 1
        if (
            logger is not None
            and log_freq > 0
            and ((batch_idx + 1) % log_freq == 0 or batch_idx == 0)
        ):
            global_step = epoch * len(loader) + batch_idx + 1
            logger.info(
                f"  step {global_step:,}  batch={batch_idx + 1}  "
                f"loss={loss.item():.4f}  lr={optimizer.param_groups[0]['lr']:.6f}"
            )

    return total_loss / n_samples


def validate_epoch(
    mapper: nn.Module,
    loader: DataLoader,
    dino: DINOModel,
    vae: VAEModel,
    device: torch.device,
) -> float:
    """Run one validation epoch (no_grad). Returns mean loss."""
    mapper.eval()
    total_loss = 0.0
    n_samples = 0

    with torch.no_grad():
        for images in loader:
            images = images.to(device)

            patch_tokens = dino.extract(images)
            pred_latents = mapper(patch_tokens)
            gt_latents = vae.encode(images * 2 - 1)

            loss = F.mse_loss(pred_latents, gt_latents)
            total_loss += loss.item() * images.shape[0]
            n_samples += images.shape[0]

    return total_loss / n_samples


# ── Cached Epoch helpers ───────────────────────────────────────────


def train_epoch_cached(
    mapper: nn.Module,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    loader: DataLoader,
    device: torch.device,
    debug: bool = False,
    log_freq: int = 100,
    logger: logging.Logger | None = None,
    epoch: int = 0,
) -> float:
    """Run one training epoch with cached data (patch_tokens + gt_latents).

    Unlike train_epoch(), this does NOT need DINOv3 or VAE - the data is already encoded.
    Only the mapper model is trained, making it ~100x faster per step.
    """
    mapper.train()
    total_loss = 0.0
    n_samples = 0

    for batch_idx, (patch_tokens, gt_latents) in enumerate(loader):
        patch_tokens = patch_tokens.to(device)  # [B, 1024, 384]
        gt_latents = gt_latents.to(device)  # [B, 16, 64, 64]
        optimizer.zero_grad()

        if debug:
            print(
                f"[debug] batch {batch_idx}: tokens={patch_tokens.shape}, gt={gt_latents.shape}"
            )

        # Mapper → pred latents
        pred_latents = mapper(patch_tokens)  # [B, 16, 64, 64]

        if debug:
            print(f"[debug] pred_latents={pred_latents.shape}")

        # Loss & backward (only mapper parameters)
        loss = F.mse_loss(pred_latents, gt_latents)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * patch_tokens.shape[0]
        n_samples += patch_tokens.shape[0]

        # Log every N steps, and always on step 1
        if (
            logger is not None
            and log_freq > 0
            and ((batch_idx + 1) % log_freq == 0 or batch_idx == 0)
        ):
            global_step = epoch * len(loader) + batch_idx + 1
            logger.info(
                f"  step {global_step:,}  batch={batch_idx + 1}  "
                f"loss={loss.item():.4f}  lr={optimizer.param_groups[0]['lr']:.6f}"
            )

    return total_loss / n_samples


def validate_epoch_cached(
    mapper: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Run one validation epoch with cached data (no_grad)."""
    mapper.eval()
    total_loss = 0.0
    n_samples = 0

    with torch.no_grad():
        for patch_tokens, gt_latents in loader:
            patch_tokens = patch_tokens.to(device)
            gt_latents = gt_latents.to(device)

            pred_latents = mapper(patch_tokens)
            loss = F.mse_loss(pred_latents, gt_latents)
            total_loss += loss.item() * patch_tokens.shape[0]
            n_samples += patch_tokens.shape[0]

    return total_loss / n_samples


# ── Checkpoint ────────────────────────────────────────────────────


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    mapper_state: dict[str, Any] | None,
    optimizer_state: dict[str, Any] | None,
    config: TrainingConfig | None,
) -> None:
    """Save mapper + optimizer state to disk."""
    ckpt: dict[str, Any] = {
        "epoch": epoch,
        "mapper_state": mapper_state,
        "optimizer_state": optimizer_state,
    }
    if config is not None:
        ckpt["config"] = asdict(config)
    torch.save(ckpt, path)


def load_checkpoint(
    path: Path,
    mapper: nn.Module | None,
    optimizer: Optimizer | None,
) -> dict[str, Any]:
    """Load checkpoint and restore mapper/optimizer state.

    Returns the raw checkpoint dict for inspecting ``["epoch"]`` etc.
    """
    ckpt = torch.load(path, weights_only=False)
    if mapper is not None and ckpt.get("mapper_state") is not None:
        mapper.load_state_dict(ckpt["mapper_state"])
    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt


# ── Scheduler ─────────────────────────────────────────────────────


def make_scheduler(
    optimizer: Optimizer, total_steps: int, warmup_steps: int
) -> LambdaLR:
    """Cosine annealing with linear warmup."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            # Linear warmup from 0.1 to 1.0
            return 0.1 + 0.9 * (step / max(warmup_steps, 1))
        # Cosine decay from 1.0 to 0.1
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.1 + 0.9 * 0.5 * (
            1 + torch.cos(torch.tensor(3.14159 * progress)).item()
        )

    return LambdaLR(optimizer, lr_lambda)


# ── High-level training ───────────────────────────────────────────


def run_training(
    config: TrainingConfig,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    dino: DINOModel | None = None,
    vae: VAEModel | None = None,
    resume_from: Path | str | None = None,
) -> tuple[nn.Module, Optimizer]:
    """Run the full training loop.

    Auto-detects training mode based on whether DINO/VAE are provided:
        - With dino+vae: image mode (images -> DINO -> mapper -> VAE)
        - Without dino/vae: cached mode (patch_tokens + latents from cache)

    Args:
        config: Training configuration.
        train_loader: Training DataLoader.
        val_loader: Optional validation DataLoader.
        dino: DINO model (None for cached mode).
        vae: VAE model (None for cached mode).
        resume_from: Path to checkpoint file to resume from.

    Returns:
        Tuple of (trained_mapper, optimizer).
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(config.output_dir)

    # Auto-detect mode
    use_cached = dino is None and vae is None
    if use_cached:
        mode_label = "cached (patch_tokens + latents)"
    else:
        mode_label = "image (DINO + VAE encode)"

    # Setup logger
    logger = setup_logging(config.output_dir)

    # Save config
    save_config(config, out)
    logger.info(f"Output directory: {out}")
    logger.info(f"Mode: {mode_label}")
    logger.info(f"Dataset: {config.dataset}")

    mapper = create_mapper(config, device)
    total_params = sum(p.numel() for p in mapper.parameters())
    logger.info(f"Mapper: {config.mapper}, params: {total_params:,}")

    optimizer = torch.optim.AdamW(mapper.parameters(), lr=config.lr, weight_decay=1e-4)
    steps_per_epoch = len(train_loader)
    total_steps = config.epochs * steps_per_epoch
    scheduler = make_scheduler(optimizer, total_steps, config.warmup_steps)

    # Resume from checkpoint
    start_epoch = 0
    if resume_from:
        ckpt_path = Path(resume_from)
        if ckpt_path.exists():
            ckpt = load_checkpoint(ckpt_path, mapper, optimizer)
            start_epoch = ckpt["epoch"]
            logger.info(f"Resumed from epoch {start_epoch}")

    logger.info(f"Training: {config.epochs} epochs, {steps_per_epoch} steps/epoch")
    logger.info(f"Checkpoints: {out}/checkpoints/")
    logger.info(f"Samples:   {out}/samples/")
    logger.info(f"Log:       {out}/logs/train.log")

    # Training loop
    for epoch in range(start_epoch, config.epochs):
        if use_cached:
            train_loss = train_epoch_cached(
                mapper,
                optimizer,
                scheduler,
                train_loader,
                device,
                debug=config.debug,
                log_freq=config.log_freq,
                logger=logger,
                epoch=epoch,
            )
        else:
            train_loss = train_epoch(
                mapper,
                optimizer,
                scheduler,
                train_loader,
                dino,
                vae,
                device,
                debug=config.debug,
                log_freq=config.log_freq,
                logger=logger,
                epoch=epoch,
            )
        logger.info(f"Epoch {epoch + 1}/{config.epochs}  loss={train_loss:.4f}")

        # Validation
        if val_loader and (epoch % config.val_freq == 0 or epoch == 0):
            if use_cached:
                val_loss = validate_epoch_cached(mapper, val_loader, device)
            else:
                val_loss = validate_epoch(mapper, val_loader, dino, vae, device)
            logger.info(f"  val={val_loss:.4f}")

            # Decode samples
            if config.sample_images > 0:
                samples_dir = out / "samples" / f"epoch_{epoch + 1:04d}"
                samples_dir.mkdir(parents=True, exist_ok=True)
                if use_cached:
                    _decode_samples_cached(
                        mapper,
                        train_loader,
                        vae,
                        device,
                        samples_dir,
                        num_samples=config.sample_images,
                    )
                else:
                    _decode_samples(
                        mapper,
                        train_loader,
                        dino,
                        vae,
                        device,
                        samples_dir,
                        num_samples=config.sample_images,
                    )
                logger.info(f"  samples saved to {samples_dir}")

        # Checkpoint
        if config.output_dir:
            ckpt_dir = out / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            save_checkpoint(
                ckpt_dir / f"epoch_{epoch + 1:04d}.pt",
                epoch=epoch + 1,
                mapper_state=mapper.state_dict(),
                optimizer_state=optimizer.state_dict(),
                config=config,
            )

    # Save final model
    final_dir = out / "checkpoints"
    final_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        final_dir / "final.pt",
        epoch=config.epochs,
        mapper_state=mapper.state_dict(),
        optimizer_state=optimizer.state_dict(),
        config=config,
    )
    logger.info(f"Training complete! Final checkpoint: {final_dir}/final.pt")

    return mapper, optimizer
