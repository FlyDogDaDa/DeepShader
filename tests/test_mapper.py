"""Tests for src/models.py."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models import DinoToVAE_Linear, DinoToVAE_MLP


def test_linear_shapes() -> None:
    """Linear mapper: [B, 1024, 384] -> (z, mean, logvar)."""
    mapper = DinoToVAE_Linear(hidden_dim=384, latent_dim=16)
    x = torch.randn(2, 1024, 384)
    z, mean, logvar = mapper(x, sample=False)
    assert z.shape == (2, 16, 64, 64)
    assert mean.shape == (2, 16, 64, 64)
    assert logvar.shape == (2, 16, 64, 64)
    # sample=False → z == mean
    assert torch.allclose(z, mean, atol=1e-4)


def test_linear_batch_variations() -> None:
    """Linear mapper works with different batch sizes."""
    mapper = DinoToVAE_Linear()
    for b in [1, 4, 8, 16]:
        z, mean, logvar = mapper(torch.randn(b, 1024, 384), sample=False)
        assert z.shape == (b, 16, 64, 64)
        assert mean.shape == (b, 16, 64, 64)
        assert logvar.shape == (b, 16, 64, 64)


def test_mlp_shapes() -> None:
    """MLP mapper: [B, 1024, 384] -> (z, mean, logvar)."""
    mapper = DinoToVAE_MLP(hidden_dim=384, latent_dim=16)
    x = torch.randn(2, 1024, 384)
    z, mean, logvar = mapper(x, sample=False)
    assert z.shape == (2, 16, 64, 64)
    assert mean.shape == (2, 16, 64, 64)
    assert logvar.shape == (2, 16, 64, 64)


def test_mlp_batch_variations() -> None:
    """MLP mapper works with different batch sizes."""
    mapper = DinoToVAE_MLP()
    for b in [1, 4, 8, 16]:
        z, mean, logvar = mapper(torch.randn(b, 1024, 384), sample=False)
        assert z.shape == (b, 16, 64, 64)
        assert mean.shape == (b, 16, 64, 64)
        assert logvar.shape == (b, 16, 64, 64)


def test_mlp_learnable_norm() -> None:
    """learnable_norm=True -> mapper has trainable LayerNorm params."""
    mapper = DinoToVAE_MLP(learnable_norm=True)
    norm_modules = [m for m in mapper.modules() if isinstance(m, nn.LayerNorm)]
    norm_params = sum(p.numel() for m in norm_modules for p in m.parameters())
    assert norm_params > 0, "Expected LayerNorm params when learnable_norm=True"


def test_mlp_no_norm() -> None:
    """learnable_norm=False -> fewer params than learnable_norm=True."""
    mapper_no = DinoToVAE_MLP(learnable_norm=False)
    mapper_yes = DinoToVAE_MLP(learnable_norm=True)
    params_no = sum(p.numel() for p in mapper_no.parameters())
    params_yes = sum(p.numel() for p in mapper_yes.parameters())
    assert params_yes > params_no, "learnable_norm should add params"


def test_mlp_forward_pass() -> None:
    """MLP forward pass runs without error."""
    mapper = DinoToVAE_MLP()
    x = torch.randn(1, 1024, 384)
    z, mean, logvar = mapper(x, sample=False)
    assert torch.isfinite(z).all(), "Output should contain only finite values"
    assert torch.isfinite(mean).all()
    assert torch.isfinite(logvar).all()


def test_linear_train_step() -> None:
    """Linear mapper: forward -> backward -> params have gradients."""
    mapper = DinoToVAE_Linear()
    x = torch.randn(2, 1024, 384)
    z, _, _ = mapper(x, sample=True)
    loss = z.sum()
    loss.backward()
    for p in mapper.parameters():
        assert p.grad is not None, "Gradient should exist"
        assert p.grad.abs().sum() > 0, "Gradient should be non-zero"


def test_mlp_train_step() -> None:
    """MLP mapper: forward -> backward -> params have gradients."""
    mapper = DinoToVAE_MLP()
    x = torch.randn(2, 1024, 384)
    z, _, _ = mapper(x, sample=True)
    loss = z.sum()
    loss.backward()
    for p in mapper.parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Gradient missing for {p.shape}"
