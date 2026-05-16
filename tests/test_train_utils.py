"""Tests for training utilities and epoch behavior."""
from contextlib import nullcontext

import torch


def test_mixed_precision_context_fp32_returns_grad_enabled_context():
    from src.training.train_utils import mixed_precision_context

    autocast, scaler = mixed_precision_context("fp32", "cpu")

    assert scaler is None
    assert autocast is not None
    with autocast:
        x = torch.tensor(1.0, requires_grad=True)
        y = x * 2
    y.backward()
    assert x.grad.item() == 2.0


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.calls = 0

    def forward(self, features, mask, targets):
        self.calls += 1
        return self.weight.square()


def test_train_epoch_with_fp32_keeps_gradients_enabled():
    from scripts.train_baseline import train_epoch

    model = _TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    loader = [
        {"features": torch.zeros(1, 1, 1), "attention_mask": torch.ones(1, 1, dtype=torch.bool), "target_texts": ["a"]},
    ]

    train_epoch(model, loader, optimizer, scaler=None, autocast=None, grad_accum_steps=1, device="cpu")

    assert model.weight.item() != 1.0


def test_train_epoch_flushes_leftover_gradients(monkeypatch):
    from scripts.train_baseline import train_epoch

    model = _TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    step_calls = {"count": 0}

    original_step = optimizer.step

    def counted_step(*args, **kwargs):
        step_calls["count"] += 1
        return original_step(*args, **kwargs)

    monkeypatch.setattr(optimizer, "step", counted_step)

    loader = [
        {"features": torch.zeros(1, 1, 1), "attention_mask": torch.ones(1, 1, dtype=torch.bool), "target_texts": ["a"]},
        {"features": torch.zeros(1, 1, 1), "attention_mask": torch.ones(1, 1, dtype=torch.bool), "target_texts": ["b"]},
        {"features": torch.zeros(1, 1, 1), "attention_mask": torch.ones(1, 1, dtype=torch.bool), "target_texts": ["c"]},
    ]

    start_weight = model.weight.item()
    train_epoch(model, loader, optimizer, scaler=None, autocast=nullcontext(), grad_accum_steps=2, device="cpu")

    assert model.calls == 3
    assert step_calls["count"] == 2
    assert model.weight.item() != start_weight
