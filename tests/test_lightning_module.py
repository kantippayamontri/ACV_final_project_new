"""Unit tests for Lightning module wrapper."""
import pytest
import torch
from src.models.visual_llm_litmodule import VisualLLMLitModule


@pytest.fixture
def litmodule():
    return VisualLLMLitModule(
        pretrained_llm="models/Meta-Llama-3-8B",
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        lr=2e-4,
        max_new_tokens=32,
    )


def test_training_step(litmodule):
    batch = {
        "features": torch.randn(2, 10, 768).half().cuda(),
        "attention_mask": torch.ones(2, 10, dtype=torch.bool).cuda(),
        "target_texts": ["test sentence 1", "test sentence 2"],
    }

    loss = litmodule.training_step(batch, 0)
    assert loss.dim() == 0
    assert loss.requires_grad


def test_validation_step(litmodule):
    batch = {
        "features": torch.randn(2, 10, 768).half().cuda(),
        "attention_mask": torch.ones(2, 10, dtype=torch.bool).cuda(),
        "target_texts": ["test sentence 1", "test sentence 2"],
    }

    litmodule.validation_step(batch, 0)
    assert len(litmodule.val_predictions) == 2
    assert len(litmodule.val_references) == 2


def test_configure_optimizers(litmodule):
    optimizer = litmodule.configure_optimizers()
    assert optimizer is not None
    assert len(optimizer.param_groups) > 0
