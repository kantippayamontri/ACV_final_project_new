"""Tests for eval baseline script behavior."""
from types import SimpleNamespace

import pytest
import torch


class _DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.loaded_state = None

    def to(self, device):
        return self

    def eval(self):
        return self

    def generate(self, features, mask, max_new_tokens):
        return ["prediction"] * features.shape[0]

    def load_state_dict(self, state_dict):
        self.loaded_state = state_dict


class _DummyOptimizer:
    def __init__(self):
        self.loaded_state = None

    def load_state_dict(self, state_dict):
        self.loaded_state = state_dict


def _run_eval_and_capture_model_kwargs(
    tmp_path,
    monkeypatch,
    checkpoint_payload,
    loaded_train_args=None,
    stub_load_checkpoint=True,
):
    from scripts import eval_baseline

    captured = {}
    torch_load_calls = {"count": 0}

    def fake_model_ctor(**kwargs):
        captured.update(kwargs)
        return _DummyModel()

    class _DummyDataset:
        def __len__(self):
            return 1

    batch = {
        "features": torch.zeros(1, 2, 768),
        "attention_mask": torch.ones(1, 2, dtype=torch.bool),
        "target_texts": ["reference"],
    }

    monkeypatch.setattr(
        eval_baseline,
        "parse_args",
        lambda: SimpleNamespace(
            lmdb="unused.lmdb",
            metadata="unused.csv",
            checkpoint=str(tmp_path / "checkpoint.pt"),
            pretrained_llm="fake-llm",
            output_dir=str(tmp_path / "outputs"),
            batch_size=1,
            max_new_tokens=4,
            device="cpu",
            num_workers=0,
        ),
    )
    monkeypatch.setattr(eval_baseline, "How2SignLMDBDataset", lambda **kwargs: _DummyDataset())
    monkeypatch.setattr(eval_baseline, "DataLoader", lambda *args, **kwargs: [batch])
    monkeypatch.setattr(eval_baseline, "VisualLLMBaseline", fake_model_ctor)
    monkeypatch.setattr(eval_baseline, "build_optimizer", lambda model, lr: _DummyOptimizer())
    monkeypatch.setattr(
        eval_baseline.torch,
        "load",
        lambda path, map_location, weights_only=False: (
            torch_load_calls.__setitem__("count", torch_load_calls["count"] + 1) or checkpoint_payload
        ),
    )
    if stub_load_checkpoint:
        monkeypatch.setattr(
            eval_baseline,
            "load_checkpoint",
            lambda model, optimizer, checkpoint_path, device, checkpoint_data=None: (
                3,
                {"BLEU": 12.0},
                loaded_train_args,
            ),
        )
    monkeypatch.setattr(eval_baseline, "compute_metrics", lambda predictions, references: {"BLEU": 0.0, "ROUGE-L": 0.0})
    monkeypatch.setattr(eval_baseline, "save_predictions_jsonl", lambda predictions, references, path: None)
    monkeypatch.setattr(eval_baseline, "save_metrics_json", lambda metrics, path: None)
    monkeypatch.setattr(eval_baseline, "tqdm", lambda iterable, desc=None: iterable)

    eval_baseline.main()

    return captured, torch_load_calls["count"]


def test_eval_builds_model_from_checkpoint_lora_args(tmp_path, monkeypatch):
    captured, _ = _run_eval_and_capture_model_kwargs(
        tmp_path,
        monkeypatch,
        checkpoint_payload={
            "args": {"lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05},
        },
        loaded_train_args={"lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05},
    )

    assert captured == {
        "pretrained_llm": "fake-llm",
        "use_lora": True,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
    }


def test_eval_uses_default_lora_args_when_checkpoint_args_missing(tmp_path, monkeypatch):
    captured, _ = _run_eval_and_capture_model_kwargs(
        tmp_path,
        monkeypatch,
        checkpoint_payload={"args": {}},
        loaded_train_args={},
    )

    assert captured == {
        "pretrained_llm": "fake-llm",
        "use_lora": True,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
    }


def test_eval_supports_legacy_checkpoint_without_args_key(tmp_path, monkeypatch):
    captured, _ = _run_eval_and_capture_model_kwargs(
        tmp_path,
        monkeypatch,
        checkpoint_payload={
            "epoch": 3,
            "metrics": {"BLEU": 12.0},
            "model_state": {},
            "optimizer_state": {},
        },
        stub_load_checkpoint=False,
    )

    assert captured == {
        "pretrained_llm": "fake-llm",
        "use_lora": True,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
    }


def test_eval_loads_checkpoint_once_when_using_real_loader(tmp_path, monkeypatch):
    _, torch_load_calls = _run_eval_and_capture_model_kwargs(
        tmp_path,
        monkeypatch,
        checkpoint_payload={
            "epoch": 3,
            "metrics": {"BLEU": 12.0},
            "model_state": {},
            "optimizer_state": {},
            "args": {"lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05},
        },
        stub_load_checkpoint=False,
    )

    assert torch_load_calls == 1


def test_eval_rejects_empty_dataset(tmp_path, monkeypatch):
    from scripts import eval_baseline

    class _EmptyDataset:
        def __len__(self):
            return 0

    monkeypatch.setattr(
        eval_baseline,
        "parse_args",
        lambda: SimpleNamespace(
            lmdb="unused.lmdb",
            metadata="unused.csv",
            checkpoint=str(tmp_path / "checkpoint.pt"),
            pretrained_llm="fake-llm",
            output_dir=str(tmp_path / "outputs"),
            batch_size=1,
            max_new_tokens=4,
            device="cpu",
            num_workers=0,
        ),
    )
    monkeypatch.setattr(eval_baseline, "How2SignLMDBDataset", lambda **kwargs: _EmptyDataset())

    with pytest.raises(ValueError, match="Evaluation dataset is empty"):
        eval_baseline.main()


def test_train_rejects_empty_validation_dataset(tmp_path, monkeypatch):
    from scripts import train_baseline

    class _Dataset:
        def __init__(self, size):
            self.size = size

        def __len__(self):
            return self.size

    datasets = [_Dataset(1), _Dataset(0)]

    monkeypatch.setattr(
        train_baseline,
        "parse_args",
        lambda: SimpleNamespace(
            train_lmdb="unused-train.lmdb",
            train_metadata="unused-train.csv",
            val_lmdb="unused-val.lmdb",
            val_metadata="unused-val.csv",
            pretrained_llm="fake-llm",
            output_dir=str(tmp_path / "outputs"),
            batch_size=1,
            grad_accum_steps=1,
            epochs=1,
            lr=1e-4,
            max_new_tokens=4,
            device="cpu",
            num_workers=0,
            seed=123,
            precision="fp32",
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            save_every_epoch=False,
            val_samples=None,
        ),
    )
    monkeypatch.setattr(train_baseline, "set_seed", lambda seed: None)
    monkeypatch.setattr(train_baseline, "How2SignLMDBDataset", lambda **kwargs: datasets.pop(0))

    with pytest.raises(ValueError, match="Validation dataset is empty"):
        train_baseline.main()
