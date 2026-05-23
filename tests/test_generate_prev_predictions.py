import csv
import json
from pathlib import Path
from types import SimpleNamespace

import torch


def _write_metadata(path: Path, rows: list[dict[str, str]]):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["VIDEO_NAME", "SENTENCE_NAME", "SENTENCE", "PREV_SENTENCE", "START_REALIGNED"],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_generate_predictions_store_previous_sentence_prediction_per_row(tmp_path, monkeypatch):
    from scripts import generate_prev_predictions

    metadata_path = tmp_path / "metadata.csv"
    _write_metadata(
        metadata_path,
        [
            {"VIDEO_NAME": "vid1", "SENTENCE_NAME": "s1", "SENTENCE": "a", "PREV_SENTENCE": "", "START_REALIGNED": "0"},
            {"VIDEO_NAME": "vid1", "SENTENCE_NAME": "s2", "SENTENCE": "b", "PREV_SENTENCE": "a", "START_REALIGNED": "1"},
            {"VIDEO_NAME": "vid1", "SENTENCE_NAME": "s3", "SENTENCE": "c", "PREV_SENTENCE": "b", "START_REALIGNED": "2"},
        ],
    )

    calls = []

    class FakeModel:
        def generate(self, features, mask, prev_texts, max_new_tokens=128):
            calls.append(prev_texts[0])
            return [f"pred-{len(calls)}"]

    monkeypatch.setattr(generate_prev_predictions, "load_model", lambda args: FakeModel())
    monkeypatch.setattr(
        generate_prev_predictions,
        "read_features",
        lambda lmdb_path, sentence_name: torch.ones(2, 768),
    )

    output_path = tmp_path / "predictions.json"
    args = generate_prev_predictions.parse_args([
        "--lmdb", "ignored.lmdb",
        "--metadata", str(metadata_path),
        "--checkpoint", "ignored.pt",
        "--pretrained-llm", "ignored-llm",
        "--output", str(output_path),
    ])

    generate_prev_predictions.run(args)

    assert calls == ["", "pred-1", "pred-2"]
    assert json.loads(output_path.read_text()) == {
        "s1": "",
        "s2": "pred-1",
        "s3": "pred-2",
    }


def test_generate_predictions_resets_prev_text_per_video(tmp_path, monkeypatch):
    from scripts import generate_prev_predictions

    metadata_path = tmp_path / "metadata.csv"
    _write_metadata(
        metadata_path,
        [
            {"VIDEO_NAME": "vid1", "SENTENCE_NAME": "s1", "SENTENCE": "a", "PREV_SENTENCE": "", "START_REALIGNED": "0"},
            {"VIDEO_NAME": "vid1", "SENTENCE_NAME": "s2", "SENTENCE": "b", "PREV_SENTENCE": "a", "START_REALIGNED": "1"},
            {"VIDEO_NAME": "vid2", "SENTENCE_NAME": "s3", "SENTENCE": "c", "PREV_SENTENCE": "", "START_REALIGNED": "0"},
        ],
    )

    calls = []

    class FakeModel:
        def generate(self, features, mask, prev_texts, max_new_tokens=128):
            calls.append(prev_texts[0])
            return [f"pred-{len(calls)}"]

    monkeypatch.setattr(generate_prev_predictions, "load_model", lambda args: FakeModel())
    monkeypatch.setattr(
        generate_prev_predictions,
        "read_features",
        lambda lmdb_path, sentence_name: torch.ones(2, 768),
    )

    output_path = tmp_path / "predictions.json"
    args = generate_prev_predictions.parse_args([
        "--lmdb", "ignored.lmdb",
        "--metadata", str(metadata_path),
        "--checkpoint", "ignored.pt",
        "--pretrained-llm", "ignored-llm",
        "--output", str(output_path),
    ])

    generate_prev_predictions.run(args)

    assert calls == ["", "pred-1", ""]
    assert json.loads(output_path.read_text()) == {
        "s1": "",
        "s2": "pred-1",
        "s3": "",
    }


def test_generate_predictions_resets_prev_text_after_missing_features(tmp_path, monkeypatch):
    from scripts import generate_prev_predictions

    metadata_path = tmp_path / "metadata.csv"
    _write_metadata(
        metadata_path,
        [
            {"VIDEO_NAME": "vid1", "SENTENCE_NAME": "s1", "SENTENCE": "a", "PREV_SENTENCE": "", "START_REALIGNED": "0"},
            {"VIDEO_NAME": "vid1", "SENTENCE_NAME": "s2", "SENTENCE": "b", "PREV_SENTENCE": "a", "START_REALIGNED": "1"},
            {"VIDEO_NAME": "vid1", "SENTENCE_NAME": "s3", "SENTENCE": "c", "PREV_SENTENCE": "b", "START_REALIGNED": "2"},
        ],
    )

    calls = []

    class FakeModel:
        def generate(self, features, mask, prev_texts, max_new_tokens=128):
            calls.append(prev_texts[0])
            return [f"pred-{len(calls)}"]

    def fake_read_features(lmdb_path, sentence_name):
        if sentence_name == "s2":
            return torch.empty(0, 768)
        return torch.ones(2, 768)

    monkeypatch.setattr(generate_prev_predictions, "load_model", lambda args: FakeModel())
    monkeypatch.setattr(generate_prev_predictions, "read_features", fake_read_features)

    output_path = tmp_path / "predictions.json"
    args = generate_prev_predictions.parse_args([
        "--lmdb", "ignored.lmdb",
        "--metadata", str(metadata_path),
        "--checkpoint", "ignored.pt",
        "--pretrained-llm", "ignored-llm",
        "--output", str(output_path),
    ])

    generate_prev_predictions.run(args)

    assert calls == ["", ""]
    assert json.loads(output_path.read_text()) == {
        "s1": "",
        "s2": "pred-1",
        "s3": "",
    }


def test_generate_predictions_moves_inputs_to_requested_device(tmp_path, monkeypatch):
    from scripts import generate_prev_predictions

    metadata_path = tmp_path / "metadata.csv"
    _write_metadata(
        metadata_path,
        [
            {"VIDEO_NAME": "vid1", "SENTENCE_NAME": "s1", "SENTENCE": "a", "PREV_SENTENCE": "", "START_REALIGNED": "0"},
        ],
    )

    class FakeModel:
        def generate(self, features, mask, prev_texts, max_new_tokens=128):
            assert features.device.type == "meta"
            assert mask.device.type == "meta"
            return ["pred-meta"]

    monkeypatch.setattr(generate_prev_predictions, "load_model", lambda args: FakeModel())
    monkeypatch.setattr(
        generate_prev_predictions,
        "read_features",
        lambda lmdb_path, sentence_name: torch.ones(2, 768),
    )

    output_path = tmp_path / "predictions.json"
    args = generate_prev_predictions.parse_args([
        "--lmdb", "ignored.lmdb",
        "--metadata", str(metadata_path),
        "--checkpoint", "ignored.pt",
        "--pretrained-llm", "ignored-llm",
        "--output", str(output_path),
        "--device", "meta",
    ])

    generate_prev_predictions.run(args)

    assert json.loads(output_path.read_text()) == {"s1": ""}


def test_load_model_uses_lora_args_saved_in_checkpoint(tmp_path, monkeypatch):
    from scripts import generate_prev_predictions

    captured = {}
    checkpoint_data = {
        "args": {"lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05},
    }
    load_checkpoint_calls = []

    class DummyModel:
        def to(self, device):
            return self

        def eval(self):
            return self

    def fake_model_ctor(**kwargs):
        captured.update(kwargs)
        return DummyModel()

    monkeypatch.setattr(
        generate_prev_predictions.torch,
        "load",
        lambda path, map_location, weights_only=False: checkpoint_data,
    )
    monkeypatch.setattr(generate_prev_predictions, "VideoPrevLLM", fake_model_ctor)
    monkeypatch.setattr(generate_prev_predictions, "build_optimizer", lambda model, lr: object())
    monkeypatch.setattr(
        generate_prev_predictions,
        "load_checkpoint",
        lambda model, optimizer, checkpoint_path, device, checkpoint_data=None: load_checkpoint_calls.append(checkpoint_data),
    )

    generate_prev_predictions.load_model(
        SimpleNamespace(
            checkpoint=str(tmp_path / "checkpoint.pt"),
            pretrained_llm="fake-llm",
            device="cpu",
        )
    )

    assert captured == {
        "pretrained_llm": "fake-llm",
        "use_lora": True,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
    }
    assert load_checkpoint_calls == [checkpoint_data]


def test_read_features_returns_empty_tensor_when_sentence_missing(tmp_path):
    import lmdb
    import numpy as np

    from scripts.generate_prev_predictions import read_features

    lmdb_path = tmp_path / "features.lmdb"
    env = lmdb.open(str(lmdb_path), map_size=1 << 20, subdir=False)
    with env.begin(write=True) as txn:
        txn.put(b"present/0000000.np", np.ones(768, dtype=np.float16).tobytes())
    env.close()

    missing = read_features(str(lmdb_path), "missing")
    present = read_features(str(lmdb_path), "present")

    assert missing.shape == (0, 768)
    assert present.shape == (1, 768)
