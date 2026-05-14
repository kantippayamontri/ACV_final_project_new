"""Tests for How2Sign LMDB dataset loader."""
import csv
import lmdb
import torch
import pytest
import numpy as np
from pathlib import Path


def _write_metadata(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["SENTENCE_NAME", "SENTENCE", "PREV_SENTENCE",
                                                "VIDEO_NAME", "START_REALIGNED", "END_REALIGNED"])
        writer.writeheader()
        writer.writerows(rows)


def _open_lmdb(path, map_size=10 * 1024 ** 2):
    return lmdb.open(str(path), subdir=False, map_size=map_size, readonly=False, meminit=False)


class TestHow2SignLMDBDataset:
    def test_loads_single_sample(self, tmp_path):
        from src.data.how2sign_lmdb_dataset import How2SignLMDBDataset

        env = _open_lmdb(tmp_path / "test.lmdb")
        name = "clip_a"
        feats = torch.randn(3, 768).half()
        with env.begin(write=True) as txn:
            for i in range(3):
                txn.put(f"{name}/{i:07d}.np".encode("ascii"), feats[i].numpy().tobytes())
            txn.put(f"{name}/done".encode("ascii"), b"1")
        env.close()

        meta_path = tmp_path / "meta.csv"
        _write_metadata(meta_path, [{"SENTENCE_NAME": name, "SENTENCE": "hello world",
                                      "PREV_SENTENCE": "", "VIDEO_NAME": "v1",
                                      "START_REALIGNED": "0.0", "END_REALIGNED": "1.0"}])

        ds = How2SignLMDBDataset(lmdb_path=str(tmp_path / "test.lmdb"), metadata_path=str(meta_path))
        sample = ds[0]
        assert sample["sentence_name"] == name
        assert sample["target_text"] == "hello world"
        assert sample["features"].shape == (3, 768)
        assert sample["features"].dtype == torch.float16
        assert sample["attention_mask"].tolist() == [True, True, True]

    def test_loads_sample_with_legacy_one_based_keys(self, tmp_path):
        from src.data.how2sign_lmdb_dataset import How2SignLMDBDataset

        env = _open_lmdb(tmp_path / "test.lmdb")
        name = "legacy"
        feats = torch.randn(4, 768).half()
        with env.begin(write=True) as txn:
            for i in range(4):
                txn.put(f"{name}/{i+1:07d}.np".encode("ascii"), feats[i].numpy().tobytes())
            txn.put(f"{name}/done".encode("ascii"), b"1")
        env.close()

        meta_path = tmp_path / "meta.csv"
        _write_metadata(meta_path, [{"SENTENCE_NAME": name, "SENTENCE": "legacy text",
                                      "PREV_SENTENCE": "", "VIDEO_NAME": "v2",
                                      "START_REALIGNED": "0.0", "END_REALIGNED": "1.0"}])

        ds = How2SignLMDBDataset(lmdb_path=str(tmp_path / "test.lmdb"), metadata_path=str(meta_path))
        sample = ds[0]
        assert sample["features"].shape == (4, 768)

    def test_skips_clip_with_no_features(self, tmp_path):
        from src.data.how2sign_lmdb_dataset import How2SignLMDBDataset

        env = _open_lmdb(tmp_path / "test.lmdb")
        name = "empty_clip"
        with env.begin(write=True) as txn:
            txn.put(f"{name}/done".encode("ascii"), b"1")
        env.close()

        meta_path = tmp_path / "meta.csv"
        _write_metadata(meta_path, [{"SENTENCE_NAME": name, "SENTENCE": "no features",
                                      "PREV_SENTENCE": "", "VIDEO_NAME": "v3",
                                      "START_REALIGNED": "0.0", "END_REALIGNED": "1.0"}])

        ds = How2SignLMDBDataset(lmdb_path=str(tmp_path / "test.lmdb"), metadata_path=str(meta_path))
        assert len(ds) == 0

    def test_collate_pads_variable_lengths(self):
        from src.data.how2sign_lmdb_dataset import collate_variable_features

        samples = [
            {"features": torch.randn(3, 768), "attention_mask": torch.ones(3, dtype=torch.bool),
             "target_text": "a", "sentence_name": "s1"},
            {"features": torch.randn(5, 768), "attention_mask": torch.ones(5, dtype=torch.bool),
             "target_text": "b", "sentence_name": "s2"},
            {"features": torch.randn(2, 768), "attention_mask": torch.ones(2, dtype=torch.bool),
             "target_text": "c", "sentence_name": "s3"},
        ]

        batch = collate_variable_features(samples)
        assert batch["features"].shape == (3, 5, 768)
        assert batch["attention_mask"].shape == (3, 5)
        assert batch["attention_mask"][0].tolist() == [True, True, True, False, False]
        assert batch["attention_mask"][2].tolist() == [True, True, False, False, False]
        assert batch["target_texts"] == ["a", "b", "c"]
        assert batch["sentence_names"] == ["s1", "s2", "s3"]

    def test_collate_handles_single_sample(self):
        from src.data.how2sign_lmdb_dataset import collate_variable_features

        samples = [
            {"features": torch.randn(4, 768), "attention_mask": torch.ones(4, dtype=torch.bool),
             "target_text": "x", "sentence_name": "sx"},
        ]
        batch = collate_variable_features(samples)
        assert batch["features"].shape == (1, 4, 768)

    def test_constructor_uses_metadata_path_not_metadata_csv(self, tmp_path):
        from src.data.how2sign_lmdb_dataset import How2SignLMDBDataset

        env = _open_lmdb(tmp_path / "test.lmdb")
        env.close()

        meta_path = tmp_path / "meta.csv"
        _write_metadata(meta_path, [])

        with pytest.raises(TypeError, match="unexpected keyword argument"):
            How2SignLMDBDataset(lmdb_path=str(tmp_path / "test.lmdb"), metadata_csv=str(meta_path))

    def test_dataset_configured_for_multiple_readers(self, tmp_path):
        from src.data.how2sign_lmdb_dataset import How2SignLMDBDataset

        env = _open_lmdb(tmp_path / "test.lmdb")
        name = "clip_a"
        feats = torch.randn(3, 768).half()
        with env.begin(write=True) as txn:
            for i in range(3):
                txn.put(f"{name}/{i:07d}.np".encode("ascii"), feats[i].numpy().tobytes())
            txn.put(f"{name}/done".encode("ascii"), b"1")
        env.close()

        meta_path = tmp_path / "meta.csv"
        _write_metadata(meta_path, [{"SENTENCE_NAME": name, "SENTENCE": "hello world",
                                      "PREV_SENTENCE": "", "VIDEO_NAME": "v1",
                                      "START_REALIGNED": "0.0", "END_REALIGNED": "1.0"}])

        from torch.utils.data import DataLoader
        from src.data.how2sign_lmdb_dataset import collate_variable_features

        ds = How2SignLMDBDataset(lmdb_path=str(tmp_path / "test.lmdb"), metadata_path=str(meta_path))
        loader = DataLoader(ds, batch_size=1, collate_fn=collate_variable_features, num_workers=2)
        batches = list(loader)
        assert len(batches) == 1
