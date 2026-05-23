"""How2Sign LMDB feature dataset for training."""
import csv
import lmdb
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset


FEATURE_DIM = 768


class How2SignLMDBDataset(Dataset):
    """Loads sentence-level VideoMAE features from LMDB and targets from metadata."""

    def __init__(self, lmdb_path: str, metadata_path: str, include_prev_text: bool = False):
        super().__init__()
        self.include_prev_text = include_prev_text
        raw = self._load_metadata(metadata_path)
        self.env = lmdb.open(lmdb_path, readonly=True, subdir=False, lock=False,
                             max_readers=126, meminit=False)
        self.metadata = [r for r in raw if self._has_features(r["SENTENCE_NAME"])]

    def _load_metadata(self, path: str):
        with open(path, "r", newline="") as f:
            return list(csv.DictReader(f))

    def _has_features(self, sentence_name: str) -> bool:
        with self.env.begin() as txn:
            for start_idx in (0, 1):
                key = f"{sentence_name}/{start_idx:07d}.np".encode("ascii")
                if txn.get(key) is not None:
                    return True
        return False

    def _read_features(self, sentence_name: str):
        with self.env.begin() as txn:
            for start_idx in (0, 1):
                key = f"{sentence_name}/{start_idx:07d}.np".encode("ascii")
                value = txn.get(key)
                if value is not None:
                    break
            else:
                return torch.empty(0, FEATURE_DIM)

            features = []
            idx = start_idx
            while True:
                key = f"{sentence_name}/{idx:07d}.np".encode("ascii")
                value = txn.get(key)
                if value is None:
                    break
                arr = np.frombuffer(value, dtype=np.float16).reshape(FEATURE_DIM)
                features.append(torch.from_numpy(arr.astype(np.float32)))
                idx += 1

            return torch.stack(features, dim=0) if features else torch.empty(0, FEATURE_DIM)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata[idx]
        name = row["SENTENCE_NAME"]
        feats = self._read_features(name)

        item = {
            "sentence_name": name,
            "features": feats.half(),
            "attention_mask": torch.ones(feats.shape[0], dtype=torch.bool),
            "target_text": row["SENTENCE"],
        }
        if self.include_prev_text:
            item["prev_text"] = row.get("PREV_SENTENCE") or ""
        return item

    def __del__(self):
        if hasattr(self, "env"):
            self.env.close()


def collate_variable_features(samples: list):
    """Pad features and attention masks to the longest sequence in the batch."""
    features = [s["features"] for s in samples]
    masks = [s["attention_mask"] for s in samples]
    texts = [s["target_text"] for s in samples]
    names = [s["sentence_name"] for s in samples]

    max_len = max(f.shape[0] for f in features)
    padded_feats = torch.zeros(len(features), max_len, FEATURE_DIM, dtype=torch.float16)
    padded_masks = torch.zeros(len(features), max_len, dtype=torch.bool)

    for i, (f, m) in enumerate(zip(features, masks)):
        t = f.shape[0]
        padded_feats[i, :t] = f
        padded_masks[i, :t] = m

    batch = {
        "features": padded_feats,
        "attention_mask": padded_masks,
        "target_texts": texts,
        "sentence_names": names,
    }
    if "prev_text" in samples[0]:
        batch["prev_texts"] = [s.get("prev_text", "") for s in samples]
    return batch
