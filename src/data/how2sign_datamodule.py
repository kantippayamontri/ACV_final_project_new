"""PyTorch Lightning DataModule for How2Sign dataset."""
import random
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Subset
from src.data.how2sign_lmdb_dataset import How2SignLMDBDataset, collate_variable_features


class How2SignDataModule(LightningDataModule):
    """LightningDataModule for How2Sign LMDB dataset."""

    def __init__(
        self,
        train_lmdb: str,
        train_metadata: str,
        val_lmdb: str,
        val_metadata: str,
        batch_size: int = 2,
        val_batch_size: int | None = None,
        num_workers: int = 0,
        val_samples: int | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.train_lmdb = train_lmdb
        self.train_metadata = train_metadata
        self.val_lmdb = val_lmdb
        self.val_metadata = val_metadata
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or batch_size
        self.num_workers = num_workers
        self.val_samples = val_samples

        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: str | None = None):
        if stage == "fit" or stage is None:
            self.train_dataset = How2SignLMDBDataset(
                lmdb_path=self.train_lmdb,
                metadata_path=self.train_metadata,
            )

            self.val_dataset = How2SignLMDBDataset(
                lmdb_path=self.val_lmdb,
                metadata_path=self.val_metadata,
            )

            if self.val_samples:
                indices = random.sample(
                    range(len(self.val_dataset)),
                    min(self.val_samples, len(self.val_dataset)),
                )
                self.val_dataset = Subset(self.val_dataset, indices)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_variable_features,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            collate_fn=collate_variable_features,
            num_workers=self.num_workers,
        )
