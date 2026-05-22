#!/usr/bin/env python3
"""Train VisualLLM baseline using PyTorch Lightning."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from src.data.how2sign_datamodule import How2SignDataModule
from src.models.visual_llm_litmodule import VisualLLMLitModule


def parse_args():
    parser = argparse.ArgumentParser(description="Train VisualLLM baseline with Lightning")
    parser.add_argument("--train-lmdb", type=str, required=True)
    parser.add_argument("--train-metadata", type=str, required=True)
    parser.add_argument("--val-lmdb", type=str, required=True)
    parser.add_argument("--val-metadata", type=str, required=True)
    parser.add_argument("--pretrained-llm", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--val-batch-size", type=int, default=None,
                        help="Validation batch size (defaults to --batch-size)")
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--val-samples", type=int, default=None)
    parser.add_argument("--devices", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datamodule = How2SignDataModule(
        train_lmdb=args.train_lmdb,
        train_metadata=args.train_metadata,
        val_lmdb=args.val_lmdb,
        val_metadata=args.val_metadata,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        val_samples=args.val_samples,
    )

    litmodule = VisualLLMLitModule(
        pretrained_llm=args.pretrained_llm,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lr=args.lr,
        max_new_tokens=args.max_new_tokens,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="best-{epoch:02d}-{val_bleu:.2f}",
        monitor="val_bleu",
        mode="max",
        save_top_k=1,
        save_last=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    logger = TensorBoardLogger(
        save_dir=str(output_dir),
        name="logs",
        default_hp_metric=True,
    )

    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices,
        accumulate_grad_batches=args.grad_accum_steps,
        precision=args.precision,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        log_every_n_steps=1,
        val_check_interval=1.0,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    trainer.fit(litmodule, datamodule=datamodule)

    print(f"\nTraining complete. Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
