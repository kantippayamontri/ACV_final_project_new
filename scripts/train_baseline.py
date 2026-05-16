#!/usr/bin/env python3
"""Train the visual-only baseline: VideoMAE features -> projector -> LoRA LLM decoder."""
import argparse
from contextlib import nullcontext
import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.data.how2sign_lmdb_dataset import How2SignLMDBDataset, collate_variable_features
from src.models.visual_llm_baseline import VisualLLMBaseline
from src.training.train_utils import (
    set_seed, build_optimizer, mixed_precision_context,
    save_checkpoint, save_metrics_json, save_predictions_jsonl,
)
from src.training.metrics import compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train visual-only baseline")
    parser.add_argument("--train-lmdb", type=str, required=True)
    parser.add_argument("--train-metadata", type=str, required=True)
    parser.add_argument("--val-lmdb", type=str, required=True)
    parser.add_argument("--val-metadata", type=str, required=True)
    parser.add_argument("--pretrained-llm", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", type=str, default="fp16")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.add_argument("--val-samples", type=int, default=None,
                        help="Limit validation set size for faster eval")
    return parser.parse_args()


def train_epoch(model, loader, optimizer, scaler, autocast, grad_accum_steps, device):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(loader, desc="Train", leave=False)):
        features = batch["features"].to(device)
        mask = batch["attention_mask"].to(device)
        targets = batch["target_texts"]

        with (autocast or nullcontext()):
            loss = model.forward(features, mask, targets)
            loss = loss / grad_accum_steps

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % grad_accum_steps == 0:
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps

    if len(loader) > 0 and len(loader) % grad_accum_steps != 0:
        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

    return total_loss / max(len(loader), 1)


def validate_epoch(model, loader, max_new_tokens, device):
    model.eval()
    all_predictions = []
    all_references = []

    for batch in tqdm(loader, desc="Val", leave=False):
        features = batch["features"].to(device)
        mask = batch["attention_mask"].to(device)
        targets = batch["target_texts"]

        predictions = model.generate(features, mask, max_new_tokens=max_new_tokens)
        all_predictions.extend(predictions)
        all_references.extend([[t] for t in targets])

    metrics = compute_metrics(all_predictions, all_references)
    return metrics, all_predictions, all_references


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)

    print(f"Loading datasets...")
    train_dataset = How2SignLMDBDataset(
        lmdb_path=args.train_lmdb, metadata_path=args.train_metadata,
    )
    val_dataset = How2SignLMDBDataset(
        lmdb_path=args.val_lmdb, metadata_path=args.val_metadata,
    )
    if args.val_samples:
        import random
        indices = random.sample(range(len(val_dataset)), min(args.val_samples, len(val_dataset)))
        val_dataset = torch.utils.data.Subset(val_dataset, indices)
    if len(val_dataset) == 0:
        raise ValueError("Validation dataset is empty")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_variable_features, num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_variable_features, num_workers=args.num_workers,
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    print(f"Loading LLM from {args.pretrained_llm}...")
    model = VisualLLMBaseline(
        pretrained_llm=args.pretrained_llm,
        use_lora=True,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    print(f"LLM hidden size: {model.llm_hidden_size}")
    model = model.to(args.device)

    optimizer = build_optimizer(model, lr=args.lr)
    autocast, scaler = mixed_precision_context(args.precision, args.device)

    best_bleu = 0.0
    all_metrics = []

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")

        train_loss = train_epoch(model, train_loader, optimizer, scaler, autocast,
                                 args.grad_accum_steps, args.device)
        print(f"Train loss: {train_loss:.4f}")

        val_metrics, val_preds, val_refs = validate_epoch(
            model, val_loader, args.max_new_tokens, args.device,
        )
        print(f"Val BLEU: {val_metrics['BLEU']:.2f}  ROUGE-L: {val_metrics['ROUGE-L']:.2f}")

        epoch_metrics = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
        all_metrics.append(epoch_metrics)

        if val_metrics["BLEU"] > best_bleu:
            best_bleu = val_metrics["BLEU"]
            save_checkpoint(model, optimizer, epoch, val_metrics,
                           vars(args), output_dir / "best.pt")

        if args.save_every_epoch:
            save_checkpoint(model, optimizer, epoch, val_metrics,
                           vars(args), output_dir / f"epoch_{epoch}.pt")

        save_predictions_jsonl(val_preds, val_refs, output_dir / f"val_predictions_epoch{epoch}.jsonl")

    save_checkpoint(model, optimizer, args.epochs, all_metrics[-1],
                   vars(args), output_dir / "last.pt")
    save_metrics_json(all_metrics, output_dir / "metrics.json")
    print(f"\nDone. Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
