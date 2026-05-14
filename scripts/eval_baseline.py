#!/usr/bin/env python3
"""Evaluate a saved visual-only baseline checkpoint."""
import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.data.how2sign_lmdb_dataset import How2SignLMDBDataset, collate_variable_features
from src.models.visual_llm_baseline import VisualLLMBaseline
from src.training.train_utils import load_checkpoint, build_optimizer, save_metrics_json, save_predictions_jsonl
from src.training.metrics import compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate visual-only baseline checkpoint")
    parser.add_argument("--lmdb", type=str, required=True)
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--pretrained-llm", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)

    print(f"Loading dataset from {args.lmdb}...")
    dataset = How2SignLMDBDataset(lmdb_path=args.lmdb, metadata_path=args.metadata)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_variable_features, num_workers=args.num_workers,
    )
    print(f"Samples: {len(dataset)}")

    print(f"Loading LLM from {args.pretrained_llm}...")
    model = VisualLLMBaseline(pretrained_llm=args.pretrained_llm, use_lora=True)
    model = model.to(args.device)

    optimizer = build_optimizer(model, lr=0.0)
    checkpoint_path = Path(args.checkpoint)
    epoch, metrics, train_args = load_checkpoint(model, optimizer, checkpoint_path, args.device)
    print(f"Loaded checkpoint from epoch {epoch}, BLEU: {metrics.get('BLEU', 'N/A')}")

    model.eval()
    all_predictions = []
    all_references = []

    for batch in tqdm(loader, desc="Evaluating"):
        features = batch["features"].to(args.device)
        mask = batch["attention_mask"].to(args.device)
        targets = batch["target_texts"]

        predictions = model.generate(features, mask, max_new_tokens=args.max_new_tokens)
        all_predictions.extend(predictions)
        all_references.extend([[t] for t in targets])

    eval_metrics = compute_metrics(all_predictions, all_references)
    print(f"\nBLEU: {eval_metrics['BLEU']:.2f}  ROUGE-L: {eval_metrics['ROUGE-L']:.2f}")

    save_predictions_jsonl(all_predictions, all_references, output_dir / "predictions.jsonl")
    save_metrics_json(eval_metrics, output_dir / "metrics.json")
    print(f"Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
