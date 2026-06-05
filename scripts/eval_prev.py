#!/usr/bin/env python3
"""Evaluate a saved VideoPrevLLM checkpoint with autoregressive previous-sentence context.

Usage:
    PYTHONPATH=. uv run python scripts/eval_prev.py \\
        --lmdb features/val_features.lmdb \\
        --metadata features/val_metadata.csv \\
        --checkpoint outputs/video_prev_run/best.pt \\
        --pretrained-llm models/Llama-3.2-1B \\
        --output-dir outputs/video_prev_run/eval
"""
import argparse
import csv
from pathlib import Path

import lmdb
import numpy as np
import torch
from tqdm import tqdm

from src.models.visual_prev_llm import VideoPrevLLM
from src.training.train_utils import build_optimizer, load_checkpoint, save_metrics_json, save_predictions_jsonl
from src.training.metrics import compute_metrics


FEATURE_DIM = 768


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate VideoPrevLLM checkpoint")
    parser.add_argument("--lmdb", type=str, required=True)
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--pretrained-llm", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def load_model(args):
    checkpoint_path = Path(args.checkpoint)
    checkpoint_data = torch.load(str(checkpoint_path), map_location=args.device, weights_only=False)
    train_args = checkpoint_data.get("args", {})
    model = VideoPrevLLM(
        pretrained_llm=args.pretrained_llm,
        use_lora=True,
        load_in_4bit=True,
        lora_r=train_args.get("lora_r", 8),
        lora_alpha=train_args.get("lora_alpha", 16),
        lora_dropout=train_args.get("lora_dropout", 0.1),
        projector_layers=train_args.get("projector_layers", 3),
    )
    optimizer = build_optimizer(model, lr=0.0)
    epoch, metrics, checkpoint_args = load_checkpoint(
        model, optimizer, checkpoint_path, args.device, checkpoint_data=checkpoint_data,
    )
    model = model.to(args.device)
    model.eval()
    print(f"Loaded checkpoint from epoch {epoch}, BLEU: {metrics.get('BLEU', 'N/A')}")
    return model


def read_features(env, sentence_name: str):
    with env.begin() as txn:
        for start_idx in (0, 1):
            key = f"{sentence_name}/{start_idx:07d}.np".encode("ascii")
            if txn.get(key) is not None:
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


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)

    print("Loading model...")
    model = load_model(args)
    print("Model loaded.")

    with open(args.metadata, "r", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError("Evaluation dataset is empty")

    env = lmdb.open(args.lmdb, readonly=True, subdir=False, lock=False, max_readers=126, meminit=False)
    try:
        all_predictions = []
        all_references = []
        previous_video = None
        previous_prediction = ""

        for row in tqdm(rows, desc="Evaluating"):
            if row["VIDEO_NAME"] != previous_video:
                previous_video = row["VIDEO_NAME"]
                previous_prediction = ""

            target = row["SENTENCE"]
            all_references.append([target])

            features = read_features(env, row["SENTENCE_NAME"])
            if features.numel() == 0:
                all_predictions.append("")
                previous_prediction = ""
                continue

            features = features.to(args.device)
            mask = torch.ones(features.shape[0], dtype=torch.bool, device=args.device)
            prediction = model.generate(
                features.unsqueeze(0),
                mask.unsqueeze(0),
                [previous_prediction],
                max_new_tokens=args.max_new_tokens,
            )[0]
            all_predictions.append(prediction)
            previous_prediction = prediction

        eval_metrics = compute_metrics(all_predictions, all_references)
        print(f"\nBLEU: {eval_metrics['BLEU']:.2f}  ROUGE-L: {eval_metrics['ROUGE-L']:.2f}")

    finally:
        env.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    save_predictions_jsonl(all_predictions, all_references, output_dir / "predictions.jsonl")
    save_metrics_json(eval_metrics, output_dir / "metrics.json")
    print(f"Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
