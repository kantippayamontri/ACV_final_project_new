#!/usr/bin/env python3
"""Generate previous-sentence predictions in video order.

Usage:
    PYTHONPATH=. uv run python scripts/generate_prev_predictions.py \\
        --lmdb features/train_features.lmdb \\
        --metadata features/train_metadata.csv \\
        --checkpoint outputs/video_prev_run/best.pt \\
        --pretrained-llm models/Llama-3.2-1B \\
        --output outputs/video_prev_run/train_prev_predictions.json

    # With 4-bit quantization for large models (8B):
    PYTHONPATH=. uv run python scripts/generate_prev_predictions.py \\
        --lmdb features/train_features.lmdb \\
        --metadata features/train_metadata.csv \\
        --checkpoint outputs/video_prev_run/best.pt \\
        --pretrained-llm models/Meta-Llama-3-8B \\
        --output outputs/video_prev_run/train_prev_predictions.json \\
        --load-in-4bit
"""
import argparse
import csv
import json
from pathlib import Path

import lmdb
import numpy as np
import torch
from tqdm import tqdm

from src.models.visual_prev_llm import VideoPrevLLM
from src.training.train_utils import build_optimizer, load_checkpoint


FEATURE_DIM = 768


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate previous-sentence predictions")
    parser.add_argument("--lmdb", type=str, required=True)
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--pretrained-llm", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Load LLM in 4-bit quantization (reduces VRAM for 8B+ models)")
    return parser.parse_args(argv)


def load_model(args):
    checkpoint_path = Path(args.checkpoint)
    checkpoint_data = torch.load(str(checkpoint_path), map_location=args.device, weights_only=False)
    train_args = checkpoint_data.get("args", {})
    model = VideoPrevLLM(
        pretrained_llm=args.pretrained_llm,
        use_lora=True,
        load_in_4bit=args.load_in_4bit,
        lora_r=train_args.get("lora_r", 8),
        lora_alpha=train_args.get("lora_alpha", 16),
        lora_dropout=train_args.get("lora_dropout", 0.1),
    )
    optimizer = build_optimizer(model, lr=2e-4)
    load_checkpoint(model, optimizer, checkpoint_path, args.device, checkpoint_data=checkpoint_data)
    model = model.to(args.device)
    model.eval()
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


def run(args):
    print("Loading model...")
    model = load_model(args)
    print("Model loaded.")

    with open(args.metadata, "r", newline="") as f:
        rows = list(csv.DictReader(f))

    env = lmdb.open(args.lmdb, readonly=True, subdir=False, lock=False, max_readers=126, meminit=False)
    try:
        predictions = {}
        previous_video = None
        previous_prediction = ""

        for row in tqdm(rows, desc="Generating predictions"):
            if row["VIDEO_NAME"] != previous_video:
                previous_video = row["VIDEO_NAME"]
                previous_prediction = ""

            predictions[row["SENTENCE_NAME"]] = previous_prediction

            features = read_features(env, row["SENTENCE_NAME"])
            if features.numel() == 0:
                previous_prediction = ""
                continue

            features = features.to(args.device)
            mask = torch.ones(features.shape[0], dtype=torch.bool, device=args.device)
            output = model.generate(
                features.unsqueeze(0),
                mask.unsqueeze(0),
                [previous_prediction],
                max_new_tokens=args.max_new_tokens,
            )[0]
            previous_prediction = output

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(predictions, indent=2))
        print(f"Saved {len(predictions)} predictions to {args.output}")
    finally:
        env.close()


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
