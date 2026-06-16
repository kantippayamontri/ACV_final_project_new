#!/usr/bin/env python3
"""Qualitative comparison of visual-only vs GT-prev-context vs mixed-context models.

Loads one model at a time to stay within VRAM limits.
Auto-selects representative examples by running the visual-only model on a pool
of validation samples and ranking by sentence-level BLEU.
Extracts sampled video frames as PNGs for the report figure.

Usage:
    PYTHONPATH=. uv run python scripts/qualitative_compare.py \
        --lmdb features/val_features.lmdb \
        --metadata features/val_metadata.csv \
        --vis-checkpoint outputs/video_only_llama32_3b/best.pt \
        --gt-prev-checkpoint outputs/video_prev_gt/best.pt \
        --mixed-prev-checkpoint outputs/video_prev_mixed/best.pt \
        --pretrained-llm models/Llama-3.2-3B \
        --video-dir datasets/val_rgb_front_clips/raw_videos \
        --output-dir outputs/qualitative_report \
        --num-examples 5 \
        --selection-samples 200 \
        --device cuda
"""
import argparse
import csv
import gc
import json
import random
import sys
from pathlib import Path

import lmdb
import numpy as np
import sacrebleu
import torch
from decord import VideoReader, cpu
from PIL import Image
from tqdm import tqdm

from src.models.visual_llm_baseline import VisualLLMBaseline
from src.models.visual_prev_llm import VideoPrevLLM
from src.training.train_utils import build_optimizer, load_checkpoint

FEATURE_DIM = 768
FRAME_SIZE = 224


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Qualitative comparison of 3 sign-language translation models"
    )
    parser.add_argument("--lmdb", type=str, required=True,
                        help="Path to LMDB feature store")
    parser.add_argument("--metadata", type=str, required=True,
                        help="Path to metadata CSV")
    parser.add_argument("--vis-checkpoint", type=str, required=True,
                        help="Checkpoint for visual-only baseline (VisualLLMBaseline)")
    parser.add_argument("--gt-prev-checkpoint", type=str, required=True,
                        help="Checkpoint for GT previous-context model (VideoPrevLLM, trained with --prev-gt-ratio 1.0)")
    parser.add_argument("--mixed-prev-checkpoint", type=str, required=True,
                        help="Checkpoint for mixed-context model (VideoPrevLLM, trained with --prev-gt-ratio 0.5)")
    parser.add_argument("--pretrained-llm", type=str, required=True,
                        help="Path to pretrained LLM (same backbone for all models)")
    parser.add_argument("--video-dir", type=str, default=None,
                        help="Directory containing <SENTENCE_NAME>.mp4 videos (for frame extraction)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory for output JSON and extracted frames")
    parser.add_argument("--num-examples", type=int, default=5,
                        help="Number of examples in the final report (default: 5)")
    parser.add_argument("--selection-samples", type=int, default=200,
                        help="Pool size for auto-selection (default: 200)")
    parser.add_argument("--num-frames", type=int, default=6,
                        help="Number of frames to sample per clip (default: 6)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Load LLM in 4-bit quantization (reduces VRAM)")
    parser.add_argument("--max-gpu-memory-gb", type=float, default=None,
                        help="Override the per-model GPU memory budget (GiB). "
                             "If unset, free VRAM is probed via torch.cuda.mem_get_info() "
                             "and multiplied by --gpu-memory-fraction.")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.9,
                        help="Fraction of free VRAM to allocate to the LLM (default: 0.9). "
                             "Only used when --max-gpu-memory-gb is unset.")
    parser.add_argument("--min-free-vram-gb", type=float, default=2.0,
                        help="Refuse to load a model if free VRAM (GiB) drops below this. "
                             "Default: 2.0. Set to 0 to disable the pre-check.")
    return parser.parse_args(argv)


def _has_features(txn, sentence_name: str) -> bool:
    for start_idx in (0, 1):
        key = f"{sentence_name}/{start_idx:07d}.np".encode("ascii")
        if txn.get(key) is not None:
            return True
    return False


def read_features(env, sentence_name: str) -> torch.Tensor:
    """Read all feature vectors for a sentence from LMDB. Returns empty tensor if missing."""
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


def extract_frames(video_dir: Path, sentence_name: str, output_dir: Path,
                   num_frames: int = 6) -> list[str]:
    """Sample num_frames evenly from a video clip, save as 224x224 PNGs.

    Returns list of output file paths.
    """
    video_path = video_dir / f"{sentence_name}.mp4"
    if not video_path.exists():
        print(f"  WARNING: video not found: {video_path}")
        return []

    vr = VideoReader(str(video_path), ctx=cpu(0))
    total = len(vr)
    if total == 0:
        return []

    if total <= num_frames:
        indices = list(range(total))
    else:
        indices = [int(i * (total - 1) / (num_frames - 1)) for i in range(num_frames)]

    frame_dir = output_dir / "frames" / sentence_name
    frame_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for i, idx in enumerate(indices):
        frame = Image.fromarray(vr[idx].asnumpy())
        frame = frame.resize((FRAME_SIZE, FRAME_SIZE), Image.BILINEAR)
        out_path = frame_dir / f"frame_{i:02d}.png"
        frame.save(str(out_path))
        paths.append(str(out_path))

    return paths


def _get_lora_config(train_args: dict) -> tuple:
    return (
        train_args.get("lora_r", 8),
        train_args.get("lora_alpha", 16),
        train_args.get("lora_dropout", 0.1),
    )


def report_free_vram(device: str) -> float | None:
    """Return currently free VRAM in GiB on the target device, or None if not CUDA."""
    if not torch.cuda.is_available() or device != "cuda":
        return None
    free_bytes, _ = torch.cuda.mem_get_info(0)
    return free_bytes / (1024 ** 3)


def check_vram_sufficient(args, label: str) -> None:
    """Raise a helpful error if free VRAM is too low to safely load the next model.

    Skips the check when the device is not CUDA, when --min-free-vram-gb <= 0,
    or when --load-in-4bit is set (4-bit quantisation is much smaller and the
    budget override already accounts for the small footprint).
    """
    if args.min_free_vram_gb <= 0 or args.load_in_4bit or args.device != "cuda":
        return
    free_gb = report_free_vram(args.device)
    if free_gb is None:
        return
    if free_gb < args.min_free_vram_gb:
        raise RuntimeError(
            f"Insufficient free VRAM before loading {label} model: "
            f"{free_gb:.2f} GiB free, --min-free-vram-gb={args.min_free_vram_gb:.2f} GiB. "
            f"Free up GPU memory (e.g. kill other processes, use --load-in-4bit, "
            f"or lower --min-free-vram-gb)."
        )
    print(f"  [VRAM] {label}: {free_gb:.2f} GiB free before load")


def load_visual_model(checkpoint_path: str, pretrained_llm: str,
                      device: str, load_in_4bit: bool,
                      max_gpu_memory_gb: float | None = None,
                      gpu_memory_fraction: float = 0.9) -> VisualLLMBaseline:
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    train_args = ckpt.get("args", {})
    lora_r, lora_alpha, lora_dropout = _get_lora_config(train_args)
    model = VisualLLMBaseline(
        pretrained_llm=pretrained_llm,
        use_lora=True,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        load_in_4bit=load_in_4bit,
        projector_layers=train_args.get("projector_layers", 3),
        max_gpu_memory_gb=max_gpu_memory_gb,
        gpu_memory_fraction=gpu_memory_fraction,
    )
    optimizer = build_optimizer(model, lr=0.0)
    epoch, metrics, _ = load_checkpoint(
        model, optimizer, Path(checkpoint_path), device, checkpoint_data=ckpt,
    )
    model = model.to(device)
    model.eval()
    print(f"Loaded visual-only model (epoch {epoch}, BLEU {metrics.get('BLEU', 'N/A')})")
    return model


def load_prev_model(checkpoint_path: str, pretrained_llm: str,
                    device: str, load_in_4bit: bool,
                    max_gpu_memory_gb: float | None = None,
                    gpu_memory_fraction: float = 0.9) -> VideoPrevLLM:
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    train_args = ckpt.get("args", {})
    lora_r, lora_alpha, lora_dropout = _get_lora_config(train_args)
    model = VideoPrevLLM(
        pretrained_llm=pretrained_llm,
        use_lora=True,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        load_in_4bit=load_in_4bit,
        projector_layers=train_args.get("projector_layers", 3),
        max_gpu_memory_gb=max_gpu_memory_gb,
        gpu_memory_fraction=gpu_memory_fraction,
    )
    optimizer = build_optimizer(model, lr=0.0)
    epoch, metrics, _ = load_checkpoint(
        model, optimizer, Path(checkpoint_path), device, checkpoint_data=ckpt,
    )
    model = model.to(device)
    model.eval()
    print(f"Loaded prev-context model (epoch {epoch}, BLEU {metrics.get('BLEU', 'N/A')})")
    return model


def unload_model(model):
    """Delete model and free GPU memory."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _score_sample(args, env, row, model):
    """Return (sentence_name, target_text, prev_text, video_name, bleu_score, pred)."""
    name = row["SENTENCE_NAME"]
    features = read_features(env, name)
    if features.numel() == 0:
        return None
    features = features.to(args.device)
    mask = torch.ones(1, features.shape[0], dtype=torch.bool, device=args.device)
    pred = model.generate(features.unsqueeze(0), mask,
                          max_new_tokens=args.max_new_tokens)[0]
    bleu = sacrebleu.sentence_bleu(pred, [row["SENTENCE"]]).score
    return (name, row["SENTENCE"], row.get("PREV_SENTENCE") or "",
            row["VIDEO_NAME"], bleu, pred)


def select_examples(args, all_rows: list[dict], env) -> list[dict]:
    """Auto-select examples by running visual-only model on a random pool.

    Returns list of dicts with keys:
        sentence_name, target_text, prev_text, video_name, bleu_visual_only, vis_pred_selection
    """
    with env.begin() as txn:
        valid_rows = [r for r in all_rows if _has_features(txn, r["SENTENCE_NAME"])]
    if not valid_rows:
        raise ValueError("No valid samples with features found")

    pool_size = min(args.selection_samples, len(valid_rows))
    if pool_size < args.num_examples:
        raise ValueError(
            f"Pool size {pool_size} is smaller than --num-examples {args.num_examples}. "
            f"Increase --selection-samples or use a larger dataset."
        )

    print(f"Selecting {args.num_examples} examples from pool of {pool_size}...")
    random.seed(args.seed)
    pool_rows = random.sample(valid_rows, pool_size)

    print("Loading visual-only model for example selection...")
    check_vram_sufficient(args, "visual-only (selection)")
    model = load_visual_model(
        args.vis_checkpoint, args.pretrained_llm,
        args.device, args.load_in_4bit,
        max_gpu_memory_gb=args.max_gpu_memory_gb,
        gpu_memory_fraction=args.gpu_memory_fraction,
    )

    scored = []
    for row in tqdm(pool_rows, desc="Scoring samples"):
        result = _score_sample(args, env, row, model)
        if result is not None:
            scored.append(result)

    unload_model(model)

    if len(scored) < args.num_examples:
        raise ValueError(
            f"Only {len(scored)} samples produced valid predictions. "
            f"Need at least {args.num_examples}."
        )

    scored.sort(key=lambda x: x[4])  # sort by BLEU ascending

    n = len(scored) - 1
    if args.num_examples == 5:
        percs = [0.0, 0.25, 0.50, 0.75, 1.0]
    else:
        step = n / max(args.num_examples - 1, 1)
        percs = [i * step / n for i in range(args.num_examples)]

    indices = []
    for p in percs[:args.num_examples]:
        idx = min(int(round(p * n)), n)
        if idx not in indices:
            indices.append(idx)

    while len(indices) < args.num_examples:
        for i in range(n + 1):
            if i not in indices:
                indices.append(i)
                break
    indices = sorted(indices[:args.num_examples])

    examples = []
    for idx in indices:
        name, target, prev, video_name, bleu, vis_pred = scored[idx]
        examples.append({
            "sentence_name": name,
            "target_text": target,
            "prev_text": prev,
            "video_name": video_name,
            "bleu_visual_only": bleu,
            "vis_pred_selection": vis_pred,
        })

    print(f"Selected BLEU range: {examples[0]['bleu_visual_only']:.1f} – "
          f"{examples[-1]['bleu_visual_only']:.1f}")
    return examples


def run_visual_model(args, examples: list[dict], env) -> dict:
    """Run visual-only model on the selected examples. Returns {name: prediction}."""
    print("\n=== Visual-only model ===")
    check_vram_sufficient(args, "visual-only")
    model = load_visual_model(
        args.vis_checkpoint, args.pretrained_llm,
        args.device, args.load_in_4bit,
        max_gpu_memory_gb=args.max_gpu_memory_gb,
        gpu_memory_fraction=args.gpu_memory_fraction,
    )

    predictions = {}
    for ex in tqdm(examples, desc="Visual-only inference"):
        name = ex["sentence_name"]
        features = read_features(env, name)
        if features.numel() == 0:
            predictions[name] = ""
            continue
        features = features.to(args.device)
        mask = torch.ones(1, features.shape[0], dtype=torch.bool, device=args.device)
        pred = model.generate(features.unsqueeze(0), mask,
                              max_new_tokens=args.max_new_tokens)[0]
        predictions[name] = pred

    unload_model(model)
    return predictions


def run_gt_prev_model(args, examples: list[dict], env) -> dict:
    """Run GT prev-context model. Feeds ground-truth PREV_SENTENCE as context."""
    print("\n=== GT previous-context model ===")
    check_vram_sufficient(args, "gt-prev")
    model = load_prev_model(
        args.gt_prev_checkpoint, args.pretrained_llm,
        args.device, args.load_in_4bit,
        max_gpu_memory_gb=args.max_gpu_memory_gb,
        gpu_memory_fraction=args.gpu_memory_fraction,
    )

    predictions = {}
    for ex in tqdm(examples, desc="GT-prev inference"):
        name = ex["sentence_name"]
        features = read_features(env, name)
        if features.numel() == 0:
            predictions[name] = ""
            continue
        features = features.to(args.device)
        mask = torch.ones(1, features.shape[0], dtype=torch.bool, device=args.device)
        prev = ex["prev_text"] or "(none)"
        pred = model.generate(features.unsqueeze(0), mask, [prev],
                              max_new_tokens=args.max_new_tokens)[0]
        predictions[name] = pred

    unload_model(model)
    return predictions


def run_mixed_prev_model(args, examples: list[dict], all_rows: list[dict],
                         env) -> dict:
    """Run mixed-context model with autoregressive previous-sentence context.

    Processes all rows in video order so the model sees its own predictions
    as previous-sentence context. Only keeps predictions for target examples.
    """
    print("\n=== Mixed previous-context model ===")
    check_vram_sufficient(args, "mixed-prev")
    model = load_prev_model(
        args.mixed_prev_checkpoint, args.pretrained_llm,
        args.device, args.load_in_4bit,
        max_gpu_memory_gb=args.max_gpu_memory_gb,
        gpu_memory_fraction=args.gpu_memory_fraction,
    )

    target_names = {ex["sentence_name"] for ex in examples}
    predictions = {}
    previous_video = None
    previous_prediction = ""

    for row in tqdm(all_rows, desc="Mixed-prev inference"):
        if row["VIDEO_NAME"] != previous_video:
            previous_video = row["VIDEO_NAME"]
            previous_prediction = ""

        name = row["SENTENCE_NAME"]
        features = read_features(env, name)

        if features.numel() == 0:
            if name in target_names:
                predictions[name] = ""
            previous_prediction = ""
            continue

        features = features.to(args.device)
        mask = torch.ones(1, features.shape[0], dtype=torch.bool, device=args.device)
        pred = model.generate(
            features.unsqueeze(0), mask, [previous_prediction],
            max_new_tokens=args.max_new_tokens,
        )[0]

        if name in target_names:
            predictions[name] = pred

        previous_prediction = pred

    unload_model(model)
    return predictions


def main(argv=None):
    args = parse_args(argv)
    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {args.device}")
    print(f"LLM backbone: {args.pretrained_llm}")

    with open(args.metadata, "r", newline="") as f:
        all_rows = list(csv.DictReader(f))
    print(f"Metadata: {len(all_rows)} rows")

    env = lmdb.open(args.lmdb, readonly=True, subdir=False, lock=False,
                    max_readers=126, meminit=False)

    try:
        examples = select_examples(args, all_rows, env)
        print(f"\nSelected {len(examples)} examples:")

        if args.video_dir:
            video_dir = Path(args.video_dir)
            if video_dir.exists():
                print(f"\nExtracting frames from {video_dir}...")
                for ex in examples:
                    paths = extract_frames(
                        video_dir, ex["sentence_name"], output_dir, args.num_frames,
                    )
                    ex["frame_paths"] = paths
                    print(f"  {ex['sentence_name']}: {len(paths)} frames")
            else:
                print(f"\nWARNING: video-dir not found: {video_dir}")
        else:
            print("\nNo --video-dir provided, skipping frame extraction")

        print("\n" + "=" * 60)
        vis_preds = run_visual_model(args, examples, env)

        print("\n" + "=" * 60)
        gt_prev_preds = run_gt_prev_model(args, examples, env)

        print("\n" + "=" * 60)
        mixed_prev_preds = run_mixed_prev_model(args, examples, all_rows, env)

        results_examples = []
        for ex in examples:
            name = ex["sentence_name"]
            vis = vis_preds.get(name, "")
            gt_prev = gt_prev_preds.get(name, "")
            mixed = mixed_prev_preds.get(name, "")
            ref = ex["target_text"]

            results_examples.append({
                "sentence_name": name,
                "video_name": ex["video_name"],
                "ground_truth": ref,
                "prev_sentence_gt": ex["prev_text"],
                "frame_paths": ex.get("frame_paths", []),
                "predictions": {
                    "visual_only": vis,
                    "gt_prev_context": gt_prev,
                    "mixed_context": mixed,
                },
                "bleu_per_model": {
                    "visual_only": sacrebleu.sentence_bleu(vis, [ref]).score if vis else 0.0,
                    "gt_prev_context": sacrebleu.sentence_bleu(gt_prev, [ref]).score if gt_prev else 0.0,
                    "mixed_context": sacrebleu.sentence_bleu(mixed, [ref]).score if mixed else 0.0,
                },
            })

        results = {
            "config": {
                "llm": args.pretrained_llm,
                "checkpoints": {
                    "visual_only": args.vis_checkpoint,
                    "gt_prev_context": args.gt_prev_checkpoint,
                    "mixed_context": args.mixed_prev_checkpoint,
                },
                "num_examples": args.num_examples,
                "selection_samples": args.selection_samples,
                "num_frames": args.num_frames,
                "max_new_tokens": args.max_new_tokens,
            },
            "examples": results_examples,
        }

        results_path = output_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n{'=' * 60}")
        print(f"Results saved to {results_path}")
        for ex in results_examples:
            print(f"\n── {ex['sentence_name']} ──")
            print(f"  Video: {ex['video_name']}")
            print(f"  GT:    {ex['ground_truth']}")
            print(f"  Vis:   {ex['predictions']['visual_only']}")
            print(f"  GT-p:  {ex['predictions']['gt_prev_context']}")
            print(f"  Mixed: {ex['predictions']['mixed_context']}")
            b = ex['bleu_per_model']
            print(f"  BLEU:  Vis={b['visual_only']:.1f}  GT-p={b['gt_prev_context']:.1f}  Mixed={b['mixed_context']:.1f}")

    finally:
        env.close()


if __name__ == "__main__":
    main()
