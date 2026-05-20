#!/usr/bin/env python3
"""
Extract video features from How2Sign videos using VideoMAE (frozen).

Usage:
    python scripts/extract_features.py --split val
    python scripts/extract_features.py --split train
    python scripts/extract_features.py --split test
"""
import argparse
import csv
import lmdb
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import VideoMAEModel
from src.dataset.preprocessing import load_video_frames, resize_frames, sliding_windows, preprocess_window


VIDEO_MODEL_NAME = "MCG-NJU/videomae-base-finetuned-kinetics"
WINDOW_SIZE = 16
STRIDE = 2
FEATURE_DIM = 768
COMMIT_INTERVAL = 100


def parse_args():
    parser = argparse.ArgumentParser(description="Extract VideoMAE features from How2Sign videos")
    parser.add_argument("--split", type=str, required=True, choices=["train", "val", "test"],
                        help="Dataset split to process")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to CSV file (auto-detected if not provided)")
    parser.add_argument("--video-dir", type=str, default=None,
                        help="Path to video directory (auto-detected if not provided)")
    parser.add_argument("--output-lmdb", type=str, default=None,
                        help="Path to output LMDB directory (auto-detected if not provided)")
    parser.add_argument("--output-metadata", type=str, default=None,
                        help="Path to output metadata CSV (auto-detected if not provided)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for extraction")
    parser.add_argument("--map-size-gb", type=int, default=50,
                        help="LMDB map size in GB (50GB should be enough for train)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for VideoMAE inference")
    return parser.parse_args()


def get_paths(split: str, args: argparse.Namespace):
    """Auto-detect or use provided paths"""
    base_dir = Path(__file__).parent.parent
    
    if args.csv:
        csv_path = Path(args.csv)
    else:
        if split == "val":
            csv_path = base_dir / "datasets" / "val_rgb_front_clips" / "how2sign_realigned_val.csv"
        elif split == "train":
            csv_path = base_dir / "datasets" / "how2sign_realigned_train.csv"
        else:  # test
            csv_path = base_dir / "datasets" / "how2sign_realigned_test.csv"
    
    if args.video_dir:
        video_dir = Path(args.video_dir)
    else:
        if split == "val":
            video_dir = base_dir / "datasets" / "val_rgb_front_clips" / "raw_videos"
        elif split == "train":
            video_dir = base_dir / "datasets" / "train_rgb_front_clips" / "raw_videos"
        else:  # test
            video_dir = base_dir / "datasets" / "test_rgb_front_clips" / "raw_videos"
    
    if args.output_lmdb:
        output_lmdb = Path(args.output_lmdb)
    else:
        output_lmdb = base_dir / "features" / f"{split}_features.lmdb"
    
    if args.output_metadata:
        output_metadata = Path(args.output_metadata)
    else:
        output_metadata = base_dir / "features" / f"{split}_metadata.csv"
    
    return csv_path, video_dir, output_lmdb, output_metadata


def load_csv(csv_path: Path):
    """Load CSV and return list of dicts with added previous_sentence field"""
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        rows = list(reader)
    
    for row in rows:
        row['START_REALIGNED'] = float(row['START_REALIGNED'])
        row['END_REALIGNED'] = float(row['END_REALIGNED'])
    
    rows.sort(key=lambda x: (x['VIDEO_NAME'], x['START_REALIGNED']))
    
    prev_sentence = None
    prev_video_name = None
    for row in rows:
        if row['VIDEO_NAME'] == prev_video_name:
            row['PREV_SENTENCE'] = prev_sentence
        else:
            row['PREV_SENTENCE'] = None
        prev_sentence = row['SENTENCE']
        prev_video_name = row['VIDEO_NAME']
    
    return rows


def save_metadata(rows: list, output_path: Path):
    """Save metadata CSV"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        fieldnames = ['SENTENCE_NAME', 'SENTENCE', 'PREV_SENTENCE', 'VIDEO_NAME', 'START_REALIGNED', 'END_REALIGNED']
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def load_model(device: str):
    """Load frozen VideoMAE model"""
    print(f"Loading VideoMAE model from {VIDEO_MODEL_NAME}...")
    model = VideoMAEModel.from_pretrained(VIDEO_MODEL_NAME)
    
    model = model.to(device)
    model.eval()
    
    for param in model.parameters():
        param.requires_grad = False
    
    print(f"Model loaded on {device}")
    return model


def extract_features(
    video_path: Path,
    model: VideoMAEModel,
    device: str,
    batch_size: int
) -> torch.Tensor:
    """
    Extract features from a single video.
    
    Returns:
        torch.Tensor of shape [T, 768]
    """
    frames = load_video_frames(str(video_path))
    frames = resize_frames(frames)
    windows = sliding_windows(frames, window_size=WINDOW_SIZE, stride=STRIDE)
    
    if len(windows) == 0:
        return torch.empty(0, FEATURE_DIM)
    
    all_features = []
    
    for i in range(0, len(windows), batch_size):
        batch_windows = windows[i:i + batch_size]
        batch_tensors = torch.stack([preprocess_window(w) for w in batch_windows])
        batch_tensors = batch_tensors.to(device)
        
        with torch.no_grad():
            outputs = model(batch_tensors)
            last_hidden_state = outputs.last_hidden_state
        
        batch_features = last_hidden_state.mean(dim=1)
        
        all_features.append(batch_features.cpu())
    
    return torch.cat(all_features, dim=0)


def migrate_legacy_feature_keys(env, sentence_name: str) -> bool:
    """Rewrite legacy 1-based LMDB keys to the current 0-based scheme."""
    done_key = f"{sentence_name}/done".encode('ascii')

    with env.begin() as txn:
        if txn.get(done_key) is None:
            return False

        prefix = f"{sentence_name}/".encode('ascii')
        cursor = txn.cursor()
        legacy_keys = []
        legacy_values = []

        if cursor.set_range(prefix):
            for key, value in cursor:
                if not key.startswith(prefix):
                    break
                if key == done_key or not key.endswith(b".np"):
                    continue
                legacy_keys.append(key)
                legacy_values.append(bytes(value))

    if not legacy_keys:
        return False

    expected_first_key = f"{sentence_name}/0000000.np".encode('ascii')
    if legacy_keys[0] == expected_first_key:
        return False

    with env.begin(write=True) as txn:
        for index, (old_key, value) in enumerate(zip(legacy_keys, legacy_values)):
            new_key = f"{sentence_name}/{index:07d}.np".encode('ascii')
            txn.put(new_key, value)
            if new_key != old_key:
                txn.delete(old_key)

    return True


def process_single_video(
    env,
    sentence_name: str,
    video_path: Path,
    model,
    device: str,
    batch_size: int,
    extract_fn=None,
):
    """
    Process a single video: extract features and write to LMDB.

    Handles three edge cases:
    - Empty videos (<16 frames) get a /done marker so resume skips them
    - Decode/inference errors are caught without marking the clip done
    - Feature keys are 0-based: {sentence_name}/{feat_idx:07d}.np

    Args:
        env: lmdb.Environment
        sentence_name: key for this clip
        video_path: path to .mp4 file
        model: VideoMAE model
        device: torch device string
        batch_size: inference batch size
        extract_fn: override extract_features for testing

    Returns:
        str status: "ok", "skipped_empty", or "skipped_error"
    """
    if extract_fn is None:
        extract_fn = extract_features

    done_key = f"{sentence_name}/done".encode('ascii')

    try:
        features = extract_fn(video_path, model, device, batch_size)
    except Exception as e:
        print(f"  ERROR processing {sentence_name}: {e}")
        return "skipped_error"

    if features.shape[0] == 0:
        with env.begin(write=True) as txn:
            txn.put(done_key, b"1")
        return "skipped_empty"

    with env.begin(write=True) as txn:
        for feat_idx in range(features.shape[0]):
            key = f"{sentence_name}/{feat_idx:07d}.np".encode('ascii')
            value = features[feat_idx].half().numpy().tobytes()
            txn.put(key, value)

        txn.put(done_key, b"1")

    return "ok"


def main():
    args = parse_args()
    csv_path, video_dir, output_lmdb, output_metadata = get_paths(args.split, args)
    
    print(f"Processing {args.split.upper()} split")
    print(f"CSV: {csv_path}")
    print(f"Video dir: {video_dir}")
    print(f"Output LMDB: {output_lmdb}")
    print(f"Output metadata: {output_metadata}")
    print(f"Device: {args.device}")
    
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}")
        return
    
    if not video_dir.exists():
        print(f"WARNING: Video directory not found: {video_dir}")
        print("Continuing to create metadata only...")
    
    rows = load_csv(csv_path)
    print(f"Loaded {len(rows)} sentences from CSV")
    
    save_metadata(rows, output_metadata)
    print(f"Saved metadata to {output_metadata}")
    
    if not video_dir.exists():
        print("Skipping feature extraction (no videos)")
        return
    
    output_lmdb.parent.mkdir(parents=True, exist_ok=True)
    
    model = load_model(args.device)
    
    map_size_bytes = args.map_size_gb * (1024 ** 3)
    env = lmdb.open(
        str(output_lmdb),
        map_size=map_size_bytes,
        subdir=False,
        readonly=False,
        meminit=False,
        map_async=True
    )
    
    stats = {"extracted": 0, "skipped_missing": 0, "skipped_done": 0, "skipped_empty": 0, "skipped_error": 0}
    
    for idx, row in enumerate(tqdm(rows, desc="Extracting features")):
        sentence_name = row['SENTENCE_NAME']
        video_filename = f"{sentence_name}.mp4"
        video_path = video_dir / video_filename

        migrated_legacy = migrate_legacy_feature_keys(env, sentence_name)
        if migrated_legacy:
            stats["skipped_done"] += 1
            continue
        
        if not video_path.exists():
            stats["skipped_missing"] += 1
            continue
        
        done_key = f"{sentence_name}/done".encode('ascii')
        with env.begin() as txn:
            if txn.get(done_key) is not None:
                stats["skipped_done"] += 1
                continue
        
        status = process_single_video(
            env=env,
            sentence_name=sentence_name,
            video_path=video_path,
            model=model,
            device=args.device,
            batch_size=args.batch_size,
        )

        if status == "ok":
            stats["extracted"] += 1
        elif status == "skipped_empty":
            stats["skipped_empty"] += 1
        elif status == "skipped_error":
            stats["skipped_error"] += 1
        
        if (idx + 1) % COMMIT_INTERVAL == 0:
            env.sync()
    
    env.sync()
    env.close()
    
    print("\n=== Extraction Complete ===")
    print(f"Extracted: {stats['extracted']}")
    print(f"Skipped (missing video): {stats['skipped_missing']}")
    print(f"Skipped (already done): {stats['skipped_done']}")
    print(f"Skipped (no features): {stats['skipped_empty']}")
    print(f"Skipped (error): {stats['skipped_error']}")
    total = sum(stats.values())
    print(f"Total processed: {total}")
    print(f"Metadata saved to: {output_metadata}")
    print(f"Features saved to: {output_lmdb}")


if __name__ == "__main__":
    main()
