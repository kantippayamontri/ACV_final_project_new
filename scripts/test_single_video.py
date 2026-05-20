#!/usr/bin/env python3
import os
import sys
import torch
import lmdb
import numpy as np
from tqdm import tqdm
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset.preprocessing import load_video_frames, resize_frames, sliding_windows, preprocess_window
from transformers import VideoMAEModel

def load_model(device):
    model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
    model = model.to(device)
    model = model.eval()
    return model

@torch.no_grad()
def extract_features(model, windows, device, batch_size=32):
    features = []
    for i in range(0, len(windows), batch_size):
        batch = windows[i:i+batch_size]
        batch_tensor = torch.stack([preprocess_window(w) for w in batch])
        batch_tensor = batch_tensor.to(device)
        outputs = model(batch_tensor)
        pooled = outputs.last_hidden_state.mean(dim=1)
        features.append(pooled.cpu())
    return torch.cat(features, dim=0)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = load_model(device)
    
    # Test video with 36 clips
    test_video_name = "-f1_kdl050s-1-rgb_front"
    
    # Load metadata
    import pandas as pd
    df = pd.read_csv("features/val_metadata.csv")
    video_clips = df[df["VIDEO_NAME"] == test_video_name].sort_values("START_REALIGNED")
    
    print(f"\nTesting video: {test_video_name}")
    print(f"Number of clips: {len(video_clips)}")
    
    # Create test LMDB
    lmdb_path = f"features/test_{test_video_name}_features.lmdb"
    env = lmdb.open(lmdb_path, subdir=False, map_size=1024**3)
    
    video_root = Path("datasets/val_rgb_front_clips/raw_videos")
    
    total_windows = 0
    
    for idx, row in tqdm(video_clips.iterrows(), total=len(video_clips), desc="Extracting"):
        sentence_name = row["SENTENCE_NAME"]
        video_path = video_root / f"{sentence_name}.mp4"
        
        if not video_path.exists():
            print(f"  Skipping {sentence_name}: video not found")
            continue
        
        frames = load_video_frames(str(video_path))
        frames = resize_frames(frames)
        windows = list(sliding_windows(frames, window_size=16, stride=2))
        
        if len(windows) == 0:
            print(f"  Skipping {sentence_name}: no windows")
            continue
        
        features = extract_features(model, windows, device, batch_size=32)
        
        # Store in LMDB
        with env.begin(write=True) as txn:
            for feat_idx, feat in enumerate(features):
                key = f"{sentence_name}/{feat_idx:07d}.np"
                txn.put(key.encode(), feat.numpy().astype(np.float16).tobytes())
            
            txn.put(f"{sentence_name}/done".encode(), b"")
        
        total_windows += len(windows)
    
    env.close()
    
    # Verify
    env = lmdb.open(lmdb_path, readonly=True, subdir=False)
    with env.begin() as txn:
        cursor = txn.cursor()
        done_count = sum(1 for k,v in cursor if k.decode().endswith('/done'))
        feat_count = txn.stat()['entries'] - done_count
    
    print(f"\n✓ Test complete: {done_count} clips, {feat_count} feature vectors")
    print(f"  LMDB path: {lmdb_path}")

if __name__ == "__main__":
    main()
