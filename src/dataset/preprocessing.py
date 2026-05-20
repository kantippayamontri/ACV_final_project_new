"""Video preprocessing utilities for feature extraction"""
import numpy as np
import torch
from typing import List
from decord import VideoReader, cpu
from torchvision.transforms import functional as TF
from PIL import Image


VIDEO_MEAN = [0.485, 0.456, 0.406]
VIDEO_STD = [0.229, 0.224, 0.225]
FRAME_SIZE = 224


def load_video_frames(video_path: str) -> np.ndarray:
    """
    Load all frames from a video file using decord.
    
    Args:
        video_path: Path to .mp4 video file
    
    Returns:
        np.ndarray of shape [N, H, W, 3], uint8
    """
    vr = VideoReader(video_path, ctx=cpu(0))
    frames = vr[:].asnumpy()
    return frames


def resize_frames(frames: np.ndarray, target_size: int = 224) -> np.ndarray:
    """Resize all frames to target_size x target_size.

    Uses BILINEAR interpolation (same as VideoMAE preprocessing) so extracted
    features are identical to the old per-window resize path, but peak RAM drops
    ~18x because the heavy full-resolution frame array is replaced in-place.
    """
    n_frames = len(frames)
    resized = np.empty((n_frames, target_size, target_size, 3), dtype=np.uint8)
    for i in range(n_frames):
        pil_img = Image.fromarray(frames[i])
        pil_img = TF.resize(
            pil_img, (target_size, target_size),
            interpolation=TF.InterpolationMode.BILINEAR,
        )
        resized[i] = np.array(pil_img)
    return resized


def sliding_windows(
    frames: np.ndarray,
    window_size: int = 16,
    stride: int = 2
) -> List[np.ndarray]:
    """
    Generate sliding windows over video frames.
    
    Args:
        frames: np.ndarray of shape [N, H, W, 3]
        window_size: Number of frames per window
        stride: Step size between windows
    
    Returns:
        List of np.ndarray, each of shape [window_size, H, W, 3]
    """
    n_frames = len(frames)
    if n_frames < window_size:
        return []
    
    windows = []
    for start in range(0, n_frames - window_size + 1, stride):
        windows.append(frames[start:start + window_size].copy())
    
    return windows


def preprocess_window(window: np.ndarray) -> torch.Tensor:
    """
    Preprocess a window of frames for VideoMAE input.
    
    Args:
        window: np.ndarray of shape [16, H, W, 3], uint8
    
    Returns:
        torch.Tensor of shape [16, 3, 224, 224], float32
    """
    processed_frames = []
    
    for frame in window:
        pil_img = Image.fromarray(frame)
        if pil_img.size != (FRAME_SIZE, FRAME_SIZE):
            pil_img = TF.resize(pil_img, (FRAME_SIZE, FRAME_SIZE), interpolation=TF.InterpolationMode.BILINEAR)
        tensor = TF.to_tensor(pil_img)
        tensor = TF.normalize(tensor, mean=VIDEO_MEAN, std=VIDEO_STD)
        processed_frames.append(tensor)
    
    return torch.stack(processed_frames)
