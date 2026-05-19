"""Background estimation for TrackNetV3: median image from first N frames."""
from __future__ import annotations

import cv2
import numpy as np


def estimate_background(
    video_path: str,
    n_frames: int = 150,
    resize_hw: tuple[int, int] = (288, 512),
) -> np.ndarray:
    """Return median image of first `n_frames` frames, resized to (H, W, 3) uint8.

    Args:
        video_path: Path to input video.
        n_frames:   Number of frames to sample for median (150 = 5s at 30fps).
        resize_hw:  Target (H, W) matching TrackNet input resolution.

    Returns:
        np.ndarray of shape (H, W, 3) dtype uint8.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    for _ in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        resized = cv2.resize(frame, (resize_hw[1], resize_hw[0]))
        frames.append(resized.astype(np.float32))
    cap.release()

    if not frames:
        return np.zeros((*resize_hw, 3), dtype=np.uint8)

    median = np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)
    return median  # (H, W, 3) BGR
