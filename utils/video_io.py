"""Video I/O helpers.

Provides:
  - VideoIOHandler: stream/write frames, get metadata
  - extract_frames(): save JPG frames from a video
"""

from __future__ import annotations

import os
from typing import Generator, Tuple

import cv2
import numpy as np


class VideoIOHandler:
    """Unified handler for reading and writing video files.

    Args:
        input_path: Path to input video file.
        output_path: Optional path for output video. Writer opened lazily on first write_frame().
    """

    def __init__(self, input_path: str, output_path: str | None = None):
        self.input_path = os.path.abspath(input_path)
        self.output_path = output_path
        self._writer: cv2.VideoWriter | None = None
        self._fps: float | None = None
        self._frame_count: int | None = None
        self._frame_w: int | None = None
        self._frame_h: int | None = None

        # Open capture to read metadata eagerly.
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.input_path}")
        self._fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self._frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

    def get_fps(self) -> float:
        return self._fps

    def get_frame_count(self) -> int:
        return self._frame_count

    def get_first_frame(self) -> np.ndarray:
        """Return first frame as BGR numpy array."""
        cap = cv2.VideoCapture(self.input_path)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            raise RuntimeError(f"Cannot read first frame from {self.input_path}")
        return frame

    def stream(self) -> Generator[Tuple[np.ndarray, int, float], None, None]:
        """Generator yielding (frame_bgr, frame_idx, timestamp_s) tuples."""
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video for streaming: {self.input_path}")
        fps = self._fps or 30.0
        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                timestamp_s = frame_idx / fps
                yield frame, frame_idx, timestamp_s
                frame_idx += 1
        finally:
            cap.release()

    def _ensure_writer(self, frame: np.ndarray) -> None:
        if self._writer is not None:
            return
        if self.output_path is None:
            raise RuntimeError("No output_path set on VideoIOHandler.")
        out_abs = os.path.abspath(self.output_path)
        os.makedirs(os.path.dirname(out_abs), exist_ok=True)
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(
            out_abs,
            fourcc,
            self._fps or 30.0,
            (w, h),
        )

    def write_frame(self, frame: np.ndarray) -> None:
        """Write frame to output video. Opens writer on first call."""
        self._ensure_writer(frame)
        self._writer.write(frame)

    def release(self) -> None:
        """Flush and close output writer."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None


def extract_frames(
    video_path: str,
    output_dir: str,
    fps: float | None = None,
) -> list[str]:
    """Extract frames from video and save as numbered JPGs.

    Args:
        video_path: Source video path.
        output_dir: Destination directory for extracted frames.
        fps: If set, extract at this rate (subsample); otherwise extract every frame.

    Returns:
        List of absolute paths to saved JPG files.
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(os.path.abspath(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    step = max(1, round(source_fps / fps)) if fps else 1

    saved: list[str] = []
    frame_idx = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        if frame_idx % step == 0:
            out_name = f"frame_{saved_idx:06d}.jpg"
            out_path = os.path.join(output_dir, out_name)
            cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved.append(os.path.abspath(out_path))
            saved_idx += 1
        frame_idx += 1

    cap.release()
    print(f"[VIDEO_IO] Extracted {len(saved)} frames to {output_dir}")
    return saved
