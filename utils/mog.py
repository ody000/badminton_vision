"""MOG2 background subtraction helpers.

Provides:
  - MOG2Manager: stateful per-video MOG2 wrapper
  - process_video(): batch frame processing
  - group_frames_by_video(): filename grouping
  - coco_mog2(): COCO annotation path rewriting

Ported from slayminton/scripts/mog.py with MOG2Manager class added.
"""

from __future__ import annotations

import cv2
import os
import glob
import json
from collections import defaultdict

TRAIN_PATH      = "data/input/train"
FRAMES_OUT_PATH = "data/input/train_mog_frames"
VIDEOS_OUT_PATH = "data/input/train_mog_mp4s"
DEFAULT_FPS     = 30
JPEG_QUALITY    = 95


# ─────────────────────────────────────────────────────────────────────────────
# MOG2Manager
# ─────────────────────────────────────────────────────────────────────────────

class MOG2Manager:
    """Stateful wrapper around cv2.BackgroundSubtractorMOG2.

    Holds a single subtractor instance per MOG2Manager object.
    Typical use: create one instance per video / per pipeline run.
    """

    def __init__(self, var_threshold: float = 200, history: int = 1000):
        self.var_threshold = float(var_threshold)
        self.history = int(history)
        self._subtractor: cv2.BackgroundSubtractorMOG2 | None = None
        self._last_mask: "cv2.Mat | None" = None
        self._init_subtractor()

    def _init_subtractor(self) -> None:
        sub = cv2.createBackgroundSubtractorMOG2()
        sub.setVarThreshold(self.var_threshold)
        sub.setHistory(self.history)
        self._subtractor = sub
        self._last_mask = None

    def apply(self, frame) -> "cv2.Mat":
        """Apply MOG2 to frame and return grayscale mask."""
        mask = self._subtractor.apply(frame)
        self._last_mask = mask
        return mask

    def get_foreground_ratio(self, frame, box: list) -> float:
        """Return fraction of pixels inside box that are foreground (>127).

        Args:
            frame: Current BGR frame (used only for shape; mask already stored).
            box: [x1, y1, x2, y2] bounding box in pixel coordinates.

        Returns:
            Float in [0.0, 1.0]. Returns 0.0 if mask is not available or box is degenerate.
        """
        if self._last_mask is None:
            return 0.0
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        h, w = self._last_mask.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            return 0.0
        roi = self._last_mask[y1:y2, x1:x2]
        total_pixels = roi.size
        if total_pixels == 0:
            return 0.0
        fg_pixels = int((roi > 127).sum())
        return fg_pixels / total_pixels

    def reset(self) -> None:
        """Reinitialize the subtractor (clears learned background model)."""
        self._init_subtractor()


# ─────────────────────────────────────────────────────────────────────────────
# Batch utilities (ported from slayminton)
# ─────────────────────────────────────────────────────────────────────────────

def group_frames_by_video(folder: str) -> dict:
    """Group all JPGs in folder by their video-id prefix.

    Filename pattern: <video_id>-<frame_num>_jpg.rf.<hash>.jpg
    Returns {video_id: [sorted frame paths...]}.
    """
    groups = defaultdict(list)

    for fp in glob.glob(os.path.join(folder, "*.jpg")):
        name = os.path.basename(fp)
        dash_idx = name.find("-")
        if dash_idx == -1:
            video_id  = "unknown"
            frame_num = 0
        else:
            video_id  = name[:dash_idx]
            remainder = name[dash_idx + 1:]
            try:
                frame_num = int(remainder.split("_")[0])
            except ValueError:
                frame_num = 0

        groups[video_id].append((frame_num, fp))

    return {
        vid: [fp for _, fp in sorted(frames)]
        for vid, frames in sorted(groups.items())
    }


def process_video(video_id: str, frame_paths: list, fps: int = DEFAULT_FPS) -> None:
    """Apply MOG2 to a sequence of frames and save output frames + video."""
    print(f"\n[{video_id}]  {len(frame_paths)} frames")

    first = cv2.imread(frame_paths[0])
    if first is None:
        print("  [skip] Cannot read first frame")
        return
    h, w = first.shape[:2]
    print(f"  size: {w}x{h}  fps: {fps}")

    frame_out_dir = os.path.join(FRAMES_OUT_PATH, video_id)
    os.makedirs(frame_out_dir, exist_ok=True)
    os.makedirs(VIDEOS_OUT_PATH, exist_ok=True)

    video_out_path = os.path.join(VIDEOS_OUT_PATH, f"{video_id}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_out_path, fourcc, fps, (w, h))

    mgr = MOG2Manager(var_threshold=200, history=1000)

    for i, fp in enumerate(frame_paths):
        frame = cv2.imread(fp)
        if frame is None:
            print(f"\n  [warn] Cannot read {fp}, skipping")
            continue

        fg_mask  = mgr.apply(frame)
        mask_3ch = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)

        out_frame_path = os.path.join(frame_out_dir, os.path.basename(fp))
        cv2.imwrite(out_frame_path, mask_3ch, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        writer.write(mask_3ch)

        print(f"  frame {i+1}/{len(frame_paths)}", end="\r")

    writer.release()
    print(f"\n  video  -> {video_out_path}")
    print(f"  frames -> {frame_out_dir}/")


def coco_mog2() -> None:
    """Rewrite image file_name paths in COCO JSON to include video_id subfolder."""
    with open("data/input/train/_annotations.coco.json") as f:
        coco = json.load(f)

        for img in coco["images"]:
            name = img["file_name"]
            dash_idx = name.find("-")
            if dash_idx != -1:
                video_id = name[:dash_idx]
            else:
                raise ValueError(f"Unexpected filename (no dash): {name}")
            img["file_name"] = video_id + "/" + name

    with open("_annotations.coco.json", "w") as f:
        json.dump(coco, f)


def main() -> None:
    groups = group_frames_by_video(TRAIN_PATH)

    if not groups:
        print(f"No JPGs found in {TRAIN_PATH}")
        return

    total_frames = sum(len(v) for v in groups.values())
    print(f"Found {len(groups)} video(s) across {total_frames} frames\n")

    for video_id, frame_paths in groups.items():
        process_video(video_id, frame_paths)

    coco_mog2()

    print("\n\nAll done.")
    print(f"  Frames -> {FRAMES_OUT_PATH}/")
    print(f"  Videos -> {VIDEOS_OUT_PATH}/")


if __name__ == "__main__":
    main()
