"""Local-only court corner setup tool.

Opens the first frame of a video in an OpenCV window and lets the user click
6 court points in order:
    1. Bottom-Left
    2. Bottom-Right
    3. Top-Right
    4. Top-Left
    5. Midline-Bottom
    6. Midline-Top

Keys:
    z  — undo last point
    q  — quit early (saves whatever was collected so far if >= 4 points)

Always saves to data/input/court_points.json keyed by video stem name.
Prints the absolute path on exit.

NOT imported by main.py.

Usage:
    python prepare_run.py --video data/input/match_clip.mp4
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

# Compute paths relative to script location
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
COURT_POINTS_PATH = os.path.join(_PROJECT_ROOT, "data/input/court_points.json")

LABELS = [
    "Bottom-Left",
    "Bottom-Right",
    "Top-Right",
    "Top-Left",
    "Midline-Bottom",
    "Midline-Top",
]


def get_first_frame(video_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(os.path.abspath(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise RuntimeError("Cannot read first frame.")
    return frame


def collect_court_points(first_frame: np.ndarray) -> list:
    """Interactive OpenCV window to collect 6 court points."""
    points: list = []
    orig = first_frame.copy()

    window_name = "Court Corner Setup"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def redraw():
        canvas = orig.copy()
        for i, pt in enumerate(points):
            cv2.circle(canvas, pt, 7, (0, 255, 0), -1)
            cv2.putText(
                canvas,
                LABELS[i],
                (pt[0] + 8, pt[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
        if len(points) < 6:
            instruction = f"Click: {LABELS[len(points)]}   (z=undo, q=quit)"
            cv2.putText(
                canvas,
                instruction,
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.80,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                canvas,
                "All 6 points collected! Press any key to save.",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        cv2.imshow(window_name, canvas)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 6:
            points.append((x, y))
            print(f"[PREP] Point {len(points)}: {LABELS[len(points)-1]} = ({x}, {y})")
            redraw()

    cv2.setMouseCallback(window_name, on_mouse)
    redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord("z"):
            if points:
                removed = points.pop()
                print(f"[PREP] Undo: removed {removed}")
                redraw()
        elif key == ord("q"):
            print("[PREP] Quit early.")
            break
        elif len(points) == 6:
            # Wait for any key press to confirm
            print("[PREP] 6 points collected. Press any key to save.")
            cv2.waitKey(0)
            break

    cv2.destroyAllWindows()
    return points


def save_court_points(video_stem: str, points: list) -> str:
    """Save points to court_points.json and return the absolute path."""
    os.makedirs(os.path.dirname(os.path.abspath(COURT_POINTS_PATH)), exist_ok=True)

    # Load existing data
    if os.path.exists(COURT_POINTS_PATH):
        with open(COURT_POINTS_PATH, "r", encoding="utf-8") as f:
            all_data = json.load(f)
    else:
        all_data = {}

    all_data[video_stem] = points

    with open(COURT_POINTS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=4)

    abs_path = os.path.abspath(COURT_POINTS_PATH)
    return abs_path


def main():
    parser = argparse.ArgumentParser(
        description="Court corner setup tool — click 6 points, save to court_points.json."
    )
    parser.add_argument("--video", required=True, help="Path to input video.")
    args = parser.parse_args()

    video_path = args.video
    if not os.path.exists(video_path):
        print(f"[PREP] Error: video not found: {video_path}")
        sys.exit(1)

    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    print(f"[PREP] Video: {video_path}  (stem: {video_stem})")
    print(f"[PREP] Click 6 points in order: {', '.join(LABELS)}")

    first_frame = get_first_frame(video_path)
    points = collect_court_points(first_frame)

    if len(points) < 4:
        print(f"[PREP] Only {len(points)} points collected (need >= 4). Aborting save.")
        sys.exit(1)

    saved_path = save_court_points(video_stem, points)
    print(f"[PREP] Saved {len(points)} points to: {saved_path}")
    print(f"[PREP] Key: '{video_stem}'")


if __name__ == "__main__":
    main()
