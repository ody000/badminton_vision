"""CourtMapper: pixel ↔ real-world court coordinate transform.

Reads 6 court corner points from a JSON file (never opens GUI).
Uses only the 4 outer corners for the homography matrix.
Midline points (index 4, 5) are stored but not used in homography computation.

Point order: Bottom-Left, Bottom-Right, Top-Right, Top-Left, Midline-Bottom, Midline-Top

Real-world coordinate system (cm):
  Bottom-Left:   (0, court_length)
  Bottom-Right:  (court_width, court_length)
  Top-Right:     (court_width, 0)
  Top-Left:      (0, 0)
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np


class CourtMapper:
    """Pixel ↔ real-world homography for a badminton court.

    Never opens a GUI window. Court corners are loaded from JSON.
    """

    def __init__(self, cfg=None):
        """
        Args:
            cfg: SimpleNamespace from load_config(). Reads court_real_width_cm,
                 court_real_length_cm, court_points_file.
        """
        self._H: np.ndarray | None = None  # homography matrix (pixel -> real)
        self._image_corners_6pts: list | None = None

        if cfg is not None:
            self.court_width_cm = float(getattr(cfg, "court_real_width_cm", 610.0))
            self.court_length_cm = float(getattr(cfg, "court_real_length_cm", 1340.0))
            self.court_points_file = getattr(cfg, "court_points_file", "data/input/court_points.json")
        else:
            self.court_width_cm = 610.0
            self.court_length_cm = 1340.0
            self.court_points_file = "data/input/court_points.json"

        # Real-world destinations for the 4 outer corners (cm).
        # Order must match input: BL, BR, TR, TL
        self._real_corners = np.array([
            [0.0,                  self.court_length_cm],  # Bottom-Left
            [self.court_width_cm,  self.court_length_cm],  # Bottom-Right
            [self.court_width_cm,  0.0],                   # Top-Right
            [0.0,                  0.0],                   # Top-Left
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_from_json(self, json_path: str, video_stem: str) -> bool:
        """Load court corners from JSON keyed by video stem.

        Args:
            json_path: Path to court_points.json.
            video_stem: Key to look up in JSON (typically video filename without extension).

        Returns:
            True if corners were found and loaded, False otherwise.
        """
        if not os.path.exists(json_path):
            print(f"[COURT] court_points.json not found: {json_path}")
            return False

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        points = data.get(video_stem)
        if points is None:
            # Try a few fallback keys (stem without suffix variants)
            for key in data:
                if video_stem in key or key in video_stem:
                    points = data[key]
                    print(f"[COURT] Using fallback key '{key}' for stem '{video_stem}'")
                    break

        if points is None:
            print(f"[COURT] No entry for '{video_stem}' in {json_path}")
            return False

        if len(points) < 4:
            print(f"[COURT] Need at least 4 points for '{video_stem}', got {len(points)}")
            return False

        self.calibrate(points)
        return True

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self, image_corners_6pts) -> None:
        """Compute homography from 6 pixel-space points.

        Only the first 4 points (outer corners) are used for the homography.
        Point order: BL, BR, TR, TL, Midline-Bottom, Midline-Top.

        Args:
            image_corners_6pts: List/array of at least 4 (x, y) pixel tuples.
        """
        self._image_corners_6pts = list(image_corners_6pts)
        # Use only the 4 outer corners (indices 0-3)
        src = np.array(image_corners_6pts[:4], dtype=np.float32)
        dst = self._real_corners
        self._H, _ = cv2.findHomography(src, dst)
        if self._H is None:
            print("[COURT] Warning: findHomography returned None — check corner points.")

    # ------------------------------------------------------------------
    # Transforms
    # ------------------------------------------------------------------

    def transform_point(self, pixel_xy) -> tuple | None:
        """Transform a pixel coordinate to real-world cm.

        Args:
            pixel_xy: (x, y) pixel tuple, or None.

        Returns:
            (x_cm, y_cm) tuple or None if not calibrated or input is None.
        """
        if self._H is None or pixel_xy is None:
            return None
        pt = np.array([[[float(pixel_xy[0]), float(pixel_xy[1])]]], dtype=np.float32)
        result = cv2.perspectiveTransform(pt, self._H)
        return (float(result[0, 0, 0]), float(result[0, 0, 1]))

    def get_player_feet(self, box) -> tuple:
        """Return bottom-center pixel of a [x1, y1, x2, y2] bounding box.

        Args:
            box: [x1, y1, x2, y2] bounding box.

        Returns:
            (cx, y2) pixel tuple.
        """
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        cx = (x1 + x2) / 2.0
        return (cx, float(y2))

    def is_calibrated(self) -> bool:
        """Return True if homography matrix has been computed."""
        return self._H is not None
