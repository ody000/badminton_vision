"""PlayerDetector: YOLOv8 detection (no ByteTrack).

Uses model.predict() for simple per-frame detection.  ByteTrack (model.track +
persist=True) was removed because the Kalman-state overhead exceeded the benefit
for the 2-player, fixed-court use case, and the stale Kalman predictions after
interval gaps caused more matching work, not less.

ID assignment is now frame-local ordinal (0, 1, …) rather than persistent
ByteTrack IDs.  PlayerContext's first-seen slot assignment still works correctly
with ordinal IDs because it assigns P1/P2 by first-appearance order, not by the
magnitude of the ID.

Performance note:
    player_detect_interval (default 3) controls how often YOLO inference runs.
    On frames between YOLO calls the last detection result is returned verbatim.
    The device is passed explicitly to model.predict() to prevent silent CPU
    fallback in environments where Ultralytics' auto-detection misfires.
"""

from __future__ import annotations

import os

import numpy as np


class PlayerDetector:
    """Detect players using YOLOv8 + ByteTrack.

    Args:
        cfg: SimpleNamespace from load_config().
    """

    def __init__(self, cfg=None):
        from ultralytics import YOLO

        self.frame_count = 0

        if cfg is not None:
            weights = getattr(cfg, "player_weights", "models/yolo.pt")
            self.conf_threshold = float(getattr(cfg, "player_conf_threshold", 0.5))
            self.device = getattr(cfg, "device", "cpu")
            self._detect_interval = max(1, int(getattr(cfg, "player_detect_interval", 3)))
        else:
            weights = "models/yolo.pt"
            self.conf_threshold = 0.5
            self.device = "cpu"
            self._detect_interval = 3

        # Internal state for interval-based detection.
        # _frames_since_detect starts at interval so YOLO fires on the very first call.
        self._frames_since_detect: int = self._detect_interval
        self._last_detections: list[dict] = []

        # Fall back to yolov8n.pt if fine-tuned weights do not exist
        if not os.path.exists(weights):
            print(f"[YOLO] Weights not found at '{weights}', falling back to yolov8n.pt")
            weights = "yolov8n.pt"

        print(f"[YOLO] Loading model from {weights} (detect_interval={self._detect_interval})")
        self.model = YOLO(weights)

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Run YOLOv8 detection on a BGR frame.

        YOLO inference is skipped on frames between detect_interval boundaries;
        the previous frame's detections are returned instead.

        IDs are frame-local ordinals (0, 1, …) — not persistent ByteTrack IDs.
        PlayerContext's first-seen slot assignment handles P1/P2 stability.

        Args:
            frame: BGR numpy frame.

        Returns:
            List of dicts: {"id": int, "box": [x1,y1,x2,y2], "feet": (cx, y2), "feet_real": None}
        """
        self.frame_count += 1
        self._frames_since_detect += 1

        # Return cached result on non-detection frames.
        if self._frames_since_detect <= self._detect_interval:
            return self._last_detections

        # It's time for a fresh YOLO inference.
        self._frames_since_detect = 1

        results = self.model.predict(
            frame,
            classes=[0],   # person only
            conf=self.conf_threshold,
            verbose=False,
            device=self.device,  # explicit — prevents silent CPU fallback on OSCAR
        )

        detections = []
        if not results or results[0].boxes is None:
            self._last_detections = detections
            return detections

        boxes_result = results[0].boxes
        xyxy = boxes_result.xyxy.cpu().numpy() if hasattr(boxes_result.xyxy, "cpu") else np.array(boxes_result.xyxy)

        for i, box_arr in enumerate(xyxy):
            x1, y1, x2, y2 = int(box_arr[0]), int(box_arr[1]), int(box_arr[2]), int(box_arr[3])
            cx = (x1 + x2) // 2
            detections.append({
                "id": i,               # frame-local ordinal — no ByteTrack dependency
                "box": [x1, y1, x2, y2],
                "feet": (cx, y2),
                "feet_real": None,     # populated by main.py via CourtMapper
            })

        self._last_detections = detections
        return detections
