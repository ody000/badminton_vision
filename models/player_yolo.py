"""PlayerDetector: YOLOv8 + ByteTrack.

Uses model.track() with persist=True for stable ByteTrack IDs.

Performance note:
    player_detect_interval (default 3) controls how often YOLO inference runs.
    On frames between YOLO calls the last detection result is returned verbatim —
    ByteTrack can extrapolate player positions across 2–3 frames without quality
    loss for the slow-moving targets in this use case.  The savings scale linearly:
    interval=3 reduces YOLO calls (and GPU time for player detection) by ~67%.

    The device is passed explicitly to model.track() to prevent silent CPU
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
        """Run YOLOv8 + ByteTrack on a BGR frame.

        YOLO inference is skipped on frames between detect_interval boundaries;
        the previous frame's detections are returned instead.  ByteTrack's
        internal Kalman state persists across the gap because persist=True is
        used on every YOLO call.

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

        results = self.model.track(
            frame,
            persist=True,
            classes=[0],  # person only
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

        # ByteTrack IDs (None before track has IDs)
        if boxes_result.id is not None:
            track_ids = boxes_result.id.cpu().numpy().astype(int)
        else:
            track_ids = list(range(len(xyxy)))

        for i, box_arr in enumerate(xyxy):
            x1, y1, x2, y2 = int(box_arr[0]), int(box_arr[1]), int(box_arr[2]), int(box_arr[3])
            box = [x1, y1, x2, y2]
            track_id = int(track_ids[i]) if i < len(track_ids) else i

            cx = (x1 + x2) // 2
            feet = (cx, y2)

            detections.append({
                "id": track_id,
                "box": box,
                "feet": feet,
                "feet_real": None,  # populated by main.py via CourtMapper
            })

        self._last_detections = detections
        return detections
