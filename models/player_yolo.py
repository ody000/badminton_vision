"""PlayerDetector: YOLOv8 detection with ByteTrack for persistent ID tracking.

Uses model.track(persist=True) for frame-by-frame player tracking with
Kalman filter-based data association. ByteTrack maintains consistent player
IDs across frames even with missed detections.

Performance note:
    With player_detect_interval=1 (frame-by-frame), every frame gets fresh
    YOLO inference + ByteTrack assignment. This is the most accurate but
    compute-intensive mode. For frame-by-frame tracking with plenty of compute,
    this is the recommended configuration.

    The device is passed explicitly to model.track() to prevent silent CPU
    fallback in environments where Ultralytics' auto-detection misfires.
"""

from __future__ import annotations

import os

import numpy as np


class PlayerDetector:
    """Detect players using YOLOv8 + ByteTrack persistent ID tracking.

    Uses model.track(persist=True) with Kalman filter for optimal frame-by-frame
    tracking accuracy. ByteTrack maintains consistent player IDs across frames.

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
            self._detect_interval = max(1, int(getattr(cfg, "player_detect_interval", 1)))
        else:
            weights = "models/yolo.pt"
            self.conf_threshold = 0.5
            self.device = "cpu"
            self._detect_interval = 1

        # Internal state for interval-based detection (caching between frames).
        # _frames_since_detect starts at interval so YOLO fires on the very first call.
        self._frames_since_detect: int = self._detect_interval
        self._last_detections: list[dict] = []

        # Fall back to yolov8n.pt if fine-tuned weights do not exist
        if not os.path.exists(weights):
            print(f"[YOLO] Weights not found at '{weights}', falling back to yolov8n.pt")
            weights = "yolov8n.pt"

        print(f"[YOLO] Loading model from {weights} (detect_interval={self._detect_interval})")
        self.model = YOLO(weights)

    def set_detect_interval(self, interval: int) -> None:
        """Set detection interval for frame-skipping cache.

        When interval=1, YOLO runs every frame (frame-by-frame tracking).
        When interval>1, YOLO output is cached between detection frames.
        """
        self._detect_interval = max(1, int(interval))

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Run YOLOv8 detection + ByteTrack on a BGR frame.

        YOLO inference + ByteTrack is skipped on frames between detect_interval
        boundaries; the previous frame's detections are returned instead.

        Persistent IDs are maintained by ByteTrack's Kalman filter + Hungarian
        matching. Each detection frame runs model.track(persist=True) for
        frame-by-frame accuracy.

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

        # It's time for a fresh YOLO + ByteTrack inference.
        self._frames_since_detect = 1

        results = self.model.track(
            frame,
            classes=[0],   # person only
            conf=self.conf_threshold,
            persist=True,  # ByteTrack: maintain IDs across frames
            verbose=False,
            device=self.device,  # explicit — prevents silent CPU fallback on OSCAR
        )

        detections = []
        if not results or results[0].boxes is None:
            self._last_detections = detections
            return detections

        boxes_result = results[0].boxes

        # Extract YOLOv8 track IDs and boxes
        track_ids = boxes_result.id.cpu().numpy() if hasattr(boxes_result, "id") and boxes_result.id is not None else None
        xyxy = boxes_result.xyxy.cpu().numpy() if hasattr(boxes_result.xyxy, "cpu") else np.array(boxes_result.xyxy)

        # Build detections with ByteTrack IDs
        for i, box_arr in enumerate(xyxy):
            x1, y1, x2, y2 = int(box_arr[0]), int(box_arr[1]), int(box_arr[2]), int(box_arr[3])
            cx = (x1 + x2) // 2

            # Use ByteTrack ID if available, else use index as fallback
            track_id = int(track_ids[i]) if track_ids is not None and i < len(track_ids) else i

            detections.append({
                "id": track_id,
                "box": [x1, y1, x2, y2],
                "feet": (cx, y2),
                "feet_real": None,  # populated by main.py via CourtMapper
            })

        self._last_detections = detections
        return detections
