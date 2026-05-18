"""PlayerDetector: YOLOv8 + ByteTrack.

Uses model.track() with persist=True for stable ByteTrack IDs.
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
        else:
            weights = "models/yolo.pt"
            self.conf_threshold = 0.5
            self.device = "cpu"

        # Fall back to yolov8n.pt if fine-tuned weights do not exist
        if not os.path.exists(weights):
            print(f"[YOLO] Weights not found at '{weights}', falling back to yolov8n.pt")
            weights = "yolov8n.pt"

        print(f"[YOLO] Loading model from {weights}")
        self.model = YOLO(weights)

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Run YOLOv8 + ByteTrack on a BGR frame.

        Args:
            frame: BGR numpy frame.

        Returns:
            List of dicts: {"id": int, "box": [x1,y1,x2,y2], "feet": (cx, y2), "feet_real": None}
        """
        self.frame_count += 1

        results = self.model.track(
            frame,
            persist=True,
            classes=[0],  # person only
            conf=self.conf_threshold,
            verbose=False,
        )

        detections = []
        if not results or results[0].boxes is None:
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

        return detections
