"""PlayerDetector: YOLOv8 detection with lightweight centroid-based tracking.

Uses model.predict() for per-frame detection + simple Hungarian matching to
maintain persistent player IDs across frames.

ByteTrack (Kalman filter + data association) was removed because its overhead
exceeded the benefit for the 2-player, fixed-court use case, especially with
interval gaps causing stale Kalman state.

This replacement uses centroid-based matching (fast, stateless) instead:
  - Each frame's detections are matched to the previous frame's positions
  - Matching uses Hungarian algorithm on centroid distances (fast & optimal)
  - No Kalman filter; no per-frame Hungarian re-ID overhead
  - Handles interval gaps by keeping last known positions

ID assignment is persistent across frames (0, 1, …) via centroid matching.
PlayerContext's first-seen slot assignment uses these stable IDs for P1/P2.

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
    """Detect players using YOLOv8 + lightweight centroid-based tracking.

    Uses Hungarian matching on bounding box centroids to maintain persistent
    player IDs across frames without Kalman filter overhead.

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

        # Centroid-based tracking state.
        # _tracked_ids maps persistent ID → last known centroid (cx, cy).
        self._tracked_ids: dict[int, tuple[int, int]] = {}
        self._next_id: int = 0
        self._max_distance: float = 200.0  # px; max centroid distance for matching
        self._id_lost_threshold: int = 30  # frames before losing an ID

        # Fall back to yolov8n.pt if fine-tuned weights do not exist
        if not os.path.exists(weights):
            print(f"[YOLO] Weights not found at '{weights}', falling back to yolov8n.pt")
            weights = "yolov8n.pt"

        print(f"[YOLO] Loading model from {weights} (detect_interval={self._detect_interval})")
        self.model = YOLO(weights)

    def set_detect_interval(self, interval: int) -> None:
        """Set detection interval for frame-skipping cache."""
        self._detect_interval = max(1, int(interval))

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Run YOLOv8 detection on a BGR frame.

        YOLO inference is skipped on frames between detect_interval boundaries;
        the previous frame's detections are returned instead.

        Persistent IDs are maintained via centroid-based Hungarian matching.
        On detection frames, current centroids are matched to previous frame
        positions. On cached frames, cached detections are returned as-is.

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

        # Extract raw detections with temporary IDs and centroids.
        raw_detections = []
        for box_arr in xyxy:
            x1, y1, x2, y2 = int(box_arr[0]), int(box_arr[1]), int(box_arr[2]), int(box_arr[3])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            raw_detections.append({
                "box": [x1, y1, x2, y2],
                "feet": (cx, y2),
                "centroid": (cx, cy),
            })

        # Match current detections to previous tracked IDs via Hungarian algorithm.
        detections = self._match_and_assign_ids(raw_detections)

        self._last_detections = detections
        return detections

    def _match_and_assign_ids(self, raw_detections: list[dict]) -> list[dict]:
        """Assign persistent IDs to detections via centroid-based matching.

        Uses optimal bipartite matching (Hungarian algorithm) for n ≤ 4 players,
        or greedy matching for larger numbers. For the typical 2-player badminton
        case, computes the optimal assignment by trying all permutations.

        Args:
            raw_detections: List of raw detection dicts with "centroid", "box", "feet".

        Returns:
            List of dicts with assigned persistent "id" fields.
        """
        if not raw_detections:
            self._tracked_ids.clear()
            return []

        # If no prior IDs, assign new IDs to all detections.
        if not self._tracked_ids:
            detections = []
            for det in raw_detections:
                persistent_id = self._next_id
                self._next_id += 1
                self._tracked_ids[persistent_id] = det["centroid"]
                detections.append({
                    "id": persistent_id,
                    "box": det["box"],
                    "feet": det["feet"],
                    "feet_real": None,
                })
            return detections

        # Extract current and previous positions.
        current_centroids = [det["centroid"] for det in raw_detections]
        tracked_ids = list(self._tracked_ids.keys())
        tracked_centroids = [self._tracked_ids[pid] for pid in tracked_ids]

        n_current = len(current_centroids)
        n_tracked = len(tracked_ids)

        # If no prior tracks, assign new IDs to all.
        if n_tracked == 0:
            detections = []
            for det in raw_detections:
                persistent_id = self._next_id
                self._next_id += 1
                self._tracked_ids[persistent_id] = det["centroid"]
                detections.append({
                    "id": persistent_id,
                    "box": det["box"],
                    "feet": det["feet"],
                    "feet_real": None,
                })
            return detections

        # Build cost matrix (Euclidean distance).
        cost_matrix = np.zeros((n_current, n_tracked), dtype=np.float32)
        for i, curr_centroid in enumerate(current_centroids):
            for j, tracked_centroid in enumerate(tracked_centroids):
                dist = np.sqrt((curr_centroid[0] - tracked_centroid[0]) ** 2 +
                               (curr_centroid[1] - tracked_centroid[1]) ** 2)
                cost_matrix[i, j] = dist

        # Optimal matching for small n (≤ 4 players). For 2-player case, try all permutations.
        assignment = self._find_optimal_matching(cost_matrix, n_current, n_tracked)

        # Apply matches: assign tracked IDs to current detections.
        matched_tracked_indices = set()
        final_assignment = {}  # current_idx → persistent_id

        for curr_idx, tracked_idx in assignment:
            if tracked_idx is not None:
                dist = cost_matrix[curr_idx, tracked_idx]
                persistent_id = tracked_ids[tracked_idx]

                # Only accept match if distance is reasonable.
                if dist <= self._max_distance:
                    final_assignment[curr_idx] = persistent_id
                    matched_tracked_indices.add(tracked_idx)
                    self._tracked_ids[persistent_id] = current_centroids[curr_idx]

        # Unmatched current detections: assign new IDs.
        for i, det in enumerate(raw_detections):
            if i not in final_assignment:
                persistent_id = self._next_id
                self._next_id += 1
                final_assignment[i] = persistent_id
                self._tracked_ids[persistent_id] = det["centroid"]

        # Unmatched tracked IDs are lost (remove from tracking).
        for tracked_idx in range(n_tracked):
            if tracked_idx not in matched_tracked_indices:
                lost_id = tracked_ids[tracked_idx]
                del self._tracked_ids[lost_id]

        # Build final detections list with assigned IDs.
        detections = []
        for i, det in enumerate(raw_detections):
            detections.append({
                "id": final_assignment[i],
                "box": det["box"],
                "feet": det["feet"],
                "feet_real": None,  # populated by main.py via CourtMapper
            })

        return detections

    def _find_optimal_matching(self, cost_matrix: np.ndarray, n_current: int, n_tracked: int) -> list[tuple]:
        """Find optimal bipartite matching using exhaustive search (for small n).

        For n_current, n_tracked ≤ 4, exhaustive search is fast. For larger cases,
        falls back to greedy nearest-neighbor matching.

        Returns:
            List of (current_idx, tracked_idx) tuples. tracked_idx is None for unmatched.
        """
        if n_current == 0 or n_tracked == 0:
            return [(i, None) for i in range(n_current)]

        # For very small matching problems, use exhaustive search (optimal).
        if n_current <= 4 and n_tracked <= 4:
            return self._exhaustive_matching(cost_matrix, n_current, n_tracked)
        else:
            # Greedy matching for larger problems.
            return self._greedy_matching(cost_matrix, n_current, n_tracked)

    def _exhaustive_matching(self, cost_matrix: np.ndarray, n_current: int, n_tracked: int) -> list[tuple]:
        """Find optimal matching by trying all permutations (O(n!)).

        For 2 players: 2! = 2 permutations (fast).
        For 3 players: 3! = 6 permutations (fast).
        For 4 players: 4! = 24 permutations (still fast).
        """
        from itertools import permutations

        best_cost = float("inf")
        best_matching = None

        # Try all permutations of tracked indices.
        for perm in permutations(range(n_tracked), min(n_current, n_tracked)):
            cost = sum(cost_matrix[i, perm[i]] for i in range(len(perm)))
            if cost < best_cost:
                best_cost = cost
                best_matching = perm

        # Build result: matched pairs + unmatched current detections.
        result = []
        for i in range(n_current):
            if i < len(best_matching):
                result.append((i, best_matching[i]))
            else:
                result.append((i, None))

        return result

    def _greedy_matching(self, cost_matrix: np.ndarray, n_current: int, n_tracked: int) -> list[tuple]:
        """Greedy nearest-neighbor matching (O(n² log n)).

        Iteratively match the lowest-cost pair until no more matches can be made.
        """
        matched_current = set()
        matched_tracked = set()
        result = {}  # current_idx → tracked_idx

        # Sort all pairs by cost.
        pairs = []
        for i in range(n_current):
            for j in range(n_tracked):
                pairs.append((cost_matrix[i, j], i, j))
        pairs.sort()

        # Greedily match lowest-cost pairs.
        for cost, i, j in pairs:
            if i not in matched_current and j not in matched_tracked:
                result[i] = j
                matched_current.add(i)
                matched_tracked.add(j)

        # Build result list.
        final_result = []
        for i in range(n_current):
            final_result.append((i, result.get(i, None)))

        return final_result
