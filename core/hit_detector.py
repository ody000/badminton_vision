"""HitDetector: RANSAC quadratic fit + proximity gate for shuttle hit detection.

Pipeline per frame:
  1. Append shuttle_pos to deque (maxlen = hit_trajectory_n). If None, return (False, None).
  2. Need >= 3 points to attempt fit.
  3. RANSAC quadratic fit: pick 3 random points, fit polyfit, count inliers (<=15px).
  4. If best fit has >= hit_ransac_min_inliers, predict next pos from fitted curve.
  5. If prediction error > threshold AND not in cooldown:
     - Find nearest player (real-world Euclidean) within hit_proximity_cm.
     - If found: confirm hit, set cooldown, clear buffer, return (True, player_id).
  6. Return (False, None) otherwise.
"""

from __future__ import annotations

import math
import random
from collections import deque
from typing import Optional

import numpy as np


class HitDetector:
    """Detects badminton hits via RANSAC quadratic trajectory fitting."""

    def __init__(self, cfg=None):
        if cfg is not None:
            self.hit_trajectory_n = int(getattr(cfg, "hit_trajectory_n", 6))
            self.hit_cooldown_s = float(getattr(cfg, "hit_cooldown_s", 0.2))
            self.hit_proximity_cm = float(getattr(cfg, "hit_proximity_cm", 200.0))
            self.hit_prediction_error_threshold = float(
                getattr(cfg, "hit_prediction_error_threshold", 20.0)
            )
            self.hit_ransac_iterations = int(getattr(cfg, "hit_ransac_iterations", 10))
            self.hit_ransac_min_inliers = int(getattr(cfg, "hit_ransac_min_inliers", 3))
        else:
            self.hit_trajectory_n = 6
            self.hit_cooldown_s = 0.2
            self.hit_proximity_cm = 200.0
            self.hit_prediction_error_threshold = 20.0
            self.hit_ransac_iterations = 10
            self.hit_ransac_min_inliers = 3

        # Inlier distance threshold for RANSAC (pixels)
        self._ransac_inlier_px = 15.0

        # Buffer of (timestamp, x, y) tuples
        self._buffer: deque = deque(maxlen=self.hit_trajectory_n)
        self._miss_count: int = 0
        self._last_hit_ts: Optional[float] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(
        self,
        timestamp: float,
        shuttle_pos: Optional[tuple],
        player_feet_real: list[dict],
    ) -> tuple[bool, Optional[int]]:
        """Update detector with latest shuttle position.

        Args:
            timestamp: Current frame timestamp (seconds).
            shuttle_pos: (x_px, y_px) shuttle center or None if not detected.
            player_feet_real: List of {"id": int, "feet_real": (x_cm, y_cm) | None}.

        Returns:
            (is_hit, player_id) — player_id is the ByteTrack int ID or None.
        """
        if shuttle_pos is None:
            self._miss_count += 1
            return False, None

        self._miss_count = 0
        x_px, y_px = float(shuttle_pos[0]), float(shuttle_pos[1])
        self._buffer.append((float(timestamp), x_px, y_px))

        if len(self._buffer) < 3:
            return False, None

        # RANSAC quadratic fit
        best_inliers, best_px, best_py = self._ransac_fit()

        if best_inliers < self.hit_ransac_min_inliers:
            return False, None

        # Predict next position from fitted curve
        pred_x, pred_y = self._predict_next(best_px, best_py, timestamp)

        # Compute prediction error vs latest actual position
        error = math.hypot(x_px - pred_x, y_px - pred_y)

        if error <= self.hit_prediction_error_threshold:
            return False, None

        # Cooldown check
        if self._last_hit_ts is not None:
            if (timestamp - self._last_hit_ts) < self.hit_cooldown_s:
                return False, None

        # Find nearest player within proximity threshold
        player_id = self._nearest_player(x_px, y_px, player_feet_real)
        if player_id is None:
            return False, None

        # Confirmed hit
        self._last_hit_ts = timestamp
        self._buffer.clear()
        return True, player_id

    def set_kalman_model(self, model) -> None:
        """No-op stub for future Kalman swap-in."""
        pass

    def reset(self) -> None:
        """Clear trajectory buffer."""
        self._buffer.clear()
        self._miss_count = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ransac_fit(self) -> tuple[int, np.ndarray, np.ndarray]:
        """Run RANSAC over buffer to find best quadratic fit.

        Returns:
            (best_inlier_count, poly_x_coeffs, poly_y_coeffs)
        """
        pts = list(self._buffer)
        n = len(pts)
        t_vals = np.array([p[0] for p in pts])
        x_vals = np.array([p[1] for p in pts])
        y_vals = np.array([p[2] for p in pts])

        best_inliers = 0
        best_px = np.array([0.0, 0.0, pts[-1][1]])
        best_py = np.array([0.0, 0.0, pts[-1][2]])

        for _ in range(self.hit_ransac_iterations):
            # Randomly pick 3 points
            idxs = random.sample(range(n), min(3, n))
            t_s = t_vals[idxs]
            x_s = x_vals[idxs]
            y_s = y_vals[idxs]

            # Need distinct t values to fit degree-2 polynomial
            if len(set(t_s)) < 2:
                continue

            deg = min(2, len(set(t_s)) - 1)
            try:
                px = np.polyfit(t_s, x_s, deg)
                py = np.polyfit(t_s, y_s, deg)
            except (np.linalg.LinAlgError, ValueError):
                continue

            # Count inliers: points where predicted position is within threshold
            pred_x = np.polyval(px, t_vals)
            pred_y = np.polyval(py, t_vals)
            dists = np.sqrt((x_vals - pred_x) ** 2 + (y_vals - pred_y) ** 2)
            inliers = int((dists <= self._ransac_inlier_px).sum())

            if inliers > best_inliers:
                best_inliers = inliers
                best_px = px
                best_py = py

        return best_inliers, best_px, best_py

    def _predict_next(
        self, px: np.ndarray, py: np.ndarray, t_next: float
    ) -> tuple[float, float]:
        """Predict position at t_next from fitted polynomial coefficients."""
        # Extrapolate one step past the last buffered timestamp using the
        # curve fitted over the buffer.
        pts = list(self._buffer)
        if len(pts) >= 2:
            dt = pts[-1][0] - pts[-2][0]
        else:
            dt = 1.0 / 30.0  # fallback: 30fps
        t_pred = pts[-1][0] + dt
        return float(np.polyval(px, t_pred)), float(np.polyval(py, t_pred))

    def _nearest_player(
        self,
        shuttle_x_px: float,
        shuttle_y_px: float,
        player_feet_real: list[dict],
    ) -> Optional[int]:
        """Find nearest player whose real-world feet are within hit_proximity_cm.

        Returns player ByteTrack id or None.
        """
        best_dist = float("inf")
        best_id = None

        for p in player_feet_real:
            feet_real = p.get("feet_real")
            if feet_real is None:
                continue
            fx_cm, fy_cm = float(feet_real[0]), float(feet_real[1])
            # Use real-world distance if available; fall back to pixel distance
            # (when no homography, feet_real may be in pixel space).
            dist = math.hypot(fx_cm - shuttle_x_px, fy_cm - shuttle_y_px)
            if dist < best_dist:
                best_dist = dist
                best_id = p.get("id")

        if best_dist <= self.hit_proximity_cm:
            return best_id
        return None
