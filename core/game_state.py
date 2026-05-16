"""Game state management module.
Tracks rally status, score, and hit counts.

Direct port of slayminton/core/game_state.py.
Only change: __init__ accepts a cfg parameter to read thresholds from config.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np


ShuttleTuple = Tuple[float, float, float, float, float]


class GameState:
    def __init__(self, cfg=None, inactive_timeout_s: float = 1.0, min_displacement_px: float = 2.0):
        # Read from cfg if provided; explicit args act as fallback defaults.
        if cfg is not None:
            inactive_timeout_s = float(getattr(cfg, "rally_inactive_timeout_s", inactive_timeout_s))
            min_displacement_px = float(getattr(cfg, "rally_min_displacement_px", min_displacement_px))
            motion_required_streak = int(getattr(cfg, "rally_motion_required_streak", 3))
            max_displacement_fraction = float(getattr(cfg, "rally_max_displacement_fraction", 1.0 / 6.0))
            stable_frame_threshold = int(getattr(cfg, "rally_stable_frame_threshold", 5))
        else:
            motion_required_streak = 3
            max_displacement_fraction = 1.0 / 6.0
            stable_frame_threshold = 5

        # Motion-based rally rule config.
        self.inactive_timeout_s = float(inactive_timeout_s)
        self.min_displacement_px = float(min_displacement_px)

        self.rally_active = False
        self.last_center: Optional[Tuple[float, float]] = None
        self.last_motion_timestamp: Optional[float] = None
        self.current_rally_start_timestamp: Optional[float] = None
        self.rally_data = []

        self.score = {"player1": 0, "player2": 0}
        self.hit_count = 0

        # Trajectory prediction-based hit detection.
        self.position_history: List[Tuple[float, float, float]] = []  # (timestamp, x, y)
        self.history_max_len = 8
        self.prediction_error_threshold = 20.0
        self.last_hit_timestamp: Optional[float] = None
        self.hit_cooldown_s = 0.2

        # Motion-debounce
        self.motion_streak: int = 0
        self.motion_required_streak: int = motion_required_streak

        # Stability detector
        self.stable_frames: int = 0
        self.stable_frame_threshold: int = stable_frame_threshold

        # Large displacement filter
        self.max_displacement_fraction: float = max_displacement_fraction

        # Internal state flags (used by update() logic only)
        self.last_detection_discarded: bool = False
        self.consecutive_stationary_frames: int = 0

    def start_rally(self):
        self.rally_active = True
        self.hit_count = 0
        ts = self.current_rally_start_timestamp
        if ts is None:
            print("[GAME] rally_start")
        else:
            print(f"[GAME] rally_start t={ts:.3f}s")

    def end_rally(self, winner=None):
        if winner in self.score:
            self.score[winner] += 1
        self.rally_active = False
        self.hit_count = 0
        print("[GAME] rally_end")

    def _record_rally_segment(self, end_timestamp: float):
        if self.current_rally_start_timestamp is None:
            return
        duration_s = max(float(end_timestamp) - float(self.current_rally_start_timestamp), 0.0)
        self.rally_data.append(
            {
                "rally_id": len(self.rally_data) + 1,
                "start_time": float(self.current_rally_start_timestamp),
                "end_time": float(end_timestamp),
                "duration_s": float(duration_s),
            }
        )
        print(
            f"[GAME] rally_segment id={len(self.rally_data)} "
            f"start={self.current_rally_start_timestamp:.3f}s end={float(end_timestamp):.3f}s "
            f"duration={duration_s:.3f}s"
        )
        self.current_rally_start_timestamp = None

    def record_hit(self):
        if self.rally_active:
            self.hit_count += 1
            if self.last_motion_timestamp is not None:
                print(f"[GAME] hit t={self.last_motion_timestamp:.3f}s count={self.hit_count}")
            else:
                print(f"[GAME] hit count={self.hit_count}")

    # should_visualize_shuttle() removed — rendering decisions belong in the
    # visualization layer, not in game state.  Callers that need to suppress
    # the shuttle dot can inspect last_detection_discarded and
    # consecutive_stationary_frames directly, or use the shuttle field in
    # tracking_results.json (None when discarded).

    @staticmethod
    def _center_from_shuttle(shuttle_det: Optional[ShuttleTuple]) -> Optional[Tuple[float, float]]:
        if shuttle_det is None:
            return None
        _, x, y, w, h = shuttle_det
        return (x + 0.5 * w, y + 0.5 * h)

    def _fit_trajectory(self) -> Optional[Tuple[float, float, float]]:
        if len(self.position_history) < 2:
            return None
        hist = self.position_history
        t1, x1, y1 = hist[-2]
        t2, x2, y2 = hist[-1]
        dt = max(t2 - t1, 1e-6)
        vx = (x2 - x1) / dt
        vy = (y2 - y1) / dt
        speed = math.hypot(vx, vy)
        return (vx, vy, speed)

    def _predict_position(self, next_timestamp: float) -> Optional[Tuple[float, float]]:
        if len(self.position_history) < 2:
            return None
        trajectory = self._fit_trajectory()
        if trajectory is None:
            return None
        vx, vy, _ = trajectory
        _, last_x, last_y = self.position_history[-1]
        last_t = self.position_history[-1][0]
        dt = next_timestamp - last_t
        pred_x = last_x + vx * dt
        pred_y = last_y + vy * dt
        return (pred_x, pred_y)

    def _detect_hit(self, current_pos: Tuple[float, float], current_timestamp: float) -> bool:
        if len(self.position_history) < 2:
            return False
        if self.last_hit_timestamp is not None:
            if (current_timestamp - self.last_hit_timestamp) < self.hit_cooldown_s:
                return False
        pred = self._predict_position(current_timestamp)
        if pred is None:
            return False
        pred_x, pred_y = pred
        cur_x, cur_y = current_pos
        error = math.hypot(cur_x - pred_x, cur_y - pred_y)
        if error > self.prediction_error_threshold:
            self.last_hit_timestamp = current_timestamp
            return True
        return False

    def update(
        self,
        timestamp: float,
        shuttle_det: Optional[ShuttleTuple],
        frame_size: Optional[Tuple[int, int]] = None,
    ) -> bool:
        center = self._center_from_shuttle(shuttle_det)
        self.last_detection_discarded = False

        if center is None:
            self.stable_frames = 0
            self.consecutive_stationary_frames = 0
        else:
            if self.last_center is not None:
                dx = center[0] - self.last_center[0]
                dy = center[1] - self.last_center[1]
                displacement = math.hypot(dx, dy)

                if frame_size is not None:
                    fh, fw = frame_size
                    max_allowed = max(fh, fw) * self.max_displacement_fraction
                    if displacement > max_allowed:
                        self.motion_streak = 0
                        self.last_detection_discarded = True
                        self.consecutive_stationary_frames = 0
                        return self.rally_active

                if displacement < 1e-3:
                    self.stable_frames += 1
                    self.consecutive_stationary_frames += 1
                    if self.stable_frames > self.stable_frame_threshold:
                        if self.rally_active:
                            self._record_rally_segment(timestamp)
                            self.end_rally()
                        self.last_center = None
                        self.last_motion_timestamp = None
                        self.position_history.clear()
                        self.motion_streak = 0
                        self.consecutive_stationary_frames = 0
                        return self.rally_active
                else:
                    self.stable_frames = 0
                    self.consecutive_stationary_frames = 0

            self.position_history.append((timestamp, center[0], center[1]))
            if len(self.position_history) > self.history_max_len:
                self.position_history.pop(0)

            if self.last_center is None:
                self.motion_streak = self.motion_required_streak
                self.last_motion_timestamp = timestamp
                if not self.rally_active:
                    self.current_rally_start_timestamp = timestamp
                self.start_rally()
                self.consecutive_stationary_frames = 0
            else:
                dx = center[0] - self.last_center[0]
                dy = center[1] - self.last_center[1]
                displacement = math.hypot(dx, dy)

                if displacement >= self.min_displacement_px:
                    self.motion_streak = min(self.motion_streak + 1, self.motion_required_streak)
                    if self.motion_streak >= self.motion_required_streak:
                        self.last_motion_timestamp = timestamp
                        if not self.rally_active:
                            self.current_rally_start_timestamp = timestamp
                            self.start_rally()
                        if self._detect_hit(center, timestamp):
                            self.record_hit()
                    self.consecutive_stationary_frames = 0
                else:
                    self.motion_streak = 0
                    self.consecutive_stationary_frames += 1

            self.last_center = center

        if self.rally_active:
            if self.last_motion_timestamp is None:
                self._record_rally_segment(timestamp)
                self.end_rally()
            elif (timestamp - self.last_motion_timestamp) >= self.inactive_timeout_s:
                self._record_rally_segment(timestamp)
                self.end_rally()

        return self.rally_active

    def update_game_state(
        self,
        player_detected,
        shuttle_detected,
        timestamp: float = 0.0,
        shuttle_det=None,
    ):
        """Compatibility wrapper using the original method name."""
        if shuttle_det is not None:
            fs = None
            if isinstance(shuttle_det, tuple) and len(shuttle_det) >= 6:
                try:
                    fh = int(shuttle_det[5])
                    fw = int(shuttle_det[6])
                    fs = (fh, fw)
                except Exception:
                    fs = None
            return self.update(timestamp=timestamp, shuttle_det=shuttle_det, frame_size=fs)

        if player_detected and shuttle_detected:
            if not self.rally_active:
                self.current_rally_start_timestamp = timestamp
                self.start_rally()
            self.record_hit()
        elif self.rally_active and (not player_detected or not shuttle_detected):
            self._record_rally_segment(timestamp)
            self.end_rally()
        return self.rally_active

    def finalize_rally_data(self, final_timestamp: float):
        if self.rally_active:
            self._record_rally_segment(final_timestamp)
            self.end_rally()
        print(f"[GAME] finalize rallies={len(self.rally_data)} at t={float(final_timestamp):.3f}s")

    def get_rally_data(self):
        return list(self.rally_data)


def build_rally_status_per_frame(
    rally_data: list,
    total_frames: int,
    fps: float,
) -> tuple[list[bool], list]:
    """Build a per-frame rally active status and consolidated rally data.

    Consolidates short inactive gaps (<0.5s) into surrounding active periods.
    """
    frame_duration = 1.0 / max(fps, 1e-6)
    min_period_duration = 0.5
    min_period_frames = max(1, int(min_period_duration / frame_duration))

    rally_status = [False] * total_frames

    for rally in rally_data:
        start_frame = max(0, int(rally["start_time"] * fps))
        end_frame = min(total_frames - 1, int(rally["end_time"] * fps))
        for i in range(start_frame, end_frame + 1):
            if i < len(rally_status):
                rally_status[i] = True

    consolidated = rally_status.copy()
    i = 0
    while i < len(consolidated):
        current_state = consolidated[i]
        start_i = i
        while i < len(consolidated) and consolidated[i] == current_state:
            i += 1
        period_len = i - start_i

        if period_len < min_period_frames and current_state == False and start_i > 0:
            next_state = consolidated[i] if i < len(consolidated) else None
            prev_state = consolidated[start_i - 1]
            if next_state == True or prev_state == True:
                merge_into = True
                for j in range(start_i, i):
                    consolidated[j] = merge_into

    consolidated_rally_data = []
    rally_id = 1
    in_rally = False
    rally_start = None

    for frame_idx, is_rally in enumerate(consolidated):
        if is_rally and not in_rally:
            rally_start = frame_idx * frame_duration
            in_rally = True
        elif not is_rally and in_rally:
            rally_end = frame_idx * frame_duration
            duration = max(rally_end - rally_start, 0.0)
            consolidated_rally_data.append({
                "rally_id": rally_id,
                "start_time": float(rally_start),
                "end_time": float(rally_end),
                "duration_s": float(duration),
            })
            rally_id += 1
            in_rally = False

    if in_rally and rally_start is not None:
        rally_end = (total_frames - 1) * frame_duration
        duration = max(rally_end - rally_start, 0.0)
        consolidated_rally_data.append({
            "rally_id": rally_id,
            "start_time": float(rally_start),
            "end_time": float(rally_end),
            "duration_s": float(duration),
        })

    return consolidated, consolidated_rally_data
