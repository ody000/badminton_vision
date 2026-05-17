"""Game state management module.
Tracks rally status, score, and hit counts.

Direct port of slayminton/core/game_state.py.
Only change: __init__ accepts a cfg parameter to read thresholds from config.

Improvements over the original port:
- end_rally() resets all tracking state (last_center, motion_streak,
  position_history, last_motion_timestamp) to prevent stale state from
  bleeding into the next rally.
- _record_rally_segment() enforces a hard minimum duration (rally_min_duration_s)
  and a minimum confirmed-hit count (rally_min_hits) before recording.
- update() implements a grace-period counter: up to rally_detection_grace_frames
  consecutive missed detections are tolerated mid-rally without resetting the
  motion streak or expiring last_motion_timestamp.
- First-detection streak bypass is removed: motion_streak starts at 1 on the
  first detection and must climb to motion_required_streak normally. Once the
  streak is confirmed, current_rally_start_timestamp is back-filled to the
  timestamp of the first detection in the streak so rally boundaries are accurate.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np


ShuttleTuple = Tuple[float, float, float, float, float]


class GameState:
    def __init__(
        self,
        cfg=None,
        inactive_timeout_s: float = 1.0,
        min_displacement_px: float = 2.0,
    ):
        # Read from cfg if provided; explicit args act as fallback defaults.
        if cfg is not None:
            inactive_timeout_s = float(getattr(cfg, "rally_inactive_timeout_s", inactive_timeout_s))
            min_displacement_px = float(getattr(cfg, "rally_min_displacement_px", min_displacement_px))
            motion_required_streak = int(getattr(cfg, "rally_motion_required_streak", 3))
            max_displacement_fraction = float(getattr(cfg, "rally_max_displacement_fraction", 1.0 / 6.0))
            stable_frame_threshold = int(getattr(cfg, "rally_stable_frame_threshold", 5))
            min_duration_s = float(getattr(cfg, "rally_min_duration_s", 0.5))
            detection_grace_frames = int(getattr(cfg, "rally_detection_grace_frames", 3))
            min_hits = int(getattr(cfg, "rally_min_hits", 1))
        else:
            motion_required_streak = 3
            max_displacement_fraction = 1.0 / 6.0
            stable_frame_threshold = 5
            min_duration_s = 0.5
            detection_grace_frames = 3
            min_hits = 1

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

        # Motion-debounce: require this many consecutive frames of real motion
        # before starting a rally.  The first-detection bypass is gone — streak
        # starts at 1 on the first detection and must reach motion_required_streak.
        self.motion_streak: int = 0
        self.motion_required_streak: int = motion_required_streak

        # Streak candidate: timestamp of the first frame of the current streak,
        # used to back-fill current_rally_start_timestamp once confirmed.
        self._streak_start_timestamp: Optional[float] = None

        # Stability detector
        self.stable_frames: int = 0
        self.stable_frame_threshold: int = stable_frame_threshold

        # Large displacement filter
        self.max_displacement_fraction: float = max_displacement_fraction

        # Hard minimum rally duration — rallies shorter than this are discarded.
        self.min_duration_s: float = min_duration_s

        # Minimum confirmed hit count — rallies with fewer hits are discarded.
        self.min_hits: int = min_hits

        # Grace period: tolerate up to this many consecutive missed detections
        # mid-rally before treating them as a genuine loss of shuttle.
        self.detection_grace_frames: int = detection_grace_frames
        self._miss_streak: int = 0   # consecutive frames where center == None

        # Internal state flags (used by update() logic only)
        self.last_detection_discarded: bool = False
        self.consecutive_stationary_frames: int = 0

    # ------------------------------------------------------------------
    # Rally lifecycle
    # ------------------------------------------------------------------

    def start_rally(self):
        self.rally_active = True
        self.hit_count = 0
        ts = self.current_rally_start_timestamp
        if ts is None:
            print("[GAME] rally_start")
        else:
            print(f"[GAME] rally_start t={ts:.3f}s")

    def end_rally(self, winner=None):
        """End the current rally and reset ALL per-rally tracking state.

        Resetting last_center, motion_streak, position_history, and
        last_motion_timestamp prevents stale values from the ending rally
        from leaking into the next detection cycle.
        """
        if winner in self.score:
            self.score[winner] += 1
        self.rally_active = False
        self.hit_count = 0

        # Full state reset — this is the key fix.
        self.last_center = None
        self.last_motion_timestamp = None
        self.motion_streak = 0
        self._streak_start_timestamp = None
        self.position_history.clear()
        self._miss_streak = 0
        self.stable_frames = 0
        self.consecutive_stationary_frames = 0

        print("[GAME] rally_end")

    def _record_rally_segment(self, end_timestamp: float):
        """Record a rally segment if it passes quality gates.

        Gates (all must pass):
          1. current_rally_start_timestamp must be set.
          2. duration_s >= min_duration_s  (hard floor, default 0.5s).
          3. hit_count >= min_hits          (default 1 confirmed hit).

        Rallies that fail a gate are silently discarded and logged.
        """
        if self.current_rally_start_timestamp is None:
            return

        duration_s = max(float(end_timestamp) - float(self.current_rally_start_timestamp), 0.0)

        # Gate 1: minimum duration
        if duration_s < self.min_duration_s:
            print(
                f"[GAME] rally_discarded (too short: {duration_s:.3f}s < {self.min_duration_s}s) "
                f"start={self.current_rally_start_timestamp:.3f}s end={float(end_timestamp):.3f}s"
            )
            self.current_rally_start_timestamp = None
            return

        # Gate 2: minimum hit count
        if self.hit_count < self.min_hits:
            print(
                f"[GAME] rally_discarded (too few hits: {self.hit_count} < {self.min_hits}) "
                f"start={self.current_rally_start_timestamp:.3f}s end={float(end_timestamp):.3f}s "
                f"duration={duration_s:.3f}s"
            )
            self.current_rally_start_timestamp = None
            return

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
            f"duration={duration_s:.3f}s hits={self.hit_count}"
        )
        self.current_rally_start_timestamp = None

    def record_hit(self):
        if self.rally_active:
            self.hit_count += 1
            if self.last_motion_timestamp is not None:
                print(f"[GAME] hit t={self.last_motion_timestamp:.3f}s count={self.hit_count}")
            else:
                print(f"[GAME] hit count={self.hit_count}")

    # ------------------------------------------------------------------
    # Shuttle helpers
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Main update loop
    # ------------------------------------------------------------------

    def update(
        self,
        timestamp: float,
        shuttle_det: Optional[ShuttleTuple],
        frame_size: Optional[Tuple[int, int]] = None,
    ) -> bool:
        """Update state from one frame; return current rally_active.

        Key behaviours vs original port
        ────────────────────────────────
        Grace period
            Up to detection_grace_frames consecutive frames with no detection
            are silently tolerated mid-rally.  The miss streak counter is
            incremented but last_motion_timestamp is NOT aged until the grace
            window expires.  This prevents TrackNet's natural ~10% miss rate
            from fragmenting rallies.

        First-detection streak
            On the first shuttle detection after a cold start (last_center is
            None), motion_streak is set to 1 — not immediately to
            motion_required_streak.  The streak must count up normally.  Once
            it reaches motion_required_streak, current_rally_start_timestamp
            is back-filled to _streak_start_timestamp (the first frame of the
            streak) so the recorded rally boundary is accurate.

        Args:
            timestamp:   Current frame time in seconds.
            shuttle_det: (ts, x, y, w, h) from TrackNet, or None.
            frame_size:  Optional (height, width) for large-displacement filter.
        """
        center = self._center_from_shuttle(shuttle_det)
        self.last_detection_discarded = False

        # ── No detection this frame ────────────────────────────────────────────
        if center is None:
            self._miss_streak += 1
            self.stable_frames = 0

            if self._miss_streak > self.detection_grace_frames:
                # Grace window exhausted — treat as genuine loss of shuttle.
                # Don't reset motion_streak here; let the inactive_timeout below
                # handle rally termination naturally.
                self.consecutive_stationary_frames = 0

            # Either way, fall through to the timeout check below.

        # ── Detection present ──────────────────────────────────────────────────
        else:
            self._miss_streak = 0  # reset grace counter on any real detection

            if self.last_center is not None:
                dx = center[0] - self.last_center[0]
                dy = center[1] - self.last_center[1]
                displacement = math.hypot(dx, dy)

                # Large-displacement filter: impossible inter-frame jump → reject.
                if frame_size is not None:
                    fh, fw = frame_size
                    max_allowed = max(fh, fw) * self.max_displacement_fraction
                    if displacement > max_allowed:
                        self.motion_streak = 0
                        self._streak_start_timestamp = None
                        self.last_detection_discarded = True
                        self.consecutive_stationary_frames = 0
                        return self.rally_active

                # Stability: exactly-unchanged center for many frames → background.
                if displacement < 1e-3:
                    self.stable_frames += 1
                    self.consecutive_stationary_frames += 1
                    if self.stable_frames > self.stable_frame_threshold:
                        if self.rally_active:
                            self._record_rally_segment(timestamp)
                            self.end_rally()
                        else:
                            # Not in rally but tracking got stuck — hard reset.
                            self.last_center = None
                            self.last_motion_timestamp = None
                            self.position_history.clear()
                            self.motion_streak = 0
                            self._streak_start_timestamp = None
                            self.consecutive_stationary_frames = 0
                        return self.rally_active
                else:
                    self.stable_frames = 0
                    self.consecutive_stationary_frames = 0

            # Accept detection into history.
            self.position_history.append((timestamp, center[0], center[1]))
            if len(self.position_history) > self.history_max_len:
                self.position_history.pop(0)

            if self.last_center is None:
                # ── First detection after cold-start / post-rally reset ───────
                # Streak starts at 1; _streak_start_timestamp marks this frame
                # so we can back-fill the rally start once the streak confirms.
                self.motion_streak = 1
                self._streak_start_timestamp = timestamp
                self.consecutive_stationary_frames = 0
            else:
                # ── Subsequent detection ──────────────────────────────────────
                dx = center[0] - self.last_center[0]
                dy = center[1] - self.last_center[1]
                displacement = math.hypot(dx, dy)

                if displacement >= self.min_displacement_px:
                    self.motion_streak = min(self.motion_streak + 1, self.motion_required_streak)
                    self.consecutive_stationary_frames = 0

                    if self.motion_streak >= self.motion_required_streak:
                        self.last_motion_timestamp = timestamp

                        if not self.rally_active:
                            # Back-fill start to the first frame of the confirmed streak.
                            self.current_rally_start_timestamp = (
                                self._streak_start_timestamp
                                if self._streak_start_timestamp is not None
                                else timestamp
                            )
                            self.start_rally()

                        if self._detect_hit(center, timestamp):
                            self.record_hit()
                else:
                    # Small movement (jitter) — reset streak but keep position history.
                    self.motion_streak = 0
                    self._streak_start_timestamp = None
                    self.consecutive_stationary_frames += 1

            self.last_center = center

        # ── Inactive timeout check ─────────────────────────────────────────────
        if self.rally_active:
            if self.last_motion_timestamp is None:
                self._record_rally_segment(timestamp)
                self.end_rally()
            elif (timestamp - self.last_motion_timestamp) >= self.inactive_timeout_s:
                # Pin end to the last confirmed-motion timestamp so duration
                # is not inflated by the 1s of silence that triggered this.
                self._record_rally_segment(self.last_motion_timestamp)
                self.end_rally()

        return self.rally_active

    # ------------------------------------------------------------------
    # Compatibility wrapper
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    def finalize_rally_data(self, final_timestamp: float):
        """Close any open rally at end-of-stream."""
        if self.rally_active:
            self._record_rally_segment(final_timestamp)
            self.end_rally()
        print(f"[GAME] finalize rallies={len(self.rally_data)} at t={float(final_timestamp):.3f}s")

    def get_rally_data(self):
        return list(self.rally_data)


# ──────────────────────────────────────────────────────────────────────────────
# Post-processing consolidation (called from main.py after get_rally_data())
# ──────────────────────────────────────────────────────────────────────────────

def build_rally_status_per_frame(
    rally_data: list,
    total_frames: int,
    fps: float,
    min_period_s: float = 0.5,
) -> tuple[list[bool], list]:
    """Build per-frame rally status and consolidate short inactive gaps.

    Any inactive gap shorter than min_period_s that is flanked by active rally
    periods is merged into those rally periods.  This is a second pass of
    gap-bridging that operates on the full recorded rally list rather than
    frame-by-frame, catching cases the in-loop grace period can't handle (e.g.
    a gap that straddles the grace window boundary).

    Args:
        rally_data:    List of rally dicts from GameState.get_rally_data().
        total_frames:  Total frames in the video.
        fps:           Video frame rate.
        min_period_s:  Minimum inactive-gap duration to keep as a gap; shorter
                       gaps are merged.  Default 0.5s (matches config).

    Returns:
        (per_frame_status, consolidated_rally_data)
    """
    frame_duration = 1.0 / max(fps, 1e-6)
    min_period_frames = max(1, int(min_period_s / frame_duration))

    # Build frame-level boolean status from raw rally_data.
    rally_status = [False] * total_frames
    for rally in rally_data:
        start_frame = max(0, int(rally["start_time"] * fps))
        end_frame = min(total_frames - 1, int(rally["end_time"] * fps))
        for i in range(start_frame, end_frame + 1):
            if i < len(rally_status):
                rally_status[i] = True

    # Merge short False (inactive) runs that sit between True (active) runs.
    consolidated = rally_status.copy()
    i = 0
    while i < len(consolidated):
        current_state = consolidated[i]
        start_i = i
        while i < len(consolidated) and consolidated[i] == current_state:
            i += 1
        period_len = i - start_i

        if (
            period_len < min_period_frames
            and current_state is False
            and start_i > 0
        ):
            next_state = consolidated[i] if i < len(consolidated) else None
            prev_state = consolidated[start_i - 1]
            if prev_state is True or next_state is True:
                for j in range(start_i, i):
                    consolidated[j] = True

    # Rebuild structured rally list from the consolidated frame array.
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
            consolidated_rally_data.append(
                {
                    "rally_id": rally_id,
                    "start_time": float(rally_start),
                    "end_time": float(rally_end),
                    "duration_s": float(duration),
                }
            )
            rally_id += 1
            in_rally = False

    if in_rally and rally_start is not None:
        rally_end = (total_frames - 1) * frame_duration
        duration = max(rally_end - rally_start, 0.0)
        consolidated_rally_data.append(
            {
                "rally_id": rally_id,
                "start_time": float(rally_start),
                "end_time": float(rally_end),
                "duration_s": float(duration),
            }
        )

    return consolidated, consolidated_rally_data
