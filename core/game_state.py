"""Game state management module.
Tracks rally status and hit counts.

Simplified rally detection:
- Motion streak (require motion_required_streak=2 consecutive frames of >= min_displacement_px=5px)
- Inactive timeout (rally ends after inactive_timeout_s with no motion)
- Detection grace period (detection_grace_frames consecutive misses tolerated mid-rally)
- Hard minimum rally duration (rally_min_duration_s)
- No trajectory prediction, no stability detector, no large-displacement filter.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

ShuttleTuple = Tuple[float, float, float, float, float]


class GameState:
    def __init__(
        self,
        cfg=None,
        inactive_timeout_s: float = 1.0,
        min_displacement_px: float = 5.0,
    ):
        # Read from cfg if provided; explicit args act as fallback defaults.
        if cfg is not None:
            inactive_timeout_s = float(getattr(cfg, "rally_inactive_timeout_s", inactive_timeout_s))
            min_displacement_px = float(getattr(cfg, "rally_min_displacement_px", min_displacement_px))
            motion_required_streak = int(getattr(cfg, "rally_motion_required_streak", 2))
            min_duration_s = float(getattr(cfg, "rally_min_duration_s", 0.1))
            detection_grace_frames = int(getattr(cfg, "rally_detection_grace_frames", 8))
            min_hits = int(getattr(cfg, "rally_min_hits", 0))
        else:
            motion_required_streak = 2
            min_duration_s = 0.1
            detection_grace_frames = 8
            min_hits = 0

        self.inactive_timeout_s = float(inactive_timeout_s)
        self.min_displacement_px = float(min_displacement_px)

        self.rally_active = False
        self.last_center: Optional[Tuple[float, float]] = None
        self.last_motion_timestamp: Optional[float] = None
        self.current_rally_start_timestamp: Optional[float] = None
        self.rally_data: list = []

        self.score = {"player1": 0, "player2": 0}
        self.hit_count = 0

        # Motion-debounce: require this many consecutive frames of real motion
        # before starting a rally.
        self.motion_streak: int = 0
        self.motion_required_streak: int = motion_required_streak

        # Timestamp of the first frame in the current streak (used to back-fill
        # rally start once the streak is confirmed).
        self._streak_start_timestamp: Optional[float] = None

        # Hard minimum rally duration.
        self.min_duration_s: float = min_duration_s

        # Minimum confirmed-hit count (0 = no gate).
        self.min_hits: int = min_hits

        # Grace period: tolerate up to this many consecutive missed detections
        # mid-rally before treating them as a genuine loss of shuttle.
        self.detection_grace_frames: int = detection_grace_frames
        self._miss_streak: int = 0

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
        """End the current rally and reset all per-rally tracking state."""
        if winner in self.score:
            self.score[winner] += 1
        self.rally_active = False
        self.hit_count = 0

        # Full state reset.
        self.last_center = None
        self.last_motion_timestamp = None
        self.motion_streak = 0
        self._streak_start_timestamp = None
        self._miss_streak = 0

        print("[GAME] rally_end")

    def _record_rally_segment(self, end_timestamp: float):
        """Record a rally segment if it passes quality gates."""
        if self.current_rally_start_timestamp is None:
            return

        duration_s = max(float(end_timestamp) - float(self.current_rally_start_timestamp), 0.0)

        if duration_s < self.min_duration_s:
            print(
                f"[GAME] rally_discarded (too short: {duration_s:.3f}s < {self.min_duration_s}s)"
            )
            self.current_rally_start_timestamp = None
            return

        if self.hit_count < self.min_hits:
            print(
                f"[GAME] rally_discarded (too few hits: {self.hit_count} < {self.min_hits}) "
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

    # ------------------------------------------------------------------
    # Main update loop
    # ------------------------------------------------------------------

    def update(
        self,
        timestamp: float,
        shuttle_det: Optional[ShuttleTuple],
        frame_size: Optional[Tuple[int, int]] = None,  # kept for API compat; not used
    ) -> bool:
        """Update state from one frame; return current rally_active.

        Grace period
            Up to detection_grace_frames consecutive frames with no detection
            are silently tolerated mid-rally.  last_motion_timestamp is NOT aged
            until the grace window expires.

        Motion streak
            motion_required_streak consecutive frames each with displacement >=
            min_displacement_px are required to start a rally.  Once confirmed,
            current_rally_start_timestamp is back-filled to _streak_start_timestamp.
        """
        center = self._center_from_shuttle(shuttle_det)

        # ── No detection this frame ────────────────────────────────────────────
        if center is None:
            self._miss_streak += 1
            # Grace window: don't update last_motion_timestamp while within grace.
            # Fall through to inactive-timeout check below.

        # ── Detection present ──────────────────────────────────────────────────
        else:
            self._miss_streak = 0  # reset grace counter

            if self.last_center is None:
                # First detection after cold-start / post-rally reset.
                self.motion_streak = 1
                self._streak_start_timestamp = timestamp
            else:
                dx = center[0] - self.last_center[0]
                dy = center[1] - self.last_center[1]
                displacement = math.hypot(dx, dy)

                if displacement >= self.min_displacement_px:
                    self.motion_streak = min(self.motion_streak + 1, self.motion_required_streak)

                    if self.motion_streak >= self.motion_required_streak:
                        self.last_motion_timestamp = timestamp

                        if not self.rally_active:
                            # Back-fill start to first frame of confirmed streak.
                            self.current_rally_start_timestamp = (
                                self._streak_start_timestamp
                                if self._streak_start_timestamp is not None
                                else timestamp
                            )
                            self.start_rally()
                else:
                    # Jitter — reset streak.
                    self.motion_streak = 0
                    self._streak_start_timestamp = None

            self.last_center = center

        # ── Inactive timeout check ─────────────────────────────────────────────
        if self.rally_active:
            if self.last_motion_timestamp is None:
                self._record_rally_segment(timestamp)
                self.end_rally()
            elif (timestamp - self.last_motion_timestamp) >= self.inactive_timeout_s:
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
            return self.update(timestamp=timestamp, shuttle_det=shuttle_det)

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
    periods is merged into those rally periods.

    Args:
        rally_data:    List of rally dicts from GameState.get_rally_data().
        total_frames:  Total frames in the video.
        fps:           Video frame rate.
        min_period_s:  Minimum inactive-gap duration to keep as a gap.

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
