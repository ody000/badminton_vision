"""Tier 1 smoke tests for GameState.

Tests:
  - Rally starts only after motion_required_streak consecutive frames of motion
  - Rally ends after inactive_timeout_s with no motion
  - end_rally() resets all per-rally state (last_center, motion_streak, etc.)
  - Grace period keeps rally alive through short detection gaps
  - Minimum duration filter discards sub-threshold rallies
  - Minimum hit-count filter discards 0-hit rallies (when min_hits > 0)
  - build_rally_status_per_frame merges short inactive gaps
  - Rally end timestamp is pinned to last_motion_timestamp, not the silence frame
"""

from __future__ import annotations

import sys
import os

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from types import SimpleNamespace
from core.game_state import GameState, build_rally_status_per_frame


def _make_cfg(**kwargs):
    defaults = dict(
        rally_inactive_timeout_s=1.0,
        rally_min_displacement_px=2.0,
        rally_motion_required_streak=3,
        rally_min_duration_s=0.5,
        rally_detection_grace_frames=3,
        rally_min_hits=1,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _make_shuttle(ts, x, y, w=16, h=16):
    """Make a shuttle detection tuple (ts, x, y, w, h)."""
    return (float(ts), float(x), float(y), float(w), float(h))


FPS = 30.0
FRAME_DT = 1.0 / FPS
FRAME_SIZE = (480, 640)


class TestGameStateRally:

    def test_rally_starts_after_streak(self):
        """Rally should NOT start until motion_required_streak frames of motion.

        With motion_required_streak=3, the rally should be inactive on frames 0
        and 1, and active only once frame 2 is processed.
        """
        cfg = _make_cfg(
            rally_motion_required_streak=3,
            rally_min_duration_s=0.0,   # disable duration gate for this test
            rally_min_hits=0,           # disable hit gate for this test
        )
        gs = GameState(cfg=cfg)

        x = 100.0
        # Frame 0: first detection, streak = 1 — should NOT be active yet
        ts0 = 0 * FRAME_DT
        gs.update(ts0, _make_shuttle(ts0, x, 240.0), FRAME_SIZE)
        assert not gs.rally_active, "Rally should NOT start on first frame (streak=1 of 3)"

        # Frame 1: streak = 2 — still not active
        x += 5.0
        ts1 = 1 * FRAME_DT
        gs.update(ts1, _make_shuttle(ts1, x, 240.0), FRAME_SIZE)
        assert not gs.rally_active, "Rally should NOT start on second frame (streak=2 of 3)"

        # Frame 2: streak = 3 — now active
        x += 5.0
        ts2 = 2 * FRAME_DT
        gs.update(ts2, _make_shuttle(ts2, x, 240.0), FRAME_SIZE)
        assert gs.rally_active, "Rally SHOULD start on third frame (streak=3 of 3)"

    def test_streak_backfills_start_timestamp(self):
        """Once streak confirms, start_time should be back-filled to first streak frame."""
        cfg = _make_cfg(
            rally_motion_required_streak=3,
            rally_min_duration_s=0.0,
            rally_min_hits=0,
        )
        gs = GameState(cfg=cfg)

        x = 100.0
        first_ts = 0 * FRAME_DT
        for i in range(3):
            ts = i * FRAME_DT
            gs.update(ts, _make_shuttle(ts, x, 240.0), FRAME_SIZE)
            x += 5.0

        assert gs.current_rally_start_timestamp == pytest.approx(first_ts, abs=1e-6), \
            "current_rally_start_timestamp should be back-filled to the first streak frame"

    def test_rally_starts_with_motion(self):
        """Rally should become active when shuttle shows consistent motion."""
        cfg = _make_cfg(rally_min_duration_s=0.0, rally_min_hits=0)
        gs = GameState(cfg=cfg)

        x = 100.0
        for i in range(10):
            ts = i * FRAME_DT
            det = _make_shuttle(ts, x, 240.0)
            x += 5.0
            gs.update(ts, det, FRAME_SIZE)

        assert gs.rally_active, "Rally should be active after consistent motion"

    def test_rally_ends_after_inactive_timeout(self):
        """Rally should end after inactive_timeout_s with no motion."""
        cfg = _make_cfg(
            rally_inactive_timeout_s=0.5,
            rally_min_duration_s=0.0,
            rally_min_hits=0,
        )
        gs = GameState(cfg=cfg)

        x = 100.0
        for i in range(10):
            ts = i * FRAME_DT
            gs.update(ts, _make_shuttle(ts, x, 240.0), FRAME_SIZE)
            x += 10.0

        # Feed > 0.5s of silence (well beyond grace_frames=3)
        inactive_start = 10 * FRAME_DT
        for i in range(30):
            ts = inactive_start + i * FRAME_DT
            gs.update(ts, None, FRAME_SIZE)

        assert not gs.rally_active, "Rally should end after inactive timeout"
        assert len(gs.get_rally_data()) >= 1, "At least one rally segment should be recorded"

    def test_end_rally_resets_state(self):
        """end_rally() must clear last_center, motion_streak, and related state."""
        cfg = _make_cfg(
            rally_inactive_timeout_s=0.5,
            rally_min_duration_s=0.0,
            rally_min_hits=0,
        )
        gs = GameState(cfg=cfg)

        # Build a rally
        x = 100.0
        for i in range(10):
            ts = i * FRAME_DT
            gs.update(ts, _make_shuttle(ts, x, 240.0), FRAME_SIZE)
            x += 10.0

        assert gs.rally_active

        # Feed exactly enough silence to trigger inactive timeout (0.5s = 15 frames at 30fps)
        inactive_start = 10 * FRAME_DT
        for i in range(20):
            ts = inactive_start + i * FRAME_DT
            gs.update(ts, None, FRAME_SIZE)
            if not gs.rally_active:
                # Capture state immediately after rally ends — before more None frames
                # increment _miss_streak again.
                assert gs.last_center is None, "last_center should be None after end_rally"
                assert gs.last_motion_timestamp is None, "last_motion_timestamp should be None"
                assert gs.motion_streak == 0, "motion_streak should be 0"
                break

        assert not gs.rally_active, "Rally should have ended within the timeout window"

    def test_grace_period_keeps_rally_alive(self):
        """Up to detection_grace_frames missed frames should not break the rally."""
        cfg = _make_cfg(
            rally_inactive_timeout_s=2.0,   # long timeout so only grace matters
            rally_detection_grace_frames=3,
            rally_min_duration_s=0.0,
            rally_min_hits=0,
        )
        gs = GameState(cfg=cfg)

        x = 100.0
        # Build confirmed rally (3 frames)
        for i in range(3):
            ts = i * FRAME_DT
            gs.update(ts, _make_shuttle(ts, x, 240.0), FRAME_SIZE)
            x += 5.0

        assert gs.rally_active

        # Insert exactly grace_frames missed frames — rally must survive
        base = 3 * FRAME_DT
        last_motion_ts_before = gs.last_motion_timestamp
        for i in range(3):  # exactly grace_frames
            ts = base + i * FRAME_DT
            gs.update(ts, None, FRAME_SIZE)

        assert gs.rally_active, "Rally should survive within the grace window"
        assert gs.last_motion_timestamp == pytest.approx(last_motion_ts_before, abs=1e-6), \
            "last_motion_timestamp should not change during grace window"

    def test_rally_ends_after_grace_plus_timeout(self):
        """Rally should end after grace window + inactive_timeout_s."""
        cfg = _make_cfg(
            rally_inactive_timeout_s=0.3,
            rally_detection_grace_frames=3,
            rally_min_duration_s=0.0,
            rally_min_hits=0,
        )
        gs = GameState(cfg=cfg)

        x = 100.0
        # Build confirmed rally
        for i in range(5):
            ts = i * FRAME_DT
            gs.update(ts, _make_shuttle(ts, x, 240.0), FRAME_SIZE)
            x += 5.0

        # Feed many more silent frames — well past grace + timeout
        base = 5 * FRAME_DT
        for i in range(30):
            ts = base + i * FRAME_DT
            gs.update(ts, None, FRAME_SIZE)

        assert not gs.rally_active

    def test_min_duration_filter_discards_short_rally(self):
        """Rallies shorter than min_duration_s should be silently discarded."""
        cfg = _make_cfg(
            rally_inactive_timeout_s=0.2,
            rally_min_duration_s=0.5,   # 0.5s hard floor
            rally_min_hits=0,
        )
        gs = GameState(cfg=cfg)

        # Build rally lasting only 3 frames (~0.1s at 30fps) — well below 0.5s
        x = 100.0
        for i in range(3):
            ts = i * FRAME_DT
            gs.update(ts, _make_shuttle(ts, x, 240.0), FRAME_SIZE)
            x += 5.0

        # End it with silence
        base = 3 * FRAME_DT
        for i in range(15):
            ts = base + i * FRAME_DT
            gs.update(ts, None, FRAME_SIZE)

        gs.finalize_rally_data(base + 0.5)

        rallies = gs.get_rally_data()
        assert rallies == [], "Sub-threshold rally should be discarded, not recorded"

    def test_min_hits_filter_discards_hitless_rally(self):
        """Rallies with fewer hits than min_hits should be discarded."""
        cfg = _make_cfg(
            rally_inactive_timeout_s=0.5,
            rally_min_duration_s=0.0,   # disable duration gate
            rally_min_hits=2,           # require at least 2 hits
        )
        gs = GameState(cfg=cfg)

        # Build a long rally but don't manually call record_hit() — hit_count stays 0
        # (GameState.update() no longer calls record_hit() internally; hits come from HitDetector)
        x = 100.0
        for i in range(30):
            ts = i * FRAME_DT
            gs.update(ts, _make_shuttle(ts, x, 240.0), FRAME_SIZE)
            x += 5.0

        # End it with silence
        base = 30 * FRAME_DT
        for i in range(20):
            ts = base + i * FRAME_DT
            gs.update(ts, None, FRAME_SIZE)

        gs.finalize_rally_data(base + 0.7)

        rallies = gs.get_rally_data()
        # With a perfect linear trajectory, hit_count stays 0 — should be discarded
        for r in rallies:
            assert False, f"Rally with 0 hits should have been discarded: {r}"

    def test_rally_data_structure(self):
        """Rally segments should be recorded with correct field structure."""
        cfg = _make_cfg(
            rally_inactive_timeout_s=0.3,
            rally_min_duration_s=0.0,
            rally_min_hits=0,
        )
        gs = GameState(cfg=cfg)

        x = 50.0
        for i in range(10):
            ts = i * FRAME_DT
            gs.update(ts, _make_shuttle(ts, x, 240.0), FRAME_SIZE)
            x += 8.0

        base_ts = 10 * FRAME_DT
        for i in range(int(1.0 / FRAME_DT)):
            ts = base_ts + i * FRAME_DT
            gs.update(ts, None, FRAME_SIZE)

        gs.finalize_rally_data(base_ts + 1.0)

        rallies = gs.get_rally_data()
        assert len(rallies) >= 1

        for r in rallies:
            assert "rally_id" in r
            assert "start_time" in r
            assert "end_time" in r
            assert "duration_s" in r
            assert r["duration_s"] >= 0.0

    def test_no_rally_without_detection(self):
        """Rally should not start if shuttle is never detected."""
        cfg = _make_cfg()
        gs = GameState(cfg=cfg)

        for i in range(30):
            ts = i * FRAME_DT
            gs.update(ts, None, FRAME_SIZE)

        assert not gs.rally_active
        assert gs.get_rally_data() == []

    def test_rally_end_time_pinned_to_last_motion(self):
        """Rally end_time should be last_motion_timestamp, not the silence-trigger frame."""
        cfg = _make_cfg(
            rally_inactive_timeout_s=0.5,
            rally_min_duration_s=0.0,
            rally_min_hits=0,
        )
        gs = GameState(cfg=cfg)

        x = 100.0
        last_motion_ts = None
        for i in range(10):
            ts = i * FRAME_DT
            gs.update(ts, _make_shuttle(ts, x, 240.0), FRAME_SIZE)
            x += 10.0
            if gs.rally_active:
                last_motion_ts = gs.last_motion_timestamp

        # Feed silence to trigger timeout
        base = 10 * FRAME_DT
        for i in range(25):
            ts = base + i * FRAME_DT
            gs.update(ts, None, FRAME_SIZE)

        gs.finalize_rally_data(base + 1.0)
        rallies = gs.get_rally_data()
        assert len(rallies) >= 1

        # The recorded end_time should be last_motion_ts, not base + silence
        r = rallies[-1]
        assert r["end_time"] == pytest.approx(last_motion_ts, abs=FRAME_DT + 1e-6), \
            "Rally end_time should be pinned to the last confirmed motion timestamp"


class TestBuildRallyStatus:

    def test_short_gap_merged(self):
        """Short inactive gap < 0.5s should be merged into surrounding rally."""
        fps = 30.0
        total_frames = 90

        rally_data = [
            {"rally_id": 1, "start_time": 0.0, "end_time": 1.0, "duration_s": 1.0},
            {"rally_id": 2, "start_time": 1.3, "end_time": 3.0, "duration_s": 1.7},
        ]
        status, consolidated = build_rally_status_per_frame(rally_data, total_frames, fps)

        assert len(status) == total_frames

        # Gap between 1.0s and 1.3s is 0.3s < 0.5s — should be merged to True
        gap_start = int(1.0 * fps)
        gap_end = int(1.3 * fps)
        for fi in range(gap_start, min(gap_end, total_frames)):
            assert status[fi] is True, f"Frame {fi} in short gap should be merged to True"

    def test_long_gap_preserved(self):
        """A gap >= 0.5s should NOT be merged."""
        fps = 30.0
        total_frames = 120

        rally_data = [
            {"rally_id": 1, "start_time": 0.0, "end_time": 1.0, "duration_s": 1.0},
            {"rally_id": 2, "start_time": 2.0, "end_time": 4.0, "duration_s": 2.0},
        ]
        status, consolidated = build_rally_status_per_frame(rally_data, total_frames, fps)

        # Gap from 1.0s to 2.0s is 1.0s >= 0.5s — should stay False
        mid_gap = int(1.5 * fps)
        assert status[mid_gap] is False, "Long gap should be preserved as False"

    def test_rally_status_length(self):
        """Output list should match total_frames."""
        fps = 30.0
        total_frames = 60
        rally_data = [
            {"rally_id": 1, "start_time": 0.5, "end_time": 1.5, "duration_s": 1.0}
        ]
        status, _ = build_rally_status_per_frame(rally_data, total_frames, fps)
        assert len(status) == total_frames

    def test_empty_rally_data(self):
        """Empty rally_data input should return all-False status."""
        fps = 30.0
        total_frames = 60
        status, consolidated = build_rally_status_per_frame([], total_frames, fps)
        assert len(status) == total_frames
        assert all(s is False for s in status)
        assert consolidated == []

    def test_consolidated_rally_ids_sequential(self):
        """Consolidated rally IDs should be sequential starting from 1."""
        fps = 30.0
        total_frames = 150
        rally_data = [
            {"rally_id": 1, "start_time": 0.0, "end_time": 1.0, "duration_s": 1.0},
            {"rally_id": 2, "start_time": 2.0, "end_time": 3.0, "duration_s": 1.0},
            {"rally_id": 3, "start_time": 4.0, "end_time": 5.0, "duration_s": 1.0},
        ]
        _, consolidated = build_rally_status_per_frame(rally_data, total_frames, fps)
        for expected_id, r in enumerate(consolidated, start=1):
            assert r["rally_id"] == expected_id
