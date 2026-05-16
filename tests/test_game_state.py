"""Tier 1 smoke tests for GameState.

Tests:
  - Rally starts within motion_required_streak frames of motion
  - Rally ends after inactive_timeout_s with no motion
  - Short gap < 0.5s merged into rally (build_rally_status_per_frame)
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
        rally_max_displacement_fraction=0.1667,
        rally_stable_frame_threshold=5,
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

    def test_rally_starts_with_motion(self):
        """Rally should become active when shuttle shows consistent motion."""
        cfg = _make_cfg()
        gs = GameState(cfg=cfg)

        # Feed frames with moving shuttle at 30fps
        x = 100.0
        for i in range(10):
            ts = i * FRAME_DT
            det = _make_shuttle(ts, x, 240.0)
            x += 5.0  # clear motion
            gs.update(ts, det, FRAME_SIZE)

        assert gs.rally_active, "Rally should be active after consistent motion"

    def test_rally_ends_after_inactive_timeout(self):
        """Rally should end after inactive_timeout_s with no motion."""
        cfg = _make_cfg(rally_inactive_timeout_s=0.5)
        gs = GameState(cfg=cfg)

        # Start rally with motion
        x = 100.0
        for i in range(5):
            ts = i * FRAME_DT
            det = _make_shuttle(ts, x, 240.0)
            x += 10.0
            gs.update(ts, det, FRAME_SIZE)

        # Now provide no detections for > 0.5s
        inactive_start = 5 * FRAME_DT
        for i in range(20):
            ts = inactive_start + i * FRAME_DT
            gs.update(ts, None, FRAME_SIZE)

        assert not gs.rally_active, "Rally should end after inactive timeout"
        assert len(gs.get_rally_data()) >= 1, "At least one rally segment should be recorded"

    def test_rally_data_recorded(self):
        """Rally segments should be recorded with correct structure."""
        cfg = _make_cfg(rally_inactive_timeout_s=0.3)
        gs = GameState(cfg=cfg)

        # Rally 1: 10 frames of motion
        x = 50.0
        for i in range(10):
            ts = i * FRAME_DT
            gs.update(ts, _make_shuttle(ts, x, 240.0), FRAME_SIZE)
            x += 8.0

        # Gap of 1 second (> inactive_timeout)
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


class TestBuildRallyStatus:

    def test_short_gap_merged(self):
        """Short inactive gap < 0.5s should be merged into surrounding rally."""
        fps = 30.0
        total_frames = 90  # 3 seconds

        # Rally from frame 0 to 30 (1s), gap 31-39 (0.3s), rally 40-89
        rally_data = [
            {"rally_id": 1, "start_time": 0.0, "end_time": 1.0, "duration_s": 1.0},
            {"rally_id": 2, "start_time": 1.3, "end_time": 3.0, "duration_s": 1.7},
        ]
        status, consolidated = build_rally_status_per_frame(rally_data, total_frames, fps)

        assert len(status) == total_frames

        # The gap between frame 30 and 39 is ~0.3s < 0.5s → should be merged into True
        gap_start = int(1.0 * fps)
        gap_end = int(1.3 * fps)
        for fi in range(gap_start, min(gap_end, total_frames)):
            assert status[fi] is True, f"Frame {fi} in short gap should be merged to True"

    def test_rally_status_length(self):
        """Output list should match total_frames."""
        fps = 30.0
        total_frames = 60
        rally_data = [
            {"rally_id": 1, "start_time": 0.5, "end_time": 1.5, "duration_s": 1.0}
        ]
        status, _ = build_rally_status_per_frame(rally_data, total_frames, fps)
        assert len(status) == total_frames
