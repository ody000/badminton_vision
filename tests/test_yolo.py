"""Tier 1 smoke tests for PlayerDetector.

Tests:
  - Blank frame → detect() returns list, no exception
  - Interval gating: YOLO fires on frame 1, is skipped on frames 2..N, fires again on N+1
  - Cached detections are returned verbatim on skipped frames
  - detect_interval=1 disables gating (YOLO fires every frame)
  - device is passed explicitly to model.track()
"""

from __future__ import annotations

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch


def _make_cfg(detect_interval: int = 3):
    return SimpleNamespace(
        player_weights="models/yolo.pt",  # may not exist; falls back to yolov8n.pt
        player_conf_threshold=0.5,
        device="cpu",
        player_detect_interval=detect_interval,
    )


def _blank_frame(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_mock_results(n_detections=0):
    """Build a mock Ultralytics results object."""
    import torch

    boxes_mock = MagicMock()
    if n_detections > 0:
        boxes_mock.xyxy = torch.zeros(n_detections, 4)
        boxes_mock.id = torch.arange(n_detections, dtype=torch.float32)
    else:
        boxes_mock.xyxy = torch.zeros(0, 4)
        boxes_mock.id = None

    result_mock = MagicMock()
    result_mock.boxes = boxes_mock

    return [result_mock]


class TestPlayerDetector:
    """Smoke tests for PlayerDetector."""

    def test_detect_returns_list_no_exception(self):
        """Blank frame → detect() returns list without raising."""
        from models.player_yolo import PlayerDetector

        cfg = _make_cfg()
        with patch("models.player_yolo.YOLO") as mock_yolo_cls:
            mock_model = MagicMock()
            mock_model.track.return_value = _make_mock_results(0)
            mock_yolo_cls.return_value = mock_model

            detector = PlayerDetector(cfg=cfg)
            result = detector.detect(_blank_frame())

        assert isinstance(result, list), f"detect() should return list, got {type(result)}"

    def test_interval_gating_fires_on_first_frame(self):
        """YOLO must fire on the very first call regardless of interval."""
        from models.player_yolo import PlayerDetector

        cfg = _make_cfg(detect_interval=3)
        with patch("models.player_yolo.YOLO") as mock_yolo_cls:
            mock_model = MagicMock()
            mock_model.track.return_value = _make_mock_results(0)
            mock_yolo_cls.return_value = mock_model

            detector = PlayerDetector(cfg=cfg)
            detector.detect(_blank_frame())  # frame 1

        assert mock_model.track.call_count == 1, (
            "YOLO should fire on the first detect() call"
        )

    def test_interval_gating_skips_intermediate_frames(self):
        """With interval=3, YOLO fires on frames 1 and 4 but is skipped on 2 and 3."""
        from models.player_yolo import PlayerDetector

        cfg = _make_cfg(detect_interval=3)
        with patch("models.player_yolo.YOLO") as mock_yolo_cls:
            mock_model = MagicMock()
            mock_model.track.return_value = _make_mock_results(0)
            mock_yolo_cls.return_value = mock_model

            detector = PlayerDetector(cfg=cfg)
            for _ in range(4):
                detector.detect(_blank_frame())

        # Should have fired on frame 1 and frame 4 = 2 calls total
        assert mock_model.track.call_count == 2, (
            f"With interval=3 and 4 frames, expected 2 YOLO calls, "
            f"got {mock_model.track.call_count}"
        )

    def test_cached_result_returned_on_skipped_frames(self):
        """Skipped frames return the same detection list as the previous YOLO call."""
        from models.player_yolo import PlayerDetector

        cfg = _make_cfg(detect_interval=3)
        with patch("models.player_yolo.YOLO") as mock_yolo_cls:
            mock_model = MagicMock()
            # First call returns 2 detections; subsequent calls return 0.
            mock_model.track.side_effect = [
                _make_mock_results(2),
                _make_mock_results(0),
            ]
            mock_yolo_cls.return_value = mock_model

            detector = PlayerDetector(cfg=cfg)
            result_frame1 = detector.detect(_blank_frame())  # YOLO fires → 2 detections
            result_frame2 = detector.detect(_blank_frame())  # cached
            result_frame3 = detector.detect(_blank_frame())  # cached

        # Frames 2 and 3 should return the cached result from frame 1
        assert result_frame2 is result_frame1, (
            "Frame 2 (skipped) should return the cached list object from frame 1"
        )
        assert result_frame3 is result_frame1, (
            "Frame 3 (skipped) should return the cached list object from frame 1"
        )

    def test_interval_one_disables_gating(self):
        """detect_interval=1 means YOLO fires on every frame."""
        from models.player_yolo import PlayerDetector

        cfg = _make_cfg(detect_interval=1)
        with patch("models.player_yolo.YOLO") as mock_yolo_cls:
            mock_model = MagicMock()
            mock_model.track.return_value = _make_mock_results(0)
            mock_yolo_cls.return_value = mock_model

            detector = PlayerDetector(cfg=cfg)
            for _ in range(5):
                detector.detect(_blank_frame())

        assert mock_model.track.call_count == 5, (
            "With interval=1, YOLO should fire on every frame"
        )

    def test_device_passed_explicitly_to_track(self):
        """model.track() must receive device= explicitly to prevent CPU fallback."""
        from models.player_yolo import PlayerDetector

        cfg = _make_cfg(detect_interval=1)
        cfg.device = "cuda"

        with patch("models.player_yolo.YOLO") as mock_yolo_cls:
            mock_model = MagicMock()
            mock_model.track.return_value = _make_mock_results(0)
            mock_yolo_cls.return_value = mock_model

            detector = PlayerDetector(cfg=cfg)
            detector.detect(_blank_frame())

        _, kwargs = mock_model.track.call_args
        assert "device" in kwargs, "device= must be passed to model.track()"
        assert kwargs["device"] == "cuda", (
            f"Expected device='cuda', got {kwargs.get('device')}"
        )
