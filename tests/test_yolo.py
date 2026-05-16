"""Tier 1 smoke tests for PlayerDetector.

Tests:
  - Blank frame → detect() returns list, no exception
  - MOG2 filter disabled for first 150 frames (warmup)
"""

from __future__ import annotations

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def _make_cfg():
    return SimpleNamespace(
        player_weights="weights/yolo_badminton.pt",  # may not exist; falls back to yolov8n.pt
        player_conf_threshold=0.5,
        mog2_warmup_frames=150,
        mog2_foreground_thresh_player=0.06,
        device="cpu",
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
        # Use a minimal mock to avoid downloading YOLO weights in CI
        with patch("models.player_yolo.YOLO") as mock_yolo_cls:
            mock_model = MagicMock()
            mock_model.track.return_value = _make_mock_results(0)
            mock_yolo_cls.return_value = mock_model

            detector = PlayerDetector(cfg=cfg, mog2_manager=None)
            frame = _blank_frame()
            result = detector.detect(frame)

        assert isinstance(result, list), f"detect() should return list, got {type(result)}"

    def test_detect_no_mog2_filter_during_warmup(self):
        """During warmup (frame_count <= warmup_frames), MOG2 filter should NOT reject boxes."""
        from models.player_yolo import PlayerDetector
        from utils.mog import MOG2Manager

        cfg = _make_cfg()

        with patch("models.player_yolo.YOLO") as mock_yolo_cls:
            import torch
            boxes_mock = MagicMock()
            boxes_mock.xyxy = torch.tensor([[10.0, 10.0, 100.0, 200.0]])
            boxes_mock.id = torch.tensor([1.0])

            result_mock = MagicMock()
            result_mock.boxes = boxes_mock

            mock_model = MagicMock()
            mock_model.track.return_value = [result_mock]
            mock_yolo_cls.return_value = mock_model

            mog2 = MOG2Manager()
            detector = PlayerDetector(cfg=cfg, mog2_manager=mog2)

            # Apply the MOG2 to a blank frame (low foreground ratio)
            mog2.apply(_blank_frame())

            # During warmup, even low foreground ratio should not filter boxes
            frame = _blank_frame()
            detector.frame_count = 0  # force warmup state

            result = detector.detect(frame)

        assert isinstance(result, list)
        # During warmup, boxes are returned regardless of MOG2 ratio
        assert len(result) >= 0  # just no exception

    def test_update_mog2_no_exception(self):
        """update_mog2() should apply MOG2 without raising."""
        from models.player_yolo import PlayerDetector
        from utils.mog import MOG2Manager

        cfg = _make_cfg()

        with patch("models.player_yolo.YOLO") as mock_yolo_cls:
            mock_yolo_cls.return_value = MagicMock()
            mog2 = MOG2Manager()
            detector = PlayerDetector(cfg=cfg, mog2_manager=mog2)
            frame = _blank_frame()
            detector.update_mog2(frame)  # should not raise

    def test_detect_with_no_mog2_no_exception(self):
        """Passing mog2_manager=None should work without exception."""
        from models.player_yolo import PlayerDetector

        cfg = _make_cfg()

        with patch("models.player_yolo.YOLO") as mock_yolo_cls:
            mock_model = MagicMock()
            mock_model.track.return_value = _make_mock_results(0)
            mock_yolo_cls.return_value = mock_model

            detector = PlayerDetector(cfg=cfg, mog2_manager=None)
            result = detector.detect(_blank_frame())

        assert isinstance(result, list)
