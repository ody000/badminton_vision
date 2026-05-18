"""Tier 1 smoke tests for TrackNetTracker.

Tests:
  - 3 blank (288x512x3) frames → detect() returns dict, no exception
  - Buffer flush: gap > 2x frame interval → buffer cleared before append
  - Works with random weights (no checkpoint needed)
"""

from __future__ import annotations

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from types import SimpleNamespace
from models.shuttle_tracknet import TrackNetTracker


def _make_cfg():
    return SimpleNamespace(
        tracknet_weights="models/tracknet.pt",  # may not exist; that's fine
        device="cpu",
        tracknet_box_size=16,
        tracknet_conf_threshold=0.0,  # accept any detection
        tracknet_expected_h=288,
        tracknet_expected_w=512,
        fps=30.0,
    )


def _blank_frame(h=288, w=512):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_detect_returns_dict_no_exception():
    """3 blank frames should return a dict without raising."""
    cfg = _make_cfg()
    tracker = TrackNetTracker(cfg=cfg)

    for i in range(3):
        frame = _blank_frame()
        result = tracker.detect(frame, timestamp=float(i) / 30.0)
        assert isinstance(result, dict), f"detect() should return a dict, got {type(result)}"


def test_detect_blank_frame_returns_dict():
    """Single blank frame (288x512) returns a dict."""
    cfg = _make_cfg()
    tracker = TrackNetTracker(cfg=cfg)
    frame = _blank_frame()
    result = tracker.detect(frame, timestamp=0.0)
    assert isinstance(result, dict)


def test_detect_arbitrary_resolution():
    """Frame at non-standard resolution is handled without exception."""
    cfg = _make_cfg()
    tracker = TrackNetTracker(cfg=cfg)
    frame = _blank_frame(h=720, w=1280)
    result = tracker.detect(frame, timestamp=0.0)
    assert isinstance(result, dict)


def test_buffer_flush_on_large_gap():
    """Gap > 2*(1/fps) should flush the buffer before the new frame is appended."""
    cfg = _make_cfg()
    tracker = TrackNetTracker(cfg=cfg)

    fps = 30.0
    tracker.set_fps(fps)

    # Feed 2 frames normally
    tracker.detect(_blank_frame(), timestamp=0.0)
    tracker.detect(_blank_frame(), timestamp=1.0 / fps)
    assert len(tracker._buffer) == 2

    # Feed a frame with a large gap (> 2 / fps)
    large_gap_ts = 1.0 / fps + (3.0 / fps)  # gap = 3 frames
    tracker.detect(_blank_frame(), timestamp=large_gap_ts)

    # Buffer should have been flushed and then the new frame appended → size = 1
    assert len(tracker._buffer) == 1, (
        f"Buffer should be 1 after flush, got {len(tracker._buffer)}"
    )


def test_set_fps():
    """set_fps updates the fps used for gap detection."""
    cfg = _make_cfg()
    tracker = TrackNetTracker(cfg=cfg)
    tracker.set_fps(60.0)
    assert tracker.fps == 60.0


def test_detect_output_keys_when_detected():
    """If shuttle is detected, result should have 'shuttle' key with 5-tuple."""
    cfg = _make_cfg()
    tracker = TrackNetTracker(cfg=cfg)

    result = tracker.detect(_blank_frame(), timestamp=0.0)
    if "shuttle" in result:
        shuttle = result["shuttle"]
        assert len(shuttle) == 5, f"shuttle tuple should have 5 elements: (ts, x, y, w, h)"
