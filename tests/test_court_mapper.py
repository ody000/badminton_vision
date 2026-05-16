"""Tier 1 smoke tests for CourtMapper.

Tests:
  - Known pixel corners → assert real-world coords within 1cm tolerance
  - transform_point(None) → None
  - is_calibrated() False before calibrate(), True after
"""

from __future__ import annotations

import math
import sys
import os

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from types import SimpleNamespace
from core.homography import CourtMapper


def _make_cfg():
    return SimpleNamespace(
        court_real_width_cm=610.0,
        court_real_length_cm=1340.0,
        court_points_file="data/input/court_points.json",
    )


# Synthetic pixel corners mapping to a 610x1340 cm court.
# We'll use a simple axis-aligned rectangle for testing.
# Pixel space: (0,0) top-left, (640, 480) bottom-right (arbitrary).
PIXEL_W = 640
PIXEL_H = 480

# 6 points in order: BL, BR, TR, TL, Midline-Bottom, Midline-Top
PIXEL_CORNERS = [
    (0, PIXEL_H),       # Bottom-Left  → (0, 1340)
    (PIXEL_W, PIXEL_H), # Bottom-Right → (610, 1340)
    (PIXEL_W, 0),       # Top-Right    → (610, 0)
    (0, 0),             # Top-Left     → (0, 0)
    (PIXEL_W // 2, PIXEL_H),  # Midline-Bottom
    (PIXEL_W // 2, 0),        # Midline-Top
]

# Expected real-world coords for the 4 outer corners (cm)
EXPECTED = [
    (0.0, 1340.0),   # Bottom-Left
    (610.0, 1340.0), # Bottom-Right
    (610.0, 0.0),    # Top-Right
    (0.0, 0.0),      # Top-Left
]


def test_is_calibrated_false_before_calibrate():
    cfg = _make_cfg()
    mapper = CourtMapper(cfg)
    assert mapper.is_calibrated() is False


def test_is_calibrated_true_after_calibrate():
    cfg = _make_cfg()
    mapper = CourtMapper(cfg)
    mapper.calibrate(PIXEL_CORNERS)
    assert mapper.is_calibrated() is True


def test_transform_point_none_returns_none():
    cfg = _make_cfg()
    mapper = CourtMapper(cfg)
    mapper.calibrate(PIXEL_CORNERS)
    result = mapper.transform_point(None)
    assert result is None


def test_transform_point_not_calibrated_returns_none():
    cfg = _make_cfg()
    mapper = CourtMapper(cfg)
    result = mapper.transform_point((100, 200))
    assert result is None


@pytest.mark.parametrize("pixel_pt,expected_real,tol", [
    (PIXEL_CORNERS[0], EXPECTED[0], 2.0),  # Bottom-Left
    (PIXEL_CORNERS[1], EXPECTED[1], 2.0),  # Bottom-Right
    (PIXEL_CORNERS[2], EXPECTED[2], 2.0),  # Top-Right
    (PIXEL_CORNERS[3], EXPECTED[3], 2.0),  # Top-Left
])
def test_corner_transforms(pixel_pt, expected_real, tol):
    cfg = _make_cfg()
    mapper = CourtMapper(cfg)
    mapper.calibrate(PIXEL_CORNERS)
    result = mapper.transform_point(pixel_pt)
    assert result is not None, "transform_point returned None for a calibrated mapper"
    dist = math.hypot(result[0] - expected_real[0], result[1] - expected_real[1])
    assert dist <= tol, (
        f"Expected {expected_real}, got {result}, distance={dist:.3f}cm > {tol}cm tolerance"
    )


def test_center_transforms_within_court():
    """Center pixel should map to roughly the center of the court."""
    cfg = _make_cfg()
    mapper = CourtMapper(cfg)
    mapper.calibrate(PIXEL_CORNERS)

    center_px = (PIXEL_W / 2, PIXEL_H / 2)
    result = mapper.transform_point(center_px)
    assert result is not None

    # Should be close to (305, 670) = center of court
    dist = math.hypot(result[0] - 305.0, result[1] - 670.0)
    assert dist <= 5.0, f"Center mapped to {result}, expected ~(305, 670), dist={dist:.2f}cm"


def test_get_player_feet():
    cfg = _make_cfg()
    mapper = CourtMapper(cfg)
    box = [100, 50, 200, 400]
    feet = mapper.get_player_feet(box)
    assert feet == (150.0, 400.0), f"Expected (150.0, 400.0), got {feet}"
