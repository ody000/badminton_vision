"""Tier 1 smoke tests for HitDetector.

Tests:
  1. Smooth parabola → 0 hits
  2. Parabola with 1 discontinuity at midpoint → exactly 1 hit
  3. Hit attributed to nearest synthetic player
  4. Cooldown prevents double-firing

No GPU needed.
"""

from __future__ import annotations

import math
import sys
import os

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from types import SimpleNamespace
from core.hit_detector import HitDetector


def _make_cfg(**kwargs):
    defaults = dict(
        hit_trajectory_n=6,
        hit_cooldown_s=0.2,
        hit_proximity_cm=500.0,  # generous for tests
        hit_prediction_error_threshold=20.0,
        hit_ransac_iterations=20,
        hit_ransac_min_inliers=3,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _parabola_points(n=12, a=0.5, x0=100.0, y0=50.0, dx=10.0):
    """Generate n points on a smooth parabola y = a*(x - x0)^2 + y0."""
    pts = []
    for i in range(n):
        x = x0 + i * dx
        y = a * (i - n / 2) ** 2 + y0
        pts.append((float(i) / 30.0, x, y))
    return pts


def _players_at(positions):
    """Build synthetic player_feet_real list from [(x_cm, y_cm), ...]."""
    return [
        {"id": i, "feet_real": (float(px), float(py))}
        for i, (px, py) in enumerate(positions)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Smooth parabola → 0 hits
# ─────────────────────────────────────────────────────────────────────────────

def test_smooth_parabola_no_hits():
    cfg = _make_cfg(hit_prediction_error_threshold=20.0)
    det = HitDetector(cfg)
    pts = _parabola_points(n=12)
    players = _players_at([(pts[0][1], pts[0][2])])

    hits = 0
    for t, x, y in pts:
        is_hit, pid = det.update(t, (x, y), players)
        if is_hit:
            hits += 1

    assert hits == 0, f"Expected 0 hits on smooth parabola, got {hits}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Parabola with 1 discontinuity → exactly 1 hit
# ─────────────────────────────────────────────────────────────────────────────

def test_parabola_with_discontinuity_one_hit():
    cfg = _make_cfg(
        hit_prediction_error_threshold=20.0,
        hit_cooldown_s=0.5,
        hit_proximity_cm=10000.0,
    )
    det = HitDetector(cfg)
    pts = _parabola_points(n=12)

    # Introduce large discontinuity at midpoint
    mid = len(pts) // 2
    modified = list(pts)
    t_mid, x_mid, y_mid = modified[mid]
    modified[mid] = (t_mid, x_mid + 300.0, y_mid + 300.0)  # large jump

    players = _players_at([(modified[mid][1], modified[mid][2])])

    hits = 0
    for t, x, y in modified:
        is_hit, pid = det.update(t, (x, y), players)
        if is_hit:
            hits += 1

    assert hits == 1, f"Expected exactly 1 hit, got {hits}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Hit attributed to nearest player
# ─────────────────────────────────────────────────────────────────────────────

def test_hit_attributed_to_nearest_player():
    cfg = _make_cfg(
        hit_prediction_error_threshold=20.0,
        hit_cooldown_s=0.5,
        hit_proximity_cm=10000.0,
    )
    det = HitDetector(cfg)
    pts = _parabola_points(n=12)

    mid = len(pts) // 2
    modified = list(pts)
    t_mid, x_mid, y_mid = modified[mid]
    jump_x, jump_y = x_mid + 300.0, y_mid + 300.0
    modified[mid] = (t_mid, jump_x, jump_y)

    # Two players: one near the jump location, one far away
    players_near = [
        {"id": 7, "feet_real": (jump_x + 5.0, jump_y + 5.0)},   # near
        {"id": 99, "feet_real": (jump_x + 2000.0, jump_y + 2000.0)},  # far
    ]

    hit_player = None
    for t, x, y in modified:
        is_hit, pid = det.update(t, (x, y), players_near)
        if is_hit:
            hit_player = pid

    assert hit_player == 7, f"Expected hit attributed to player 7 (nearest), got {hit_player}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Cooldown prevents double-firing
# ─────────────────────────────────────────────────────────────────────────────

def test_cooldown_prevents_double_hit():
    cfg = _make_cfg(
        hit_prediction_error_threshold=15.0,
        hit_cooldown_s=1.0,   # 1 second cooldown
        hit_proximity_cm=10000.0,
    )
    det = HitDetector(cfg)
    pts = _parabola_points(n=20)

    # Two discontinuities very close in time (at indices 5 and 7)
    modified = list(pts)
    for jump_idx in [6, 8]:
        t_j, x_j, y_j = modified[jump_idx]
        modified[jump_idx] = (t_j, x_j + 300.0, y_j + 300.0)

    players = [{"id": 1, "feet_real": (200.0, 200.0)}]

    hits = 0
    for t, x, y in modified:
        is_hit, _ = det.update(t, (x, y), players)
        if is_hit:
            hits += 1

    # With 1-second cooldown, the two discontinuities are ~0.067s apart (2/30fps)
    # So at most 1 hit should fire
    assert hits <= 1, f"Expected at most 1 hit due to cooldown, got {hits}"
