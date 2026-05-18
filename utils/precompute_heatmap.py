"""Precompute a static player-footwork heatmap from tracking_results.json.

Produces a white-to-JET heatmap for each player (P1 tinted green, P2 tinted
orange), blended at 50 % opacity onto the court background.  The result is a
static PNG saved to the run directory; the viewer loads it once at startup and
only redraws the live player-dot overlay per-frame.

Court-insert geometry intentionally mirrors slayminton/scripts/visualizations.py
so that the viewer's existing homography and court-insert code stay consistent.

Public API
----------
precompute_heatmap(tracking_results, court_points, frame_w, frame_h,
                   output_path, ...) -> np.ndarray
"""

from __future__ import annotations

import math
import os
from typing import Optional

import cv2
import numpy as np

# ── Court-insert dimensions (match slayminton/scripts/visualizations.py) ──────
COURT_LEN_M = 13.4
COURT_WID_M = 6.1
INSERT_H = 300
INSERT_W = int(INSERT_H * COURT_WID_M / COURT_LEN_M)   # ≈ 137 px
COURT_PAD = 14

_CX0 = COURT_PAD
_CX1 = INSERT_W - COURT_PAD
_CY0 = COURT_PAD
_CY1 = INSERT_H - COURT_PAD
_CMX = (_CX0 + _CX1) / 2.0
_CW  = _CX1 - _CX0
_CH  = _CY1 - _CY0


# ── Geometry helpers ───────────────────────────────────────────────────────────

def compute_homography(court_points: list, frame_h: int, frame_w: int) -> Optional[np.ndarray]:
    """Compute homography: video pixel coords → court-insert pixel coords.

    Args:
        court_points: 6 points [BL, BR, TR, TL, midline-bottom, midline-top]
                      in video pixel coordinates.
        frame_h, frame_w: source frame dimensions (unused but kept for API parity
                          with slayminton version).

    Returns:
        3×3 float32 homography matrix, or None if computation fails.
    """
    if court_points is None or len(court_points) < 6:
        return None
    src = np.array(court_points[:6], dtype=np.float32)
    dst = np.array([
        [_CX0, _CY0],   # BL  → top-left of insert
        [_CX1, _CY0],   # BR  → top-right of insert
        [_CX1, _CY1],   # TR  → bottom-right of insert
        [_CX0, _CY1],   # TL  → bottom-left of insert
        [_CMX, _CY0],   # midline-bottom
        [_CMX, _CY1],   # midline-top
    ], dtype=np.float32)
    H, _ = cv2.findHomography(src, dst)
    return H


def video_to_insert(cx: float, cy: float, H: np.ndarray) -> tuple[int, int]:
    """Map a video-pixel position to court-insert pixel coords.

    The y-axis is flipped so that the near-side of the court (large y in video)
    appears at the *bottom* of the insert image (conventional top-down view).
    """
    pt = np.array([[[float(cx), float(cy)]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(pt, H)[0][0]
    ix = int(np.clip(mapped[0], 0, INSERT_W - 1))
    iy = int(np.clip(INSERT_H - 1 - mapped[1], 0, INSERT_H - 1))
    return ix, iy


# ── Court background ───────────────────────────────────────────────────────────

def draw_court_background() -> np.ndarray:
    """Return a fresh INSERT_H × INSERT_W BGR image with a top-down court."""
    img = np.zeros((INSERT_H, INSERT_W, 3), dtype=np.uint8)
    img[:] = (28, 60, 28)   # dark green

    WHITE = (220, 220, 220)
    THIN  = 1

    def hline(y_m: float) -> None:
        y = _CY0 + int(y_m * _CH / COURT_LEN_M)
        cv2.line(img, (_CX0, y), (_CX1, y), WHITE, THIN)

    def vline(x_m: float, y0_m: float = 0.0, y1_m: float = COURT_LEN_M) -> None:
        x  = _CX0 + int(x_m * _CW / COURT_WID_M)
        y0 = _CY0 + int(y0_m * _CH / COURT_LEN_M)
        y1 = _CY0 + int(y1_m * _CH / COURT_LEN_M)
        cv2.line(img, (x, y0), (x, y1), WHITE, THIN)

    cv2.rectangle(img, (_CX0, _CY0), (_CX1, _CY1), WHITE, THIN)
    vline(0.46)
    vline(5.64)
    net_y = _CY0 + int(6.7 * _CH / COURT_LEN_M)
    cv2.line(img, (_CX0, net_y), (_CX1, net_y), WHITE, 2)
    hline(4.72)
    hline(8.68)
    hline(0.76)
    hline(12.64)
    vline(3.05, 0.0, COURT_LEN_M)
    cv2.putText(img, "NET",
                (_CX0 + int(2.6 * _CW / COURT_WID_M), net_y - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (INSERT_W - 1, INSERT_H - 1), (90, 90, 90), 1)
    return img


# ── Heatmap builder ────────────────────────────────────────────────────────────

def _make_tinted_jet(
    accumulator: np.ndarray,
    sigma: float,
    player_bgr: tuple[int, int, int],
    stamp_radius: int,
) -> Optional[np.ndarray]:
    """Convert a float32 accumulator to a JET-tinted BGR heatmap.

    Returns None if the accumulator is all zeros (no data).
    """
    if accumulator.max() <= 0:
        return None

    # Gaussian blur — kernel size derived from sigma (ceil(6σ)|1)
    ksize = max(3, 2 * int(math.ceil(3.0 * sigma)) + 1)
    if ksize % 2 == 0:
        ksize += 1
    blurred = cv2.GaussianBlur(accumulator, (ksize, ksize), sigma)

    # Normalise to [0, 255]
    norm = cv2.normalize(blurred, None, 0.0, 255.0, cv2.NORM_MINMAX)
    heat_u8 = np.clip(norm, 0, 255).astype(np.uint8)

    # JET colormap
    jet = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)

    # Tint: multiply JET channels by the player colour (normalised to [0,1]).
    # This pushes cold blues toward black and warm reds toward the tint colour.
    tint = np.array(player_bgr, dtype=np.float32) / 255.0
    tinted = np.clip(
        jet.astype(np.float32) * tint[np.newaxis, np.newaxis, :], 0, 255
    ).astype(np.uint8)
    return tinted


def precompute_heatmap(
    tracking_results: list[dict],
    court_points: list,
    frame_w: int,
    frame_h: int,
    output_path: str,
    gaussian_sigma: float = 40.0,
    p1_color_bgr: tuple = (57, 255, 20),    # neon green
    p2_color_bgr: tuple = (0, 165, 255),    # orange
    opacity: float = 0.5,
    stamp_radius: int = 6,
    player_ids: list = None,
) -> np.ndarray:
    """Precompute a static player-footwork heatmap and save it as a PNG.

    Args:
        tracking_results: List of per-frame dicts (from tracking_results.json).
                          Each entry has a "players" list with "id", "feet" fields.
        court_points:     6-point list [BL, BR, TR, TL, mid-B, mid-T] in video pixels.
        frame_w, frame_h: Source video frame dimensions.
        output_path:      Where to write heatmap.png.
        gaussian_sigma:   Gaussian blur sigma (default 40; larger = softer).
        p1_color_bgr:     BGR tint for player 1 (neon green).
        p2_color_bgr:     BGR tint for player 2 (orange).
        opacity:          Heatmap overlay opacity [0, 1] (default 0.5 = 50 %).
        stamp_radius:     Radius in insert-pixels of each position stamp.
        player_ids:       List of player IDs to include (default [1, 2]).

    Returns:
        The saved BGR image (INSERT_H × INSERT_W, uint8).
    """
    if player_ids is None:
        player_ids = [1, 2]

    H = compute_homography(court_points, frame_h, frame_w)

    # Create heatmap accumulators for each player
    heatmaps = {pid: np.zeros((INSERT_H, INSERT_W), dtype=np.float32) for pid in player_ids}
    player_colors = {player_ids[0]: p1_color_bgr}
    if len(player_ids) > 1:
        player_colors[player_ids[1]] = p2_color_bgr

    for frame in tracking_results:
        players = frame.get("players") or []
        for p in players:
            pid = p.get("id")
            # Try feet_px first (video pixel coords), fall back to feet if it exists
            feet = p.get("feet_px") or p.get("feet")
            if feet is None or pid not in player_ids:
                continue
            fx, fy = float(feet[0]), float(feet[1])
            if H is not None:
                ix, iy = video_to_insert(fx, fy, H)
            else:
                # Fallback: linear map
                ix = int(np.clip(_CX0 + fx * _CW / max(frame_w, 1), 0, INSERT_W - 1))
                iy = int(np.clip(_CY0 + fy * _CH / max(frame_h, 1), 0, INSERT_H - 1))
                iy = INSERT_H - 1 - iy   # flip y

            target = heatmaps[pid]
            cv2.circle(target, (ix, iy), stamp_radius, 1.0, -1)

    # Build tinted JET heatmaps
    tinted_heatmaps = []
    for pid in player_ids:
        color = player_colors.get(pid, (57, 255, 20))
        tinted = _make_tinted_jet(heatmaps[pid], gaussian_sigma, color, stamp_radius)
        if tinted is not None:
            tinted_heatmaps.append(tinted)

    # Start from the court background
    canvas = draw_court_background()

    # Blend each player heatmap at the requested opacity
    for tinted in tinted_heatmaps:
        canvas = cv2.addWeighted(canvas, 1.0 - opacity, tinted, opacity, 0)

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, canvas)
    print(f"[HEATMAP] saved {output_path} ({INSERT_W}×{INSERT_H})")

    return canvas
