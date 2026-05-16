"""Rendering utilities: court insert, heatmaps, bounding boxes, frame overlays.

No game logic. No cv2.imshow or GUI calls.
Ported from slayminton/scripts/visualizations.py.
"""

from __future__ import annotations

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_insert_dims(cfg) -> tuple[int, int, int]:
    """Return (INSERT_H, INSERT_W, COURT_PAD) from cfg."""
    COURT_LEN_M = 13.4
    COURT_WID_M = 6.1
    INSERT_H = int(getattr(cfg, "court_insert_h", 300))
    INSERT_W = int(INSERT_H * COURT_WID_M / COURT_LEN_M)
    COURT_PAD = 14
    return INSERT_H, INSERT_W, COURT_PAD


def _court_bounds(cfg):
    INSERT_H, INSERT_W, COURT_PAD = _get_insert_dims(cfg)
    CX0 = COURT_PAD
    CX1 = INSERT_W - COURT_PAD
    CY0 = COURT_PAD
    CY1 = INSERT_H - COURT_PAD
    CW = CX1 - CX0
    CH = CY1 - CY0
    return INSERT_H, INSERT_W, COURT_PAD, CX0, CX1, CY0, CY1, CW, CH


# ─────────────────────────────────────────────────────────────────────────────
# Court background
# ─────────────────────────────────────────────────────────────────────────────

def draw_court_background(cfg) -> np.ndarray:
    """Return INSERT_H x INSERT_W BGR image with top-down court lines.

    Ported verbatim from slayminton/scripts/visualizations.py draw_court_background().
    """
    COURT_LEN_M = 13.4
    COURT_WID_M = 6.1
    INSERT_H, INSERT_W, COURT_PAD, CX0, CX1, CY0, CY1, CW, CH = _court_bounds(cfg)

    def m_to_px(mx, my):
        px = CX0 + int(mx * CW / COURT_WID_M)
        py = CY0 + int(my * CH / COURT_LEN_M)
        return px, py

    img = np.zeros((INSERT_H, INSERT_W, 3), dtype=np.uint8)
    img[:] = (28, 60, 28)  # dark green background

    WHITE = (220, 220, 220)
    THIN = 1

    def hline(y_m):
        y = m_to_px(0, y_m)[1]
        cv2.line(img, (CX0, y), (CX1, y), WHITE, THIN)

    def vline(x_m, y0_m=0.0, y1_m=COURT_LEN_M):
        x = m_to_px(x_m, 0)[0]
        y0 = m_to_px(0, y0_m)[1]
        y1 = m_to_px(0, y1_m)[1]
        cv2.line(img, (x, y0), (x, y1), WHITE, THIN)

    # Outer doubles boundary
    cv2.rectangle(img, (CX0, CY0), (CX1, CY1), WHITE, THIN)

    # Singles sidelines
    vline(0.46)
    vline(5.64)

    # Net
    net_y = m_to_px(0, 6.7)[1]
    cv2.line(img, (CX0, net_y), (CX1, net_y), WHITE, 2)

    # Short service lines
    hline(4.72)
    hline(8.68)

    # Long service lines (doubles)
    hline(0.76)
    hline(12.64)

    # Centre line
    vline(3.05, 0.0, COURT_LEN_M)

    # Net label
    cv2.putText(
        img,
        "NET",
        (m_to_px(2.6, 6.7)[0], net_y - 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.28,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )

    # Thin outer border
    cv2.rectangle(img, (0, 0), (INSERT_W - 1, INSERT_H - 1), (90, 90, 90), 1)

    return img


# ─────────────────────────────────────────────────────────────────────────────
# Court insert with heatmaps
# ─────────────────────────────────────────────────────────────────────────────

def render_court_insert(
    player_feet_real_history: dict,
    cfg,
) -> np.ndarray:
    """Render court insert with per-player footwork heatmap.

    Args:
        player_feet_real_history: {player_id: [(x_cm, y_cm), ...]}
            P1 = lowest ByteTrack ID, P2 = second lowest.
        cfg: config SimpleNamespace.

    Returns:
        BGR court insert image (INSERT_H x INSERT_W).
    """
    INSERT_H, INSERT_W, COURT_PAD, CX0, CX1, CY0, CY1, CW, CH = _court_bounds(cfg)
    court_real_width_cm  = float(getattr(cfg, "court_real_width_cm", 610.0))
    court_real_length_cm = float(getattr(cfg, "court_real_length_cm", 1340.0))
    # heatmap_gaussian_sigma is the primary knob; player_heatmap_blur is legacy.
    sigma   = int(getattr(cfg, "heatmap_gaussian_sigma", 25))
    stamp_r = int(getattr(cfg, "player_stamp_radius", 6))
    p1_color = tuple(getattr(cfg, "player_p1_color_bgr", [57, 255, 20]))
    p2_color = tuple(getattr(cfg, "player_p2_color_bgr", [0, 165, 255]))

    court_base = draw_court_background(cfg)
    canvas = court_base.copy()

    hm_p1 = np.zeros((INSERT_H, INSERT_W), dtype=np.float32)
    hm_p2 = np.zeros((INSERT_H, INSERT_W), dtype=np.float32)

    # Assign P1/P2 by lowest ByteTrack IDs
    sorted_ids = sorted(player_feet_real_history.keys())
    id_to_slot = {}
    if len(sorted_ids) >= 1:
        id_to_slot[sorted_ids[0]] = "p1"
    if len(sorted_ids) >= 2:
        id_to_slot[sorted_ids[1]] = "p2"

    def real_to_insert(x_cm, y_cm):
        """Map real-world cm coords to court insert pixel coords."""
        ix = CX0 + int(x_cm * CW / court_real_width_cm)
        iy = CY0 + int(y_cm * CH / court_real_length_cm)
        ix = int(np.clip(ix, 0, INSERT_W - 1))
        iy = int(np.clip(iy, 0, INSERT_H - 1))
        return ix, iy

    last_p1 = None
    last_p2 = None

    for pid, history in player_feet_real_history.items():
        slot = id_to_slot.get(pid)
        if slot is None:
            continue
        hm_acc = hm_p1 if slot == "p1" else hm_p2
        for pos in history:
            if pos is None:
                continue
            ix, iy = real_to_insert(float(pos[0]), float(pos[1]))
            cv2.circle(hm_acc, (ix, iy), stamp_r, 1.0, -1)
        if history:
            last_pos = history[-1]
            if last_pos is not None:
                pt = real_to_insert(float(last_pos[0]), float(last_pos[1]))
                if slot == "p1":
                    last_p1 = pt
                else:
                    last_p2 = pt

    # ── Per-player heatmaps ───────────────────────────────────────────────────
    # Each player is normalised independently so neither player's coverage
    # washes out the other.  The Gaussian kernel is derived from sigma so
    # one config knob (heatmap_gaussian_sigma) controls the entire envelope.
    # Each heatmap is colourised in the player's team colour then composited.
    blur_k = max(3, int(6 * sigma + 1))
    if blur_k % 2 == 0:
        blur_k += 1  # kernel must be odd

    def _render_player_heat(hm: np.ndarray, color_bgr: tuple) -> np.ndarray | None:
        if hm.max() <= 0:
            return None
        blurred  = cv2.GaussianBlur(hm, (blur_k, blur_k), sigmaX=sigma, sigmaY=sigma)
        # Normalise to [0, 1] relative to this player's own peak
        hm_norm  = blurred / float(blurred.max())
        # Power < 1 lifts mid-range values, making the gradient more visible
        hm_pow   = np.power(hm_norm, 0.45)
        heat_u8  = np.clip(hm_pow * 255.0, 0, 255).astype(np.uint8)
        # Colourize with INFERNO then tint toward the player's team colour
        heat_inferno = cv2.applyColorMap(heat_u8, cv2.COLORMAP_INFERNO)
        # Blend inferno with a solid team-colour layer for identity clarity
        team_layer = np.zeros_like(heat_inferno)
        team_layer[:] = color_bgr
        return cv2.addWeighted(heat_inferno, 0.65, team_layer, 0.35, 0)

    for hm_layer, color in [(hm_p1, p1_color), (hm_p2, p2_color)]:
        layer_bgr = _render_player_heat(hm_layer, color)
        if layer_bgr is not None:
            # alpha = heatmap intensity mask so blank court shows through
            hm_alpha = cv2.GaussianBlur(hm_layer.copy(), (blur_k, blur_k),
                                        sigmaX=sigma, sigmaY=sigma)
            hm_alpha = np.clip(hm_alpha / max(hm_alpha.max(), 1e-6), 0.0, 1.0)
            hm_alpha_3 = np.stack([hm_alpha] * 3, axis=-1)
            canvas = (canvas * (1.0 - 0.55 * hm_alpha_3)
                      + layer_bgr * (0.55 * hm_alpha_3)).astype(np.uint8)

    # Current player dots
    for pos, color, label in [
        (last_p1, p1_color, "P1"),
        (last_p2, p2_color, "P2"),
    ]:
        if pos is not None:
            cv2.circle(canvas, pos, 7, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.circle(canvas, pos, 5, color, -1, cv2.LINE_AA)
            cv2.putText(
                canvas, label,
                (pos[0] + 6, pos[1] + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1, cv2.LINE_AA,
            )

    # Legend
    cv2.putText(canvas, "Footwork", (4, 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(canvas, "P1", (INSERT_W - 22, 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, p1_color, 1, cv2.LINE_AA)
    cv2.putText(canvas, "P2", (INSERT_W - 10, 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, p2_color, 1, cv2.LINE_AA)

    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# Full frame rendering
# ─────────────────────────────────────────────────────────────────────────────

def render_frame(
    frame: np.ndarray,
    detections: list[dict],
    rally_active: bool,
    court_insert: np.ndarray | None,
    hit_events: list[dict],
    cfg,
) -> np.ndarray:
    """Annotate a video frame with detections, rally status, and court insert.

    Args:
        frame: BGR frame (will be copied before modification).
        detections: List of player dicts {"id", "box", "feet", "feet_real"}.
        rally_active: Whether a rally is currently active.
        court_insert: Pre-rendered court insert or None.
        hit_events: List of hit event dicts (last one checked for stroke type).
        cfg: Config SimpleNamespace.

    Returns:
        Annotated BGR frame.
    """
    canvas = frame.copy()
    h, w = canvas.shape[:2]

    p1_color = tuple(getattr(cfg, "player_p1_color_bgr", [57, 255, 20]))
    p2_color = tuple(getattr(cfg, "player_p2_color_bgr", [0, 165, 255]))
    insert_alpha = float(getattr(cfg, "court_insert_alpha", 0.82))

    # Assign P1/P2 colors by ByteTrack ID order
    seen_ids = sorted(set(d["id"] for d in detections))
    id_to_label: dict[int, tuple] = {}
    if len(seen_ids) >= 1:
        id_to_label[seen_ids[0]] = ("P1", p1_color)
    if len(seen_ids) >= 2:
        id_to_label[seen_ids[1]] = ("P2", p2_color)

    # Draw bounding boxes
    for det in detections:
        box = det["box"]
        pid = det["id"]
        label, color = id_to_label.get(pid, (f"P{pid}", (200, 200, 200)))
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            canvas, label,
            (x1, max(y1 - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
        )
        # Feet dot
        fx, fy = int(det["feet"][0]), int(det["feet"][1])
        cv2.circle(canvas, (fx, fy), 4, color, -1, cv2.LINE_AA)

    # Rally status text (top-left)
    rally_label = "Rally: ACTIVE" if rally_active else "Rally: INACTIVE"
    rally_color = (0, 255, 0) if rally_active else (0, 0, 255)
    cv2.putText(
        canvas, rally_label,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75, rally_color, 2, cv2.LINE_AA,
    )

    # Last hit stroke type (show if within 1.5s)
    if hit_events:
        import time
        last_event = hit_events[-1]
        stroke_type = last_event.get("stroke_type")
        event_ts = last_event.get("timestamp", 0.0)
        # We don't have realtime clock; show the last stroke_type always if not None
        if stroke_type is not None:
            stroke_label = f"Stroke: {stroke_type}"
            cv2.putText(
                canvas, stroke_label,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2, cv2.LINE_AA,
            )

    # Overlay court insert (bottom-right corner)
    if court_insert is not None:
        ih, iw = court_insert.shape[:2]
        if ih <= h and iw <= w:
            roi = canvas[h - ih: h, w - iw: w]
            cv2.addWeighted(court_insert, insert_alpha, roi, 1 - insert_alpha, 0, roi)

    return canvas
