"""viewer.py — local overlay viewer for badminton_vision runs.

Reads tracking_results.json and events.json from a run directory and
composites overlays on the *original* source video at display time.
No pixels of the source video are modified.  No output files are written.

Usage:
    pip install dearpygui
    python viewer.py --run ../data/output/match_clip_20250515_120000 \\
                     --video ../data/input/match_clip.mp4

Keyboard shortcuts:
    Space         Play / Pause
    Left / Right  Step one frame
    PgUp / PgDn   ±10 frames
    Q / Escape    Quit

Buttons (sidebar):
    Play  -20s / -5s / +5s / +20s   Playback control
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

# Ensure the project root (parent of tools/) is on sys.path so that
# `utils.visualization` is importable regardless of where the script
# is launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

try:
    import dearpygui.dearpygui as dpg
except ImportError:
    print(
        "[VIEWER] DearPyGui is not installed.\n"
        "         pip install dearpygui\n"
        "         then re-run viewer.py"
    )
    sys.exit(1)



# ─────────────────────────────────────────────────────────────────────────────
# Layout constants
# ─────────────────────────────────────────────────────────────────────────────

SIDEBAR_W   = 260      # left control panel width
PANEL_R_W   = 224      # right info panel width
TIMELINE_H  = 64       # timeline strip height
WIN_PAD     = 8        # inner padding for child windows
MAX_VID_W   = 1100     # max display width for the video panel
MIN_VID_W   = 560
MIN_VID_H   = 315

# ─── Colour palette (RGBA 0-255) ─────────────────────────────────────────────
C_BG          = (22,  22,  22, 255)
C_PANEL       = (32,  32,  32, 255)
C_PANEL_DARK  = (18,  18,  18, 255)
C_BORDER      = (60,  60,  60, 255)
C_TEXT        = (215, 215, 215, 255)
C_TEXT_DIM    = (120, 120, 120, 255)
C_ACCENT      = (66,  150, 250, 255)
C_ACCENT_DIM  = (40,  100, 190, 255)
C_GREEN       = (55,  190,  70, 255)
C_RED         = (220,  60,  60, 255)
C_ORANGE      = (240, 140,  30, 255)
C_P1_RGBA     = (20,  255,  57, 255)
C_P2_RGBA     = (0,   165, 255, 255)

# ─── OpenCV BGR overlay colours ───────────────────────────────────────────────
OCV_P1    = (20, 255, 57)
OCV_P2    = (0, 165, 255)
OCV_SHUT  = (80, 220, 255)
OCV_HIT   = (30,  40, 210)     # deep red-blue flash border
OCV_WHITE = (220, 220, 220)


# ─────────────────────────────────────────────────────────────────────────────
# Viewer
# ─────────────────────────────────────────────────────────────────────────────

class Viewer:
    """DearPyGui-powered local overlay viewer."""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, run_dir: str, video_path: str) -> None:
        self.run_dir    = Path(run_dir)
        self.video_path = Path(video_path)

        # ── Load run data ──────────────────────────────────────────────
        self.tracking:    List[dict] = self._load_json("tracking_results.json", [])
        self.events:      List[dict] = self._load_json("events.json",           [])
        self.rally_data:  List[dict] = self._load_json("rally_data.json",       [])
        self.analytics:   dict       = self._load_json("analytics.json",        {})

        # Fast per-frame lookup tables
        self._tracking_by_frame: Dict[int, dict] = {
            t["frame_idx"]: t for t in self.tracking
        }
        self._events_by_frame: Dict[int, List[dict]] = {}
        for ev in self.events:
            fi = ev.get("frame_idx", -1)
            self._events_by_frame.setdefault(fi, []).append(ev)

        # Patch rally count from per-frame rally_active flags when rally_data.json
        # recorded 0 (e.g. because all rallies were discarded as too short/few hits).
        if self.analytics.get("rally_count", 0) == 0 and self.tracking:
            observed = 0
            prev_active = False
            for t in self.tracking:
                active = t.get("rally_active", False)
                if active and not prev_active:
                    observed += 1
                prev_active = active
            if observed > 0:
                self.analytics["rally_count"] = observed
                print(f"[VIEWER] rally_data.json empty; derived {observed} observed rallies "
                      "from per-frame rally_active flags.")

        # ── Open video ────────────────────────────────────────────────
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            print(f"[VIEWER] Cannot open video: {self.video_path}")
            sys.exit(1)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.source_fps   = float(self.cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self.vid_w        = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.vid_h        = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # ── Playback state ────────────────────────────────────────────
        self.frame_idx      = 0
        self.playing        = False
        self.speed          = 1.0
        self._last_tick     = 0.0
        self._tex_w         = 0
        self._tex_h         = 0
        self._tex_tag       = "vid_tex"
        # Sequential-read optimisation: track the last decoded frame index.
        # If the next requested frame is exactly last+1 the cap is already
        # positioned there, so we skip the expensive cap.set() seek.
        # -2 = cap position unknown (force a seek on next read).
        self._last_decoded_idx: int = -2
        # Shuttle trail: keep ring visible for SHUTTLE_TRAIL_FRAMES after
        # the last real detection so dropout gaps don't blank the display.
        self._last_shuttle_j:        Optional[dict] = None
        self._last_shuttle_frame_idx: int           = -9999
        self.SHUTTLE_TRAIL_FRAMES:   int            = 30   # 1 s at 30 fps
        # Pending render request set by callbacks; consumed by the main loop
        # AFTER render_dearpygui_frame() returns.  This decouples video I/O
        # from DearPyGui callback context, preventing macOS segfaults.
        self._needs_render: Optional[int] = None

        # ── Overlay toggles ───────────────────────────────────────────
        self.show_players = True
        self.show_shuttle = True
        self.show_hits    = True
        self.show_heatmap = True
        self.show_timeline = True

        # ── Heatmap cfg (minimal subset needed for rendering) ─────────
        self._heat_cfg = SimpleNamespace(
            court_insert_h=300,
            court_real_width_cm=610.0,
            court_real_length_cm=1340.0,
            player_heatmap_blur=7,
            heatmap_gaussian_sigma=25,
            player_stamp_radius=6,
            player_p1_color_bgr=[57, 255, 20],
            player_p2_color_bgr=[0, 165, 255],
            court_insert_alpha=0.85,
        )

        # Pre-compute cumulative feet history per frame for partial heatmaps.
        # We store it as a list indexed by frame_idx: each entry is a dict
        # {player_id: [(x_cm, y_cm), ...]} containing all positions *up to*
        # that frame.  Built lazily on first heatmap request.
        self._cumulative_history: Optional[List[Dict]] = None

        # ── P1/P2 assignment ──────────────────────────────────────────
        all_ids: List[int] = []
        for t in self.tracking:
            for p in t.get("players", []):
                pid = p["id"]
                if pid not in all_ids:
                    all_ids.append(pid)
        self._p1_id: Optional[int] = all_ids[0] if all_ids else None
        self._p2_id: Optional[int] = all_ids[1] if len(all_ids) >= 2 else None

        # ── Precomputed heatmap ───────────────────────────────────────
        # Static base image (INSERT_H × INSERT_W BGR) loaded/computed once.
        # Per-frame: copy + draw player dots → upload to DPG texture.
        self._heatmap_base:    Optional[np.ndarray] = None
        self._heatmap_H:       Optional[np.ndarray] = None   # homography
        self._heatmap_tex_tag: str  = "heatmap_tex"
        self._heatmap_w:       int  = 0
        self._heatmap_h:       int  = 0
        self._heatmap_loaded:  bool = False
        self._load_or_compute_heatmap()

    # ------------------------------------------------------------------
    # JSON helpers
    # ------------------------------------------------------------------

    def _load_json(self, name: str, default):
        path = self.run_dir / name
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[VIEWER] {name}: {e}")
        return default

    # ------------------------------------------------------------------
    # Precomputed heatmap helpers
    # ------------------------------------------------------------------

    def _load_or_compute_heatmap(self) -> None:
        """Load heatmap.png from run dir; compute it if missing."""
        try:
            from utils.precompute_heatmap import (
                precompute_heatmap, compute_homography, INSERT_W, INSERT_H,
            )
        except Exception as e:
            print(f"[VIEWER] heatmap module unavailable: {e}")
            return

        # Court points for homography
        cp_data = self._load_json("court_points.json", {})
        court_points = next(iter(cp_data.values()), []) if cp_data else []
        if court_points:
            self._heatmap_H = compute_homography(court_points, self.vid_h, self.vid_w)

        hm_path = self.run_dir / "heatmap.png"
        if hm_path.exists():
            img = cv2.imread(str(hm_path))
            if img is not None:
                self._heatmap_base = img
                self._heatmap_w    = img.shape[1]
                self._heatmap_h    = img.shape[0]
                self._heatmap_loaded = True
                print(f"[VIEWER] heatmap loaded ({self._heatmap_w}×{self._heatmap_h})")
                return

        # heatmap.png absent — compute it now (takes a few seconds)
        if self.tracking and court_points:
            print("[VIEWER] heatmap.png not found — computing from tracking data…")
            try:
                # Detect actual player IDs from tracking data
                player_ids = set()
                for t in self.tracking:
                    for p in t.get("players", []):
                        player_ids.add(p.get("id"))

                img = precompute_heatmap(
                    tracking_results=self.tracking,
                    court_points=court_points,
                    frame_w=self.vid_w,
                    frame_h=self.vid_h,
                    output_path=str(hm_path),
                    player_ids=sorted(list(player_ids)) if player_ids else [1, 2],
                )
                self._heatmap_base = img
                self._heatmap_w    = img.shape[1]
                self._heatmap_h    = img.shape[0]
                self._heatmap_loaded = True
            except Exception as e:
                print(f"[VIEWER] heatmap compute failed: {e}")
        else:
            print("[VIEWER] heatmap skipped: no tracking data or court points")

    def _init_heatmap_texture(self) -> None:
        """Register the heatmap DPG dynamic texture (must be inside texture_registry)."""
        if not self._heatmap_loaded or self._heatmap_w == 0:
            return
        blank = [0.0] * (self._heatmap_w * self._heatmap_h * 4)
        if dpg.does_item_exist(self._heatmap_tex_tag):
            dpg.delete_item(self._heatmap_tex_tag)
        dpg.add_dynamic_texture(
            self._heatmap_w, self._heatmap_h, blank, tag=self._heatmap_tex_tag
        )

    def _update_heatmap_texture(self, idx: int) -> None:
        """Copy the static base heatmap, draw live player dots, upload to DPG."""
        if (not self._heatmap_loaded
                or self._heatmap_base is None
                or not dpg.does_item_exist(self._heatmap_tex_tag)):
            return

        try:
            from utils.precompute_heatmap import video_to_insert
        except Exception:
            return

        canvas = self._heatmap_base.copy()

        # Apply slight brightness boost to make heatmap more visible
        canvas = cv2.convertScaleAbs(canvas, alpha=1.15, beta=0)

        t = self._tracking_by_frame.get(idx, {})
        for p in t.get("players", []):
            pid  = p.get("id")
            feet = p.get("feet_px") or p.get("feet")
            if feet is None or self._heatmap_H is None:
                continue
            ix, iy = video_to_insert(float(feet[0]), float(feet[1]), self._heatmap_H)
            color = OCV_P1 if pid == self._p1_id else OCV_P2
            label = "P1"   if pid == self._p1_id else "P2"
            cv2.circle(canvas, (ix, iy), 7, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.circle(canvas, (ix, iy), 5, color, -1, cv2.LINE_AA)
            cv2.putText(canvas, label, (ix + 6, iy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1, cv2.LINE_AA)

        rgba = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGBA)
        flat = np.ascontiguousarray(rgba, dtype=np.float32) / 255.0
        dpg.set_value(self._heatmap_tex_tag, flat)

    # ------------------------------------------------------------------
    # Cumulative heatmap history (built once, lazily)
    # ------------------------------------------------------------------

    def _ensure_cumulative_history(self) -> None:
        if self._cumulative_history is not None:
            return
        acc: Dict[int, List] = {}
        result: List[Dict] = []
        for t in self.tracking:
            fi = t["frame_idx"]
            for p in t.get("players", []):
                pid  = p["id"]
                fr   = p.get("feet_real")
                if fr is not None:
                    acc.setdefault(pid, []).append(tuple(fr))
            result.append({pid: list(pts) for pid, pts in acc.items()})
        # Pad if tracking doesn't cover all frames
        while len(result) < self.total_frames:
            result.append(result[-1] if result else {})
        self._cumulative_history = result

    def _history_at(self, frame_idx: int) -> Dict[int, List]:
        self._ensure_cumulative_history()
        idx = min(frame_idx, len(self._cumulative_history) - 1)
        return self._cumulative_history[idx]

    # ------------------------------------------------------------------
    # Frame rendering  (overlay on top of raw video — no disk writes)
    # ------------------------------------------------------------------

    def _read_frame(self, idx: int) -> Optional[np.ndarray]:
        """Read frame at idx.  Skips cap.set() when reading sequentially
        (idx == last_decoded + 1) to avoid expensive keyframe seeks."""
        if idx != self._last_decoded_idx + 1:
            try:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
            except Exception:
                # cap.set() can raise on macOS / certain codecs; mark dirty
                self._last_decoded_idx = -2
        ok, frame = self.cap.read()
        self._last_decoded_idx = idx if ok else -2
        return frame if ok else None

    def _player_color(self, pid: int) -> Tuple[int, int, int]:
        if pid == self._p1_id:
            return OCV_P1
        if pid == self._p2_id:
            return OCV_P2
        return OCV_WHITE

    def _player_label(self, pid: int) -> str:
        if pid == self._p1_id:
            return "P1"
        if pid == self._p2_id:
            return "P2"
        return f"P{pid}"

    def _render_frame(self, idx: int) -> np.ndarray:
        """Return BGR frame with overlays. Source video is never modified."""
        frame = self._read_frame(idx)
        if frame is None:
            h = self.vid_h or MIN_VID_H
            w = self.vid_w or MIN_VID_W
            return np.zeros((h, w, 3), dtype=np.uint8)

        canvas = frame.copy()
        fh, fw = canvas.shape[:2]

        tracking   = self._tracking_by_frame.get(idx, {})
        players    = tracking.get("players", [])
        shuttle_j  = tracking.get("shuttle")
        rally_act  = tracking.get("rally_active", False)
        hit_events = self._events_by_frame.get(idx, [])

        # ── Player bounding boxes ─────────────────────────────────────
        if self.show_players:
            for p in players:
                pid   = p["id"]
                color = self._player_color(pid)
                label = self._player_label(pid)
                box   = p.get("box", [])
                if len(box) == 4:
                    x1, y1, x2, y2 = (int(v) for v in box)
                    # Faint fill
                    overlay = canvas.copy()
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                    cv2.addWeighted(overlay, 0.08, canvas, 0.92, 0, canvas)
                    # Solid 2-px border
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
                    # Label pill
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
                    pill_x0 = x1
                    pill_y0 = max(y1 - th - 8, 0)
                    pill_x1 = x1 + tw + 8
                    pill_y1 = max(y1, th + 8)
                    cv2.rectangle(canvas, (pill_x0, pill_y0), (pill_x1, pill_y1), color, -1)
                    cv2.putText(
                        canvas, label,
                        (pill_x0 + 4, pill_y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (10, 10, 10), 2, cv2.LINE_AA,
                    )
                # Feet dot
                feet = p.get("feet_px") or p.get("feet")
                if feet:
                    fx, fy = int(feet[0]), int(feet[1])
                    cv2.circle(canvas, (fx, fy), 5, (0, 0, 0), -1, cv2.LINE_AA)
                    cv2.circle(canvas, (fx, fy), 4, color, -1, cv2.LINE_AA)

        # ── Shuttle ring ──────────────────────────────────────────────
        # Update trail state.
        if shuttle_j is not None:
            self._last_shuttle_j         = shuttle_j
            self._last_shuttle_frame_idx = idx

        # Choose which position to draw: live detection or ghost trail.
        _trail_age = idx - self._last_shuttle_frame_idx
        _is_live   = (shuttle_j is not None)
        _is_ghost  = (not _is_live
                      and self._last_shuttle_j is not None
                      and _trail_age <= self.SHUTTLE_TRAIL_FRAMES)
        _draw_src  = shuttle_j if _is_live else (self._last_shuttle_j if _is_ghost else None)

        if self.show_shuttle and _draw_src is not None:
            cx  = int(_draw_src["x"] + _draw_src.get("w", 16) / 2)
            cy  = int(_draw_src["y"] + _draw_src.get("h", 16) / 2)
            ring_r = max(14, int(max(_draw_src.get("w", 16),
                                     _draw_src.get("h", 16)) * 0.85))
            if _is_live:
                # Full-brightness red-orange ring
                cv2.circle(canvas, (cx, cy), ring_r + 2, (0, 0, 0),    3, cv2.LINE_AA)
                cv2.circle(canvas, (cx, cy), ring_r,     (0, 60, 255),  2, cv2.LINE_AA)
                cv2.circle(canvas, (cx, cy), 3,          (0, 60, 255), -1, cv2.LINE_AA)
            else:
                # Ghost: dimmed dashed-style ring (two arcs give a visual cue it's a trail)
                fade = max(0.25, 1.0 - _trail_age / self.SHUTTLE_TRAIL_FRAMES)
                ghost_col = (int(0 * fade), int(60 * fade), int(200 * fade))
                cv2.circle(canvas, (cx, cy), ring_r + 2, (0, 0, 0),  2, cv2.LINE_AA)
                cv2.circle(canvas, (cx, cy), ring_r,     ghost_col,   1, cv2.LINE_AA)

        # ── Hit flash ─────────────────────────────────────────────────
        if self.show_hits and hit_events:
            ev     = hit_events[0]
            stroke = (ev.get("stroke_type") or "HIT").upper()
            pid    = ev.get("player_id")
            pcolor = self._player_color(pid) if pid is not None else OCV_WHITE
            plabel = self._player_label(pid) if pid is not None else "?"
            # Red border flash
            for thickness, col in [(8, (0, 30, 160)), (3, (60, 80, 230))]:
                cv2.rectangle(canvas, (0, 0), (fw-1, fh-1), col, thickness)
            # Centred hit label with drop-shadow
            hit_text = f"  {plabel}  ·  {stroke}  "
            font     = cv2.FONT_HERSHEY_DUPLEX
            (tw, th), _ = cv2.getTextSize(hit_text, font, 0.85, 2)
            tx = (fw - tw) // 2
            ty = 54
            # Shadow
            cv2.putText(canvas, hit_text, (tx + 2, ty + 2), font, 0.85, (0, 0, 0), 3, cv2.LINE_AA)
            # Main text in player colour
            cv2.putText(canvas, hit_text, (tx, ty), font, 0.85, pcolor, 2, cv2.LINE_AA)

        # ── Rally pill (top-left) ─────────────────────────────────────
        pill_txt   = "RALLY" if rally_act else "IDLE"
        pill_col   = (20, 130, 20)  if rally_act else (45, 45, 45)
        pill_tcol  = (200, 240, 200) if rally_act else (100, 100, 100)
        (pw, ph), _ = cv2.getTextSize(pill_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
        cv2.rectangle(canvas, (10, 10), (10 + pw + 10, 10 + ph + 8), pill_col, -1, cv2.LINE_AA)
        cv2.putText(canvas, pill_txt, (15, 10 + ph + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, pill_tcol, 1, cv2.LINE_AA)

        return canvas

    # ------------------------------------------------------------------
    # Texture management
    # ------------------------------------------------------------------

    def _frame_to_rgba_flat(self, frame: np.ndarray) -> np.ndarray:
        """Convert BGR frame to flat float32 RGBA for DearPyGui texture upload.

        Returns a contiguous float32 numpy array (no Python list conversion)
        which DearPyGui accepts directly and is ~10x faster than .tolist().
        """
        rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        return np.ascontiguousarray(rgba, dtype=np.float32) / 255.0

    def _init_texture(self, w: int, h: int) -> None:
        self._tex_w = w
        self._tex_h = h
        blank = [0.0] * (w * h * 4)
        with dpg.texture_registry():
            if dpg.does_item_exist(self._tex_tag):
                dpg.delete_item(self._tex_tag)
            dpg.add_dynamic_texture(w, h, blank, tag=self._tex_tag)

    def _push_frame(self, idx: int) -> None:
        rendered = self._render_frame(idx)
        rh, rw   = rendered.shape[:2]
        if rw != self._tex_w or rh != self._tex_h:
            self._init_texture(rw, rh)
            if dpg.does_item_exist("vid_image"):
                dpg.configure_item("vid_image", width=rw, height=rh)
        dpg.set_value(self._tex_tag, self._frame_to_rgba_flat(rendered))
        self._update_heatmap_texture(idx)
        self._update_info_panel(idx)
        self._update_timeline_playhead(idx)

    # ------------------------------------------------------------------
    # Info panel
    # ------------------------------------------------------------------

    def _update_info_panel(self, idx: int) -> None:
        if not dpg.does_item_exist("inf_time"):
            return
        t           = self._tracking_by_frame.get(idx, {})
        ts          = t.get("timestamp", idx / self.source_fps)
        total_dur   = self.total_frames / self.source_fps
        rally       = t.get("rally_active", False)
        shut        = "detected" if t.get("shuttle") else "—"

        # Find most recent event at or before current frame
        last_stroke = "—"
        for ev in reversed(self.events):
            if ev.get("frame_idx", 0) <= idx:
                st   = (ev.get("stroke_type") or "?").upper()
                pid  = ev.get("player_id")
                plbl = f"P{pid}" if pid is not None else "?"
                last_stroke = f"{plbl}  {st}"
                break

        dpg.set_value("inf_time",    f"{ts:.3f}s / {total_dur:.3f}s")
        dpg.set_value("inf_rally",   "ACTIVE" if rally else "inactive")
        dpg.configure_item("inf_rally",
                           color=list(C_GREEN) if rally else list(C_TEXT_DIM))
        dpg.set_value("inf_shuttle", shut)
        dpg.set_value("inf_stroke",  last_stroke)
        # Update time slider
        if dpg.does_item_exist("sl_time"):
            dpg.set_value("sl_time", ts)

    # ------------------------------------------------------------------
    # Timeline
    # ------------------------------------------------------------------

    def _draw_timeline_static(self, tl_w: int, tl_h: int) -> None:
        """Draw rally bars and hit ticks (static — drawn once at startup)."""
        bar_y0, bar_y1 = 6, tl_h - 20
        # Background
        dpg.draw_rectangle((0, 0), (tl_w, tl_h),
                            color=(0, 0, 0, 0), fill=(18, 18, 18, 255))
        # Rally segments
        for rally in self.rally_data:
            x0 = int(rally["start_time"] * self.source_fps / max(self.total_frames, 1) * tl_w)
            x1 = int(rally["end_time"]   * self.source_fps / max(self.total_frames, 1) * tl_w)
            x1 = max(x0 + 2, x1)
            dpg.draw_rectangle(
                (x0, bar_y0), (x1, bar_y1),
                color=(0, 0, 0, 0),
                fill=(55, 120, 175, 130),
            )
        # Hit ticks
        for ev in self.events:
            fi  = ev.get("frame_idx", 0)
            pid = ev.get("player_id")
            x   = int(fi / max(self.total_frames, 1) * tl_w)
            col = list(C_P1_RGBA) if pid == self._p1_id else list(C_P2_RGBA)
            dpg.draw_line((x, bar_y0), (x, bar_y1), color=col, thickness=2)
        # Labels
        dpg.draw_text((4, tl_h - 14), "Timeline", size=11, color=list(C_TEXT_DIM))

    def _update_timeline_playhead(self, idx: int) -> None:
        if not dpg.does_item_exist("tl_playhead"):
            return
        tl_w = dpg.get_item_width("tl_draw") or 400
        tl_h = TIMELINE_H
        bar_y0, bar_y1 = 6, tl_h - 20
        x = int(idx / max(self.total_frames, 1) * tl_w)
        dpg.configure_item("tl_playhead",
                            p1=(x, 0),
                            p2=(x, tl_h))
        dpg.configure_item("tl_head_tri",
                            p1=(x, 0),
                            p2=(x - 5, 0),
                            p3=(x + 5, 0))

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------

    def _apply_theme(self) -> None:
        with dpg.theme() as g:
            with dpg.theme_component(dpg.mvAll):
                # Backgrounds
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg,        C_BG)
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg,         C_PANEL)
                dpg.add_theme_color(dpg.mvThemeCol_PopupBg,         C_PANEL)
                # Interactive
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg,         (42, 42, 42, 255))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered,  (58, 58, 58, 255))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive,   (75, 75, 75, 255))
                dpg.add_theme_color(dpg.mvThemeCol_Button,          (50, 50, 52, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered,   (72, 72, 76, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,    C_ACCENT)
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrab,      C_ACCENT)
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive,(100, 170, 255, 255))
                dpg.add_theme_color(dpg.mvThemeCol_CheckMark,       C_ACCENT)
                dpg.add_theme_color(dpg.mvThemeCol_Header,          (48, 48, 50, 255))
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered,   (66, 66, 70, 255))
                dpg.add_theme_color(dpg.mvThemeCol_HeaderActive,    C_ACCENT)
                # Separators / borders
                dpg.add_theme_color(dpg.mvThemeCol_Separator,       C_BORDER)
                dpg.add_theme_color(dpg.mvThemeCol_Border,          C_BORDER)
                dpg.add_theme_color(dpg.mvThemeCol_BorderShadow,    (0, 0, 0, 0))
                # Text / title
                dpg.add_theme_color(dpg.mvThemeCol_Text,            C_TEXT)
                dpg.add_theme_color(dpg.mvThemeCol_TitleBg,         (28, 28, 28, 255))
                dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive,   (36, 36, 36, 255))
                # Scrollbar
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg,     (20, 20, 20, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab,   (55, 55, 55, 255))
                # Rounding / spacing
                dpg.add_theme_style(dpg.mvStyleVar_WindowRounding,  6)
                dpg.add_theme_style(dpg.mvStyleVar_ChildRounding,   4)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding,   3)
                dpg.add_theme_style(dpg.mvStyleVar_GrabRounding,    3)
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding,   WIN_PAD, WIN_PAD)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing,     6, 5)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding,    5, 4)
        dpg.bind_theme(g)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _seek(self, idx: int) -> None:
        self.frame_idx = max(0, min(self.total_frames - 1, idx))
        # Any manual seek invalidates the sequential-read position.
        self._last_decoded_idx = -2
        if dpg.does_item_exist("sl_time"):
            dpg.set_value("sl_time", self.frame_idx / self.source_fps)
        # DO NOT call _push_frame here — we are inside a DearPyGui callback
        # (fired from render_dearpygui_frame).  Doing video I/O from inside
        # the DearPyGui render pass causes a macOS segfault.  Instead, set a
        # flag that the main loop processes after render_dearpygui_frame returns.
        self._needs_render = self.frame_idx

    def _cb_play_pause(self) -> None:
        self.playing = not self.playing
        if self.playing:
            # Reset tick so stale elapsed time doesn't cause an instant multi-frame skip.
            self._last_tick = time.perf_counter()
        dpg.configure_item("btn_play",
                           label="|| Pause" if self.playing else "> Play")

    def _cb_prev(self)       -> None: self._seek(self.frame_idx - 1)
    def _cb_next(self)       -> None: self._seek(self.frame_idx + 1)
    def _cb_skip_b(self)     -> None: self._seek(self.frame_idx - 10)
    def _cb_skip_f(self)     -> None: self._seek(self.frame_idx + 10)
    def _cb_skip_b5s(self)   -> None: self._seek(self.frame_idx - int(5  * self.source_fps))
    def _cb_skip_f5s(self)   -> None: self._seek(self.frame_idx + int(5  * self.source_fps))
    def _cb_skip_b20s(self)  -> None: self._seek(self.frame_idx - int(20 * self.source_fps))
    def _cb_skip_f20s(self)  -> None: self._seek(self.frame_idx + int(20 * self.source_fps))

    def _cb_frame_slider(self, _, val: int)   -> None: self._seek(int(val))
    def _cb_speed(self,        _, val: float) -> None: self.speed = float(val)

    def _cb_tog_players(self, _, v) -> None:
        self.show_players = v; self._needs_render = self.frame_idx
    def _cb_tog_shuttle(self, _, v) -> None:
        self.show_shuttle = v; self._needs_render = self.frame_idx
    def _cb_tog_hits(self, _, v)    -> None:
        self.show_hits    = v; self._needs_render = self.frame_idx
    def _cb_tog_heatmap(self, _, v) -> None:
        self.show_heatmap = v; self._needs_render = self.frame_idx
    def _cb_tog_timeline(self, _, v) -> None:
        self.show_timeline = v
        if dpg.does_item_exist("tl_panel"):
            dpg.configure_item("tl_panel", show=v)

    def _cb_key(self, _, key: int) -> None:
        if   key == dpg.mvKey_Spacebar: self._cb_play_pause()
        elif key == dpg.mvKey_Right:    self._cb_next()
        elif key == dpg.mvKey_Left:     self._cb_prev()
        elif key == dpg.mvKey_Next:     self._cb_skip_f()
        elif key == dpg.mvKey_Prior:    self._cb_skip_b()
        elif key in (dpg.mvKey_Q, dpg.mvKey_Escape): dpg.stop_dearpygui()

    # ------------------------------------------------------------------
    # UI layout
    # ------------------------------------------------------------------

    def _section_label(self, text: str) -> None:
        """White all-caps section header with separator."""
        dpg.add_spacer(height=6)
        label_tag = f"label_{text.lower().replace(' ', '_')}"
        text_item = dpg.add_text(text, color=list(C_TEXT), tag=label_tag)
        # Try to increase font size using theme
        try:
            dpg.configure_item(label_tag, size=13)
        except:
            # If size parameter not supported, just continue with default
            pass
        dpg.add_separator()
        dpg.add_spacer(height=3)

    def _build_ui(self) -> None:
        # Compute display dimensions
        vid_display_w = max(MIN_VID_W, min(self.vid_w, MAX_VID_W))
        vid_display_h = max(MIN_VID_H, int(vid_display_w * self.vid_h / max(self.vid_w, 1)))
        total_w = SIDEBAR_W + vid_display_w + PANEL_R_W + WIN_PAD * 6 + 12
        total_h = vid_display_h + TIMELINE_H + WIN_PAD * 5 + 50

        dpg.create_viewport(
            title="Badminton Vision — Viewer",
            width=total_w,
            height=total_h,
            min_width=860,
            min_height=480,
            small_icon="",
            large_icon="",
        )

        # Init texture with first frame
        first = self._render_frame(0)
        rh, rw = first.shape[:2]
        self._init_texture(rw, rh)

        # Register heatmap texture (inside the same texture registry)
        with dpg.texture_registry():
            self._init_heatmap_texture()

        with dpg.handler_registry():
            dpg.add_key_press_handler(callback=self._cb_key)

        with dpg.window(tag="root", no_scrollbar=True, no_scroll_with_mouse=True):

            with dpg.group(horizontal=True):

                # ── Left sidebar ──────────────────────────────────────
                with dpg.child_window(tag="sidebar", width=SIDEBAR_W,
                                      height=vid_display_h + TIMELINE_H + WIN_PAD * 2,
                                      border=True, no_scrollbar=False):

                    self._section_label("SOURCE")
                    run_name = self.run_dir.name
                    vid_name = self.video_path.name
                    dpg.add_text(f"{run_name[:30]}", wrap=SIDEBAR_W - 18,
                                 color=list(C_ACCENT))
                    dpg.add_text(f"{vid_name[:30]}", wrap=SIDEBAR_W - 18,
                                 color=list(C_TEXT_DIM))
                    dpg.add_text(
                        f"{self.total_frames} frames  ·  {self.source_fps:.1f} fps",
                        color=list(C_TEXT_DIM),
                    )

                    self._section_label("PLAYBACK")
                    # Condensed row: play/pause and time-based skip buttons
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="-20s", width=40, callback=self._cb_skip_b20s, tag="btn_skip_b20s")
                        dpg.add_button(label="-5s",  width=35, callback=self._cb_skip_b5s,  tag="btn_skip_b5s")
                        dpg.add_button(label="Play", width=60, callback=self._cb_play_pause, tag="btn_play")
                        dpg.add_button(label="+5s",  width=35, callback=self._cb_skip_f5s,  tag="btn_skip_f5s")
                        dpg.add_button(label="+20s", width=40, callback=self._cb_skip_f20s, tag="btn_skip_f20s")

                    # Time slider (in seconds)
                    total_dur = self.total_frames / self.source_fps
                    dpg.add_slider_float(
                        tag="sl_time",
                        label="Time",
                        default_value=0.0,
                        min_value=0.0,
                        max_value=total_dur,
                        width=SIDEBAR_W - 18,
                        callback=lambda _, v: self._seek(int(v * self.source_fps)),
                        format="%.1f s",
                    )
                    dpg.add_slider_float(
                        tag="sl_speed",
                        label="Speed",
                        default_value=1.0,
                        min_value=0.125,
                        max_value=4.0,
                        width=SIDEBAR_W - 18,
                        callback=self._cb_speed,
                        format="%.2f×",
                    )

                    self._section_label("OVERLAYS")
                    dpg.add_checkbox(tag="chk_players", label="Players  (H)",
                                     default_value=True, callback=self._cb_tog_players)
                    dpg.add_checkbox(tag="chk_shuttle", label="Shuttle  (S)",
                                     default_value=True, callback=self._cb_tog_shuttle)
                    dpg.add_checkbox(tag="chk_hits",    label="Hit events  (E)",
                                     default_value=True, callback=self._cb_tog_hits)
                    dpg.add_checkbox(tag="chk_timeline", label="Timeline",
                                     default_value=True, callback=self._cb_tog_timeline)

                    self._section_label("ANALYTICS")
                    n_rallies  = self.analytics.get("rally_count", len(self.rally_data))
                    mean_dur   = self.analytics.get("mean_rally_duration_s", 0.0)
                    n_events   = len(self.events)
                    dpg.add_text(f"Rallies      {n_rallies}")
                    dpg.add_text(f"Mean rally   {mean_dur:.1f} s")
                    dpg.add_text(f"Hit events   {n_events}")
                    per_player = self.analytics.get("per_player_hit_counts", {})
                    for pid_str, cnt in sorted(per_player.items()):
                        col = list(C_P1_RGBA) if int(pid_str) == self._p1_id else list(C_P2_RGBA)
                        dpg.add_text(f"  ID {pid_str}:  {cnt} hits", color=col)

                    self._section_label("KEYBOARD")
                    with dpg.group(horizontal=True):
                        dpg.add_text("Space",       color=list(C_ACCENT))
                        dpg.add_text("Play / Pause", color=list(C_TEXT_DIM))
                    with dpg.group(horizontal=True):
                        dpg.add_text("<- ->",       color=list(C_ACCENT))
                        dpg.add_text("Step frame", color=list(C_TEXT_DIM))
                    with dpg.group(horizontal=True):
                        dpg.add_text("PgUp / PgDn", color=list(C_ACCENT))
                        dpg.add_text("±10 frames", color=list(C_TEXT_DIM))
                    with dpg.group(horizontal=True):
                        dpg.add_text("Q / Esc",    color=list(C_ACCENT))
                        dpg.add_text("Quit",       color=list(C_TEXT_DIM))

                # ── Centre column: video + timeline ───────────────────
                with dpg.group():
                    # Video panel
                    with dpg.child_window(tag="vid_panel",
                                          width=vid_display_w,
                                          height=vid_display_h + WIN_PAD * 2,
                                          border=True,
                                          no_scrollbar=True):
                        dpg.add_image(
                            self._tex_tag,
                            tag="vid_image",
                            width=vid_display_w,
                            height=vid_display_h,
                        )

                    # Timeline strip
                    with dpg.child_window(tag="tl_panel",
                                          width=vid_display_w,
                                          height=TIMELINE_H + WIN_PAD,
                                          border=True,
                                          no_scrollbar=True):
                        tl_inner_w = vid_display_w - WIN_PAD * 2
                        with dpg.drawlist(tag="tl_draw",
                                          width=tl_inner_w,
                                          height=TIMELINE_H):
                            self._draw_timeline_static(tl_inner_w, TIMELINE_H)
                            # Playhead (updated each frame)
                            dpg.draw_line((0, 0), (0, TIMELINE_H),
                                          tag="tl_playhead",
                                          color=list(C_TEXT),
                                          thickness=2)
                            dpg.draw_triangle(
                                (0, 0), (-4, 0), (4, 0),
                                tag="tl_head_tri",
                                color=list(C_TEXT),
                                fill=list(C_TEXT),
                            )

                # ── Right info panel ──────────────────────────────────
                with dpg.child_window(tag="info_panel", width=PANEL_R_W,
                                      height=vid_display_h + TIMELINE_H + WIN_PAD * 2,
                                      border=True):

                    self._section_label("INFO")
                    for tag, lbl in [
                        ("inf_time",    "Time    "),
                        ("inf_rally",   "Rally   "),
                        ("inf_shuttle", "Shuttle "),
                        ("inf_stroke",  "Stroke  "),
                    ]:
                        with dpg.group(horizontal=True):
                            dpg.add_text(lbl, color=list(C_TEXT_DIM))
                            dpg.add_text("—", tag=tag)

                    self._section_label("COURT HEATMAP")
                    if self._heatmap_loaded and dpg.does_item_exist(self._heatmap_tex_tag):
                        dpg.add_image(
                            self._heatmap_tex_tag,
                            width=self._heatmap_w,
                            height=self._heatmap_h,
                        )
                    else:
                        dpg.add_text("No heatmap — run main.py first",
                                     color=list(C_TEXT_DIM), wrap=PANEL_R_W - 18)

        dpg.set_primary_window("root", True)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        dpg.create_context()
        self._apply_theme()
        self._build_ui()
        dpg.setup_dearpygui()
        dpg.show_viewport()

        # Show frame 0
        self._push_frame(0)
        self._last_tick = time.perf_counter()

        while dpg.is_dearpygui_running():
            now = time.perf_counter()

            # ── Playback advance ─────────────────────────────────────────
            if self.playing:
                interval = 1.0 / max(self.source_fps * self.speed, 0.1)
                if (now - self._last_tick) >= interval:
                    # Advance by however many intervals elapsed (frame-skip
                    # when rendering is slow) to prevent speed drift.
                    frames_elapsed = max(1, int((now - self._last_tick) / interval))
                    self._last_tick += frames_elapsed * interval
                    next_idx = self.frame_idx + frames_elapsed
                    if next_idx >= self.total_frames:
                        next_idx = self.total_frames - 1
                        self.playing = False
                        dpg.configure_item("btn_play", label="> Play")
                    self.frame_idx = next_idx
                    if dpg.does_item_exist("sl_time"):
                        dpg.set_value("sl_time", self.frame_idx / self.source_fps)
                    # _push_frame here is safe: we are NOT inside
                    # render_dearpygui_frame — the DearPyGui pass hasn't started.
                    self._push_frame(self.frame_idx)

            # ── DearPyGui render pass (callbacks fire here) ──────────────
            dpg.render_dearpygui_frame()

            # ── Dequeue pending seek requested by button/key callbacks ───
            # Video I/O happens HERE, safely outside the DearPyGui render
            # pass, so cap.set()/cap.read() cannot re-enter and segfault.
            if self._needs_render is not None:
                idx = self._needs_render
                self._needs_render = None
                self._push_frame(idx)

        self.cap.release()
        dpg.destroy_context()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Local overlay viewer for badminton_vision runs.\n"
            "Composites tracking overlays on the original video without "
            "re-encoding or modifying any files."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--run",
        required=True,
        metavar="RUN_DIR",
        help="Run output directory produced by main.py.",
    )
    parser.add_argument(
        "--video",
        required=True,
        metavar="VIDEO",
        help="Original source video file (must match the run).",
    )
    args = parser.parse_args()

    viewer = Viewer(run_dir=args.run, video_path=args.video)
    viewer.run()


if __name__ == "__main__":
    main()
