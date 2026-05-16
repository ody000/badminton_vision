"""viewer.py — local overlay viewer for badminton_vision runs.

Reads tracking_results.json and events.json from a run directory and
composites overlays on the *original* source video at display time.
No pixels of the source video are modified.  No output files are written.

Usage:
    pip install dearpygui
    python viewer.py --run data/output/match_clip_20250515_120000 \\
                     --video data/input/match_clip.mp4

Keyboard shortcuts:
    Space         Play / Pause
    ← / →         Step one frame
    [ / ]  or
    PgUp / PgDn   ±10 frames
    Home / End    Jump to first / last frame
    + / -         Speed ×2 / ÷2
    H             Toggle player boxes
    S             Toggle shuttle dot
    E             Toggle hit-event flash
    M             Toggle court heatmap insert
    Q / Escape    Quit
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

SIDEBAR_W   = 230      # left control panel width
PANEL_R_W   = 215      # right info panel width
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
        self.frame_idx   = 0
        self.playing     = False
        self.speed       = 1.0
        self._last_tick  = 0.0
        self._tex_w      = 0
        self._tex_h      = 0
        self._tex_tag    = "vid_tex"

        # ── Overlay toggles ───────────────────────────────────────────
        self.show_players = True
        self.show_shuttle = True
        self.show_hits    = True
        self.show_heatmap = True

        # ── Heatmap cfg (minimal subset needed for rendering) ─────────
        self._heat_cfg = SimpleNamespace(
            court_insert_h=200,
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
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
        ok, frame = self.cap.read()
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

        # ── Shuttle glow ──────────────────────────────────────────────
        if self.show_shuttle and shuttle_j is not None:
            cx = int(shuttle_j["x"] + shuttle_j["w"] / 2)
            cy = int(shuttle_j["y"] + shuttle_j["h"] / 2)
            # Outer soft glow (drawn first, darkened)
            for r, alpha in [(18, 0.06), (13, 0.10), (9, 0.18)]:
                glow = canvas.copy()
                cv2.circle(glow, (cx, cy), r, OCV_SHUT, -1, cv2.LINE_AA)
                cv2.addWeighted(glow, alpha, canvas, 1 - alpha, 0, canvas)
            # Solid core
            cv2.circle(canvas, (cx, cy), 5, OCV_SHUT, -1, cv2.LINE_AA)
            cv2.circle(canvas, (cx, cy), 5, (0, 0, 0), 1, cv2.LINE_AA)

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

        # ── Court heatmap insert (bottom-right) ───────────────────────
        if self.show_heatmap:
            try:
                from utils.visualization import render_court_insert
                history = self._history_at(idx)
                sorted_ids = sorted(history.keys())[:2]
                display_history = {pid: history[pid] for pid in sorted_ids if history[pid]}
                if display_history:
                    insert = render_court_insert(display_history, self._heat_cfg)
                    ih, iw = insert.shape[:2]
                    if ih <= fh and iw <= fw:
                        roi = canvas[fh - ih:fh, fw - iw:fw]
                        cv2.addWeighted(insert, 0.88, roi, 0.12, 0, roi)
            except Exception:
                pass

        return canvas

    # ------------------------------------------------------------------
    # Texture management
    # ------------------------------------------------------------------

    def _frame_to_rgba_flat(self, frame: np.ndarray) -> List[float]:
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgba = cv2.cvtColor(rgb, cv2.COLOR_RGB2RGBA)
        return (rgba.astype(np.float32) / 255.0).flatten().tolist()

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
        self._update_info_panel(idx)
        self._update_timeline_playhead(idx)

    # ------------------------------------------------------------------
    # Info panel
    # ------------------------------------------------------------------

    def _update_info_panel(self, idx: int) -> None:
        if not dpg.does_item_exist("inf_frame"):
            return
        t      = self._tracking_by_frame.get(idx, {})
        ts     = t.get("timestamp", idx / self.source_fps)
        rally  = t.get("rally_active", False)
        shut   = "detected" if t.get("shuttle") else "—"
        npl    = len(t.get("players", []))
        dpg.set_value("inf_frame",   f"{idx:>6d} / {self.total_frames - 1}")
        dpg.set_value("inf_time",    f"{ts:.3f} s")
        dpg.set_value("inf_rally",   "ACTIVE" if rally else "inactive")
        dpg.set_value("inf_rally",   "ACTIVE" if rally else "inactive")
        dpg.configure_item("inf_rally",
                           color=list(C_GREEN) if rally else list(C_TEXT_DIM))
        dpg.set_value("inf_shuttle", shut)
        dpg.set_value("inf_players", str(npl))

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
        dpg.set_value("sl_frame", self.frame_idx)
        self._push_frame(self.frame_idx)

    def _cb_play_pause(self) -> None:
        self.playing = not self.playing
        dpg.configure_item("btn_play",
                           label="⏸  Pause" if self.playing else "▶  Play")

    def _cb_prev(self)      -> None: self._seek(self.frame_idx - 1)
    def _cb_next(self)      -> None: self._seek(self.frame_idx + 1)
    def _cb_skip_b(self)    -> None: self._seek(self.frame_idx - 10)
    def _cb_skip_f(self)    -> None: self._seek(self.frame_idx + 10)
    def _cb_first(self)     -> None: self._seek(0)
    def _cb_last(self)      -> None: self._seek(self.total_frames - 1)

    def _cb_frame_slider(self, _, val: int)   -> None: self._seek(int(val))
    def _cb_speed(self,        _, val: float) -> None: self.speed = float(val)

    def _cb_tog_players(self, _, v) -> None:
        self.show_players = v; self._push_frame(self.frame_idx)
    def _cb_tog_shuttle(self, _, v) -> None:
        self.show_shuttle = v; self._push_frame(self.frame_idx)
    def _cb_tog_hits(self, _, v)    -> None:
        self.show_hits    = v; self._push_frame(self.frame_idx)
    def _cb_tog_heatmap(self, _, v) -> None:
        self.show_heatmap = v; self._push_frame(self.frame_idx)

    def _cb_key(self, _, key: int) -> None:
        if   key == dpg.mvKey_Space:  self._cb_play_pause()
        elif key == dpg.mvKey_Right:  self._cb_next()
        elif key == dpg.mvKey_Left:   self._cb_prev()
        elif key == dpg.mvKey_Next:   self._cb_skip_f()
        elif key == dpg.mvKey_Prior:  self._cb_skip_b()
        elif key == dpg.mvKey_Home:   self._cb_first()
        elif key == dpg.mvKey_End:    self._cb_last()
        elif key == dpg.mvKey_H:      dpg.set_value("chk_players", not self.show_players); self._cb_tog_players(None, not self.show_players)
        elif key == dpg.mvKey_S:      dpg.set_value("chk_shuttle", not self.show_shuttle); self._cb_tog_shuttle(None, not self.show_shuttle)
        elif key == dpg.mvKey_E:      dpg.set_value("chk_hits",    not self.show_hits);    self._cb_tog_hits(None, not self.show_hits)
        elif key == dpg.mvKey_M:      dpg.set_value("chk_heatmap", not self.show_heatmap); self._cb_tog_heatmap(None, not self.show_heatmap)
        elif key in (dpg.mvKey_Q, dpg.mvKey_Escape): dpg.stop_dearpygui()

    # ------------------------------------------------------------------
    # UI layout
    # ------------------------------------------------------------------

    def _section_label(self, text: str) -> None:
        """Dimmed all-caps section header with separator."""
        dpg.add_spacer(height=4)
        dpg.add_text(text, color=list(C_TEXT_DIM))
        dpg.add_separator()
        dpg.add_spacer(height=2)

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
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="|◀", width=30, callback=self._cb_first,  tag="btn_first")
                        dpg.add_button(label="⏮", width=30, callback=self._cb_skip_b,  tag="btn_skipb")
                        dpg.add_button(label="◀",  width=28, callback=self._cb_prev,    tag="btn_prev")
                        dpg.add_button(label="▶  Play", width=76,
                                       callback=self._cb_play_pause, tag="btn_play")
                        dpg.add_button(label="▶",  width=28, callback=self._cb_next,    tag="btn_next")
                        dpg.add_button(label="⏭", width=30, callback=self._cb_skip_f,  tag="btn_skipf")
                        dpg.add_button(label="▶|", width=30, callback=self._cb_last,    tag="btn_last")

                    dpg.add_slider_int(
                        tag="sl_frame",
                        label="Frame",
                        default_value=0,
                        min_value=0,
                        max_value=max(self.total_frames - 1, 1),
                        width=SIDEBAR_W - 18,
                        callback=self._cb_frame_slider,
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
                    dpg.add_checkbox(tag="chk_heatmap", label="Court heatmap  (M)",
                                     default_value=True, callback=self._cb_tog_heatmap)

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
                    for k, v in [
                        ("Space",    "Play / Pause"),
                        ("← →",      "Step frame"),
                        ("PgUp/Dn",  "±10 frames"),
                        ("Home/End", "First/Last"),
                        ("H S E M",  "Toggle overlays"),
                        ("Q / Esc",  "Quit"),
                    ]:
                        with dpg.group(horizontal=True):
                            dpg.add_text(f"{k:<10}", color=list(C_ACCENT))
                            dpg.add_text(v, color=list(C_TEXT_DIM))

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

                    self._section_label("FRAME INFO")
                    for tag, lbl in [
                        ("inf_frame",   "Frame   "),
                        ("inf_time",    "Time    "),
                        ("inf_rally",   "Rally   "),
                        ("inf_shuttle", "Shuttle "),
                        ("inf_players", "Players "),
                    ]:
                        with dpg.group(horizontal=True):
                            dpg.add_text(lbl, color=list(C_TEXT_DIM))
                            dpg.add_text("—", tag=tag)

                    self._section_label("RECENT EVENTS")
                    # Show last 10 events, most recent first
                    recent = self.events[-10:][::-1]
                    for i, ev in enumerate(recent):
                        ts   = ev.get("timestamp", 0.0)
                        st   = (ev.get("stroke_type") or "?").upper()
                        pid  = ev.get("player_id")
                        plbl = f"P{pid}" if pid is not None else "?"
                        col  = list(C_P1_RGBA) if pid == self._p1_id else list(C_P2_RGBA)
                        fi   = ev.get("frame_idx", 0)
                        row  = f"● {ts:6.2f}s  {plbl}  {st}"
                        dpg.add_text(row, tag=f"ev_{i}", color=col)
                        # Make events clickable to jump to that frame
                        with dpg.item_handler_registry(tag=f"ev_hr_{i}"):
                            dpg.add_item_clicked_handler(
                                callback=lambda _, __, fi=fi: self._seek(fi)
                            )
                        dpg.bind_item_handler_registry(f"ev_{i}", f"ev_hr_{i}")

                    self._section_label("PLAYERS")
                    if self._p1_id is not None:
                        dpg.add_text(f"P1  ID {self._p1_id}", color=list(C_P1_RGBA))
                    if self._p2_id is not None:
                        dpg.add_text(f"P2  ID {self._p2_id}", color=list(C_P2_RGBA))
                    if self._p1_id is None and self._p2_id is None:
                        dpg.add_text("No players detected", color=list(C_TEXT_DIM))

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

            if self.playing:
                interval = 1.0 / max(self.source_fps * self.speed, 0.1)
                if (now - self._last_tick) >= interval:
                    self._last_tick = now
                    next_idx = self.frame_idx + 1
                    if next_idx >= self.total_frames:
                        next_idx = self.total_frames - 1
                        self.playing = False
                        dpg.configure_item("btn_play", label="▶  Play")
                    self.frame_idx = next_idx
                    dpg.set_value("sl_frame", self.frame_idx)
                    self._push_frame(self.frame_idx)

            dpg.render_dearpygui_frame()

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
