"""PlayerContext: single module that owns the full player lifecycle.

Responsibilities:
  - Consume raw YOLO detection dicts each frame.
  - Apply CourtMapper homography to produce real-world feet coordinates.
  - Maintain stable P1/P2 assignment by first-seen ByteTrack ID.
  - Accumulate per-player feet history for heatmap rendering.

This concentrates all per-player bookkeeping that was previously scattered
across main.py (inline accumulation), visualization.py (P1/P2 sorting), and
hit_detector.py (coordinate space) into one deep module.

Interface (what callers need to know):
  ctx = PlayerContext()
  players: List[Player] = ctx.update(raw_dicts, court_mapper)
  history: Dict[int, List] = ctx.get_feet_history(max_players=2)
  ctx.p1_id  # stable int or None
  ctx.p2_id  # stable int or None
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from core.tracking_types import Player


class PlayerContext:
    """Owns player identity, coordinate transform, and position history.

    Usage per frame:
        players = ctx.update(raw_yolo_dicts, court_mapper)
        insert  = render_court_insert(ctx.get_feet_history(2), cfg)

    Player identity is capped at exactly 2 slots (P1, P2).  Once both slots
    are filled, any new ByteTrack ID that appears (due to re-ID after
    occlusion) is remapped to whichever existing player's last-known feet
    position is spatially closest to the new detection.  This prevents P3/P4/…
    from ever being registered.
    """

    def __init__(self) -> None:
        # Ordered list of ByteTrack IDs in first-appearance order.
        # Invariant: once appended, order never changes — P1/P2 are stable.
        # Maximum length is capped at 2.
        self._seen_ids: List[int] = []
        # Cumulative real-world feet positions per player ID.
        self._feet_history: Dict[int, List[Tuple[float, float]]] = {}
        # Last pixel feet position per registered player ID (for remapping).
        self._last_feet_px: Dict[int, Tuple[float, float]] = {}
        # Map: unknown/new ByteTrack ID → established player ID
        self._id_remap: Dict[int, int] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, raw_detections: List[dict], court_mapper) -> List[Player]:
        """Process one frame of raw YOLO output.

        Args:
            raw_detections: List of dicts from PlayerDetector.detect():
                {"id": int, "box": [...], "feet": (x_px, y_px)}
            court_mapper: CourtMapper instance (may be uncalibrated).

        Returns:
            List of typed Player objects with feet_real filled in if
            court_mapper.is_calibrated() is True.
        """
        players: List[Player] = []
        calibrated = (
            court_mapper is not None and court_mapper.is_calibrated()
        )

        for d in raw_detections:
            raw_pid = int(d["id"])

            feet_px: Tuple[float, float] = (
                float(d["feet"][0]),
                float(d["feet"][1]),
            )

            # ── ID assignment / remapping ──────────────────────────────
            if raw_pid in self._seen_ids:
                # Known established player — use as-is.
                pid = raw_pid
            elif raw_pid in self._id_remap:
                # Previously remapped transient ID.
                pid = self._id_remap[raw_pid]
            elif len(self._seen_ids) < 2:
                # Still have open slots — register this as a new player.
                self._seen_ids.append(raw_pid)
                pid = raw_pid
            else:
                # Both slots filled; remap to the spatially nearest player.
                pid = self._nearest_player(feet_px)
                self._id_remap[raw_pid] = pid

            # Update last known pixel position for the established slot.
            self._last_feet_px[pid] = feet_px

            feet_real: Optional[Tuple[float, float]] = None
            if calibrated:
                result = court_mapper.transform_point(feet_px)
                if result is not None:
                    feet_real = (float(result[0]), float(result[1]))

            player = Player(
                id=pid,
                box=list(d["box"]),
                feet_px=feet_px,
                feet_real=feet_real,
            )
            players.append(player)

            # Accumulate history (real-world only — pixel history not needed)
            if pid not in self._feet_history:
                self._feet_history[pid] = []
            if feet_real is not None:
                self._feet_history[pid].append(feet_real)

        return players

    def get_feet_history(self, max_players: int = 2) -> Dict[int, List[Tuple[float, float]]]:
        """Return accumulated real-world feet positions for the first N players.

        Args:
            max_players: How many players to include (P1, then P2, …).

        Returns:
            {player_id: [(x_cm, y_cm), ...]} for the first max_players IDs.
            Empty dict when no history is available.
        """
        sorted_ids = sorted(self._seen_ids)[:max_players]
        return {
            pid: self._feet_history[pid]
            for pid in sorted_ids
            if pid in self._feet_history and self._feet_history[pid]
        }

    def player_feet_real_list(self, players: List[Player]) -> List[dict]:
        """Build the player_feet_real list expected by HitDetector.update().

        Returns:
            [{"id": int, "feet_real": (x_cm, y_cm) | None}, ...]
        """
        return [{"id": p.id, "feet_real": p.feet_real} for p in players]

    # ------------------------------------------------------------------
    # P1 / P2 stable assignments
    # ------------------------------------------------------------------

    @property
    def p1_id(self) -> Optional[int]:
        """ByteTrack ID of the first player seen (stable across the run)."""
        return self._seen_ids[0] if self._seen_ids else None

    @property
    def p2_id(self) -> Optional[int]:
        """ByteTrack ID of the second player seen (stable across the run)."""
        return self._seen_ids[1] if len(self._seen_ids) >= 2 else None

    @property
    def seen_ids(self) -> List[int]:
        """All seen ByteTrack IDs in first-appearance order (read-only copy)."""
        return list(self._seen_ids)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _nearest_player(self, feet_px: Tuple[float, float]) -> int:
        """Return the established player ID whose last known feet position
        is closest (Euclidean, pixel space) to *feet_px*.

        Called only when both P1/P2 slots are filled.  Falls back to
        self._seen_ids[0] if no last_feet_px is recorded yet.
        """
        best_id   = self._seen_ids[0]
        best_dist = float("inf")
        fx, fy    = feet_px
        for pid in self._seen_ids:
            if pid in self._last_feet_px:
                lx, ly = self._last_feet_px[pid]
                dist   = math.hypot(fx - lx, fy - ly)
                if dist < best_dist:
                    best_dist = dist
                    best_id   = pid
        return best_id

    def reset(self) -> None:
        """Clear all state (useful for unit tests)."""
        self._seen_ids.clear()
        self._feet_history.clear()
        self._last_feet_px.clear()
        self._id_remap.clear()
