"""Typed data contracts for the badminton vision pipeline.

All inter-module data exchange uses these dataclasses so that coordinate
space ambiguity (pixel vs real-world cm) and optional fields are explicit
at every call site rather than buried in dict accesses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Player
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Player:
    """Single player detection with pixel and optional real-world coordinates.

    feet_px:   bottom-centre of bounding box in pixel space  (x_px, y_px).
    feet_real: transformed to real-world court space (x_cm, y_cm) after
               CourtMapper.transform_point(), or None when uncalibrated.
    """

    id: int
    box: List[float]                               # [x1, y1, x2, y2] pixels
    feet_px: Tuple[float, float]                   # (x_px, y_px)
    feet_real: Optional[Tuple[float, float]] = None  # (x_cm, y_cm)

    def to_dict(self) -> dict:
        """Serialise for JSON output (tracking_results.json)."""
        return {
            "id": self.id,
            "box": self.box,
            "feet_px": list(self.feet_px),
            "feet_real": list(self.feet_real) if self.feet_real is not None else None,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Shuttle
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Shuttle:
    """Shuttle detection from TrackNet.

    x, y:  top-left corner of bounding box in pixels.
    w, h:  width and height in pixels.
    """

    timestamp: float
    x: float
    y: float
    w: float
    h: float
    confidence: float = 1.0

    @property
    def center_px(self) -> Tuple[float, float]:
        """Centre of bounding box in pixel space."""
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    @classmethod
    def from_tuple(cls, t: tuple) -> "Shuttle":
        """Create from TrackNet's (timestamp, x, y, w, h) tuple."""
        return cls(
            timestamp=float(t[0]),
            x=float(t[1]),
            y=float(t[2]),
            w=float(t[3]),
            h=float(t[4]),
        )

    def to_dict(self) -> dict:
        """Serialise for JSON output."""
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}


# ─────────────────────────────────────────────────────────────────────────────
# HitEvent
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HitEvent:
    """Input contract for StrokeClassifier.classify().

    trajectory_pre:  list of (t, x_px, y_px) tuples *before* the hit.
    trajectory_post: list of (t, x_px, y_px) tuples *after* the hit
                     (empty when classifying in real-time).
    keyframe:        BGR np.ndarray of the frame at hit time, or None.
    surrounding_frames:     list of BGR np.ndarray frames around hit (Phase 4-A multi-frame).
    surrounding_timestamps: list of float timestamps for surrounding_frames.
    """

    trajectory_pre: List[Tuple[float, float, float]] = field(default_factory=list)
    trajectory_post: List[Tuple[float, float, float]] = field(default_factory=list)
    keyframe: Optional[object] = None  # np.ndarray | None — avoids numpy import here
    surrounding_frames: List[object] = field(default_factory=list)  # list of BGR np.ndarray
    surrounding_timestamps: List[float] = field(default_factory=list)
