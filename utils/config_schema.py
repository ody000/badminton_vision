"""Typed configuration schema for badminton_vision.

All tunable parameters live here with their types and defaults.
Modules that need config should call ConfigSchema.from_namespace(cfg)
once in __init__ instead of scattering getattr(cfg, "key", default) calls.

Example:
    from utils.config_schema import ConfigSchema
    cfg = ConfigSchema.from_namespace(load_config("config.yaml"))
    self.n = cfg.hit_trajectory_n        # typed int, IDE-navigable
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class ConfigSchema:
    """Single source of truth for all config keys, types, and defaults."""

    # ── MOG2 ─────────────────────────────────────────────────────────────────
    mog2_warmup_frames: int = 150
    mog2_foreground_thresh_player: float = 0.06
    mog2_foreground_thresh_shuttle: float = 0.05
    mog2_var_threshold: float = 200.0
    mog2_history: int = 1000

    # ── TrackNet ──────────────────────────────────────────────────────────────
    tracknet_buffer_size: int = 3
    tracknet_box_size: int = 16
    tracknet_expected_h: int = 288
    tracknet_expected_w: int = 512
    tracknet_conf_threshold: float = 0.001
    tracknet_weights: str = "weights/tracknet.pt"

    # ── Hit detection ─────────────────────────────────────────────────────────
    hit_trajectory_n: int = 6
    hit_cooldown_s: float = 0.2
    hit_proximity_cm: float = 200.0
    hit_prediction_error_threshold: float = 20.0
    hit_ransac_iterations: int = 10
    hit_ransac_min_inliers: int = 3

    # ── Rally ─────────────────────────────────────────────────────────────────
    rally_inactive_timeout_s: float = 1.0
    rally_min_period_s: float = 0.5
    rally_motion_required_streak: int = 3
    rally_max_displacement_fraction: float = 0.1667
    rally_min_displacement_px: float = 2.0
    rally_stable_frame_threshold: int = 5

    # ── Player tracking ───────────────────────────────────────────────────────
    player_conf_threshold: float = 0.5
    player_bytetrack_history: int = 1000
    player_weights: str = "weights/yolo_badminton.pt"

    # ── Court ─────────────────────────────────────────────────────────────────
    court_real_width_cm: float = 610.0
    court_real_length_cm: float = 1340.0
    court_points_file: str = "data/input/court_points.json"

    # ── Stroke classifier ─────────────────────────────────────────────────────
    stroke_pose_joints: int = 33
    stroke_trajectory_n: int = 6
    stroke_weights: str = "weights/stroke_classifier.pt"

    # ── Visualization ─────────────────────────────────────────────────────────
    court_insert_h: int = 300
    court_insert_alpha: float = 0.82
    # Legacy blur knob (kept for back-compat; prefer heatmap_gaussian_sigma)
    player_heatmap_blur: int = 7
    # Gaussian sigma for heatmap smoothing — larger = softer gradient.
    # Kernel size is derived as ceil(6*sigma)|1 so changing this one
    # value controls the entire smoothing envelope.
    heatmap_gaussian_sigma: int = 25
    player_stamp_radius: int = 6
    player_p1_color_bgr: List[int] = field(default_factory=lambda: [57, 255, 20])
    player_p2_color_bgr: List[int] = field(default_factory=lambda: [0, 165, 255])

    # ── Training ──────────────────────────────────────────────────────────────
    train_val_split: float = 0.8
    train_iou_threshold: float = 0.5
    train_epochs_tracknet: int = 50
    train_epochs_yolo: int = 50
    train_epochs_stroke: int = 30
    train_batch_size: int = 16
    train_learning_rate: float = 3.0e-4
    train_min_lr: float = 1.0e-6
    train_num_workers: int = 4
    train_data_dir: str = "data/input/train_mog_reflect"
    train_finebadminton_dir: str = "data/input/finebadminton"

    # ── Runtime ───────────────────────────────────────────────────────────────
    fps: float = 30.0
    device: str = "cpu"
    output_dir: str = "data/output"

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_namespace(cls, ns) -> "ConfigSchema":
        """Build a ConfigSchema from a SimpleNamespace (load_config output).

        Unknown keys on the namespace are silently ignored.
        Missing keys fall back to the class defaults declared above.
        Type coercion is NOT performed — YAML already parsed the types.
        """
        schema = cls()
        if ns is None:
            return schema
        for fname in schema.__dataclass_fields__:
            if hasattr(ns, fname):
                setattr(schema, fname, getattr(ns, fname))
        return schema

    def to_namespace(self):
        """Convert back to a SimpleNamespace for modules that still use getattr."""
        from types import SimpleNamespace
        return SimpleNamespace(**{
            fname: getattr(self, fname)
            for fname in self.__dataclass_fields__
        })
