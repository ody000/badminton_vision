# badminton_vision — Agent Notes

Architectural constraints and module contracts. Do not deviate without updating this file.

---

## Overview

Pipeline: video → TrackNet (shuttle) + YOLOv8 (players) + RANSAC hit detection → stroke classification (MediaPipe + Transformer) → JSON artifacts + annotated MP4 + heatmap PNG.

`main.py` is headless SLURM-compatible. `tools/prepare_run.py` (local-only) handles court-corner GUI. `tools/viewer.py` (local-only DearPyGui) reviews results with overlay playback. `config.yaml` is single source of truth for all parameters.

**MOG2 is no longer part of the inference pipeline.** Background subtraction was removed from both `TrackNetTracker` and `PlayerDetector` because its foreground threshold was suppressing ≥ 97% of valid detections. All frames are now processed as-is.

---

## Key Modules

### `models/shuttle_tracknet.py` — `TrackNetTracker`
- Accepts 3 consecutive RGB frames (or 1 frame replicated ×3).
- Maintains timestamp-aware buffer; flushes if gap > 2/(fps).
- Resizes to (288, 512) before inference.
- Returns `{"shuttle": (timestamp, x, y, w, h)}` if confidence > threshold, else `{}`.
- **No MOG2 filter** — raw frames only; `mog2_manager` parameter removed.

### `models/player_yolo.py` — `PlayerDetector`
- Uses `model.track(frame, persist=True)` for stable ByteTrack IDs (person class only).
- **No MOG2 filter** — `mog2_manager` parameter and `update_mog2()` method removed.
- Returns `[{"id": int, "box": [x1,y1,x2,y2], "feet": (cx, y2)}]`.

### `core/homography.py` — `CourtMapper`
- `calibrate(image_corners)`: takes 6 pixel points (4 court corners + 2 midline).
- `transform_point(pixel_xy)` → real-world (cm) or None.
- Never opens GUI; corners from `data/input/court_points.json`.

### `core/hit_detector.py` — `HitDetector`
- Maintains deque of (timestamp, x, y) shuttle positions (N=6).
- RANSAC quadratic fit + prediction error > threshold → hit candidate.
- Gate: shuttle within `hit_proximity_cm` of player feet (real-world).
- Cooldown: reject if within `hit_cooldown_s` of last hit.

### `core/game_state.py` — `GameState`
Debounces shuttle motion, detects rally start/end, records rally segments.

Key behaviours (simplified May 2026):
- **Motion streak:** `motion_required_streak` (default 2) consecutive frames each with displacement ≥ `min_displacement_px` (default 5 px) required to start a rally.
- **Inactive timeout:** rally ends after `inactive_timeout_s` (default 1.0 s) of no qualifying motion.  End timestamp is pinned to `last_motion_timestamp`, not the silence frame.
- **Grace period:** up to `detection_grace_frames` (default 8) consecutive missed detections tolerated mid-rally without timing out.
- **Minimum duration gate:** rallies shorter than `rally_min_duration_s` (default 0.1 s) are silently discarded.
- **Removed (May 2026):** stability detector (`stable_frames`), large-displacement filter (`max_displacement_fraction`/`frame_size`), trajectory prediction (`position_history`, `_fit_trajectory`, `_predict_position`, `_detect_hit`).  The `frame_size` arg to `update()` is kept for API compatibility but ignored.

`build_rally_status_per_frame()` (module-level): post-processing second pass that merges short inactive gaps (`< rally_min_period_s`) into surrounding active rally segments.

### `core/analysis.py` — `Analysis`
Direct port of slayminton. Rally duration statistics only.

### `models/stroke_classifier.py` — `StrokeClassifier`
- MediaPipe Pose (33 joints) + trajectory (pre/post) → `stroke_transformer.py`.
- Output: `{"stroke_type": str, "confidence": float, ...}`.

### `utils/mog.py` — `MOG2Manager`
- Stateful wrapper for cv2.BackgroundSubtractorMOG2.
- `get_foreground_ratio(frame, box)` → white-pixel fraction.
- **Not used by the inference pipeline** (removed May 2026).  File retained for training augmentation scripts.

### `utils/precompute_heatmap.py` — `precompute_heatmap`
Computes a static player-footwork heatmap from `tracking_results.json` and saves it as `heatmap.png` in the run directory.

- **Inputs:** list of per-frame tracking dicts, 6-point `court_points` (BL, BR, TR, TL, midline-B, midline-T), frame dimensions, output path.
- **Process:** accumulate P1/P2 `feet` pixel positions → stamp radius-6 circles → large Gaussian blur (σ=40 by default) → `COLORMAP_JET` → tint P1 green / P2 orange (element-wise multiply) → blend 50% onto court background.
- **Court insert dimensions:** `INSERT_H=300`, `INSERT_W≈136` px (court aspect ratio 13.4 m × 6.1 m).
- **Homography:** same 6-point dst mapping as `slayminton/scripts/visualizations.py`; y-axis flipped so near-side appears at bottom of insert image.
- Called by `main.py` at end of pipeline run.  `tools/viewer.py` loads `heatmap.png` at startup (or calls `precompute_heatmap()` if the PNG is absent).
- Also exports `compute_homography()` and `video_to_insert()` used by the viewer for live player-dot positioning.

### `tools/prepare_run.py` (local-only)
- OpenCV GUI on first frame; click 6 court points (undo=z, quit=q).
- Always saves to `data/input/court_points.json` keyed by video stem.

### `tools/viewer.py` (local-only, DearPyGui)
- Loads `tracking_results.json`, `events.json`, `rally_data.json`, `analytics.json` from a run directory.
- Composites overlays (player boxes, shuttle ring, hit flash, RALLY/IDLE pill) onto the original source video without re-encoding.
- Right sidebar: frame info, recent hit events (clickable), P1/P2 IDs, **FOOTWORK HEATMAP** (static precomputed PNG with live player dots updated per frame via homography).
- Heatmap texture is a DearPyGui dynamic texture; base is loaded once, player dots are redrawn on every `_push_frame()` call.
- Shuttle ring persists for 30 frames after the last real detection (ghost ring) to bridge TrackNet dropout gaps.
- Video I/O is decoupled from DearPyGui callbacks via a `_needs_render` deferred flag to prevent macOS segfaults.

### `tools/dashboard.py` (local-only)
- Streamlit app; reads JSON artifacts (does NOT re-run pipeline).
- Rally scrubber, stroke table, heatmaps, kinematics plots.

---

## Configuration (`config.yaml`)

All parameters in one file. CLI flags and SLURM `--export` take precedence.
Key sections: MOG2, TrackNet, hit detection, rally logic, player tracking, court dims, stroke classifier, visualization, training, runtime.

---

## Output Schema (per run)

```
data/output/<video_stem>_<timestamp>/
  tracking_results.json    — per-frame: frame_idx, timestamp, shuttle, players, rally_active
  events.json              — hit events: player_id, stroke_type, confidence, trajectory_pre/post
  rally_data.json          — rally segments: rally_id, start_time, end_time, duration_s
  analytics.json           — summary: rally_count, mean/min/max duration, per-player hits
  court_points.json        — copy of 6 court corners used
  heatmap.png              — precomputed player-footwork heatmap (JET, 136×300 px)
  annotated_video.mp4      — rendered output (only with --annotate)
  logs/slurm-<jobid>.out   — SLURM stdout/stderr
```

---

## Training Pipelines

### TrackNet (`training/train_tracknet.py`)
- Input: Roboflow shuttle annotations + MOG2-augmented frames.
- DDP via `torchrun --nproc_per_node=$NGPUS`.
- Metrics: distance error (px) + mAP@0.5.
- Output: `data/output/checkpoints/`.

### YOLO (`training/train_yolo.py`)
- Input: Roboflow person + shuttle, converted to YOLO format.
- Model: YOLOv8n.
- Metrics: Ultralytics built-in mAP, precision, recall.
- Output: `models/yolo.pt`.

### Stroke Classifier (`training/train_stroke.py`)
- Input: FineBadminton Foundational Actions + MediaPipe pose + trajectory.
- Model: StrokeTransformer (Transformer encoder).
- Metrics: per-class accuracy + macro F1.
- Output: `models/stroke.pt`.

---

## Known Issues & Fixes

| Issue | Fix |
|---|---|
| TrackNet extracts wrong heatmap channel | Adopt slayminton `TrackNetTracker` (detects channel from checkpoint) |
| Court GUI never saves JSON | `tools/prepare_run.py` always saves; `main.py` only reads |
| MOG2 filter suppressed ≥ 97% of valid detections | Removed MOG2 from inference pipeline entirely (May 2026) |
| Player ID swaps after occlusion | `PlayerContext._nearest_player()` remaps ByteTrack IDs to stable P1/P2 |
| SLURM logs to submit directory | `#SBATCH -o data/output/logs/slurm-%j.out` |
| Homography + kinematics + viz in one 900-line file | Split into `core/`, `utils/visualization.py` |
| Constants scattered across 8+ files | `config.yaml` as single source |
| GameState trajectory prediction slowing down long runs | Removed `position_history`, `_fit_trajectory`, `_predict_position`, `_detect_hit` (May 2026) |
| Zero rally count when min_hits gate set to 1 | `rally_min_hits: 0` — hit detection is separate from rally gating |
| Viewer segfault on button click (macOS) | All video I/O deferred to after `render_dearpygui_frame()` via `_needs_render` flag |
| Playback speed 0.75× instead of 1.0× | `_last_tick += frames_elapsed * interval` (not `= now`) |

---

## Testing

**Tier 1 (smoke tests):** `pytest tests/ -k "not integration"` (no GPU/video)
- Feed blank/synthetic inputs to each module; assert no exceptions.

**Tier 2 (integration):** `pytest tests/test_pipeline.py` (requires `test_clip.mp4`)
- Runs `main.py` on test clip; assert all 5 JSON outputs exist and are non-empty.

---

## SLURM Rules

- `mkdir -p data/output/logs` before any invocation.
- Training: `torchrun --nproc_per_node=$NGPUS` (default NGPUS=1).
- Inference: single-process Python (no DDP).
- Pass `--config config.yaml` or override with `--set key=value`.
