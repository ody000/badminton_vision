# badminton_vision — Agent Notes

Architectural constraints and module contracts. Do not deviate without updating this file.

---

## Overview

Pipeline: video → TrackNetV3 (shuttle) + YOLOv8 (players, centroid-tracked) + RANSAC hit detection → stroke classification (MediaPipe + StrokeTransformer) → JSON artifacts + annotated MP4 + heatmap PNG.

`main.py` is headless SLURM-compatible. `tools/prepare_run.py` (local-only) handles court-corner GUI. `tools/viewer.py` (local-only DearPyGui) reviews results with overlay playback. `config.yaml` is single source of truth for all parameters.

**MOG2 is no longer part of the inference pipeline.** Background subtraction was removed from inference because its foreground threshold was suppressing ≥ 97% of valid detections. All frames are now processed as-is.

**TrackNetV2 (shuttle_tracknet.py) has been removed.** Production pipeline uses TrackNetV3 exclusively (`tracknet_version: 3` in config). Legacy V2 code was cleaned up (May 21, 2026).

---

## Key Modules

### `models/shuttle_tracknetv3.py` — `TrackNetV3Tracker`
- Accepts a single BGR frame + timestamp; requires pre-computed background tensor.
- Maintains internal 3-frame buffer (handles timestamp gaps > 2/(fps)).
- Resizes to (288, 512) before inference.
- Returns `{"shuttle": (timestamp, x, y, w, h)}` if confidence > threshold, else `{}`.
- `detect_batch(frames, timestamps)`: batches N frames for GPU efficiency. **CUDA only**; CPU falls back to sequential `detect()`.
- **No MOG2 filter** — raw frames; background subtraction is implicit in TrackNetV3 architecture.
- **Background pre-computation:** `estimate_background(video_path, n_frames=150)` extracts mean frame from first N frames; stored in memory for all subsequent inferences.

### `models/player_yolo.py` — `PlayerDetector` (YOLOv8)
- CNN-based detection: YOLOv8n fine-tuned on badminton player dataset.
- Per-frame inference: ~5 ms/frame on GPU; deterministic predictions.
- Multi-player support with **centroid-based persistent ID tracking** (no Kalman filter):
  - Computes centroid of each detection bounding box.
  - Matches current-frame centroids to previous-frame positions using exhaustive search (n≤4 players) or greedy matching (n>4).
  - Assigns persistent IDs based on optimal matching; new detections get new IDs; unmatched old IDs are retired.
  - Max matching distance: 200 px (configurable `_max_distance`).
  - Performance: O(n³) on n=2 players ≈ negligible overhead.
- Returns `[{"id": 0, "box": [x1,y1,x2,y2], "feet": (cx, y2)}]` or `[]` if no detections.
- `player_conf_threshold` (default 0.5): min confidence to report a detection.
- `set_detect_interval()` method: skip inference every N frames for throughput trade-off.

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

### `models/stroke_classifier.py` & `models/stroke_transformer.py`
- **StrokeClassifier** wrapper: MediaPipe Pose (33 joints) + pre/post trajectory → **StrokeTransformer** model.
- **StrokeTransformer**: Transformer encoder + multi-head classification (100+ stroke classes).
- Output: `{"stroke_type": str, "confidence": float, ...}` per HitEvent.
- `stroke_classify_enabled` (default `false` in `config.yaml`): skips MediaPipe init and model load entirely. Set to `true` for local post-processing runs where stroke labels are needed. SLURM inference jobs leave this off (CPU bottleneck).

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

### DINOv3 Player Detection (`models/player_dino.py` — `train_dino` function)
- Input: COCO-format player annotations (min 5K images, recommended 10K+ for 88%+ accuracy).
- Model: DINOv3-ViT (ViT backbone + 2-layer detection head).
- Optimizations: removed SSL loss, removed EMA teacher, cosine annealing LR, gradient clipping.
- Metrics: IoU, mAP@0.5.
- Output: `models/dino_player.pt`.
- **Training time:** 30–60 min on GPU (50 epochs, 16 batch size, 10K images); 3–5 hours on CPU.

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
| Player ID swaps after occlusion | `PlayerContext._nearest_player()` remaps ordinal IDs to stable P1/P2 |
| SLURM logs to submit directory | `#SBATCH -o data/output/logs/slurm-%j.out` |
| Homography + kinematics + viz in one 900-line file | Split into `core/`, `utils/visualization.py` |
| Constants scattered across 8+ files | `config.yaml` as single source |
| GameState trajectory prediction slowing down long runs | Removed `position_history`, `_fit_trajectory`, `_predict_position`, `_detect_hit` (May 2026) |
| Zero rally count when min_hits gate set to 1 | `rally_min_hits: 0` — hit detection is separate from rally gating |
| Viewer segfault on button click (macOS) | All video I/O deferred to after `render_dearpygui_frame()` via `_needs_render` flag |
| Playback speed 0.75× instead of 1.0× | `_last_tick += frames_elapsed * interval` (not `= now`) |
| YOLO silently falling back to CPU on OSCAR | Pass `device=` explicitly to `model.predict()` (May 2026) |
| ByteTrack Kalman overhead slowing player detection | Replaced `model.track(persist=True)` with `model.predict()`; IDs are frame-local ordinals (May 2026) |
| `detect_batch()` slower than sequential on CPU | Gate batched TrackNet path on `cfg.device.startswith("cuda")`; CPU falls back to `detect()` (May 2026) |
| MediaPipe loading at startup even in headless SLURM runs | `stroke_classify_enabled: false` in `config.yaml` skips init entirely (May 2026) |
| `player_feet_real_list()` called on every frame | Gate on shuttle presence — skip homography transform when shuttle not detected (May 2026) |

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
