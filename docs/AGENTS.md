# badminton_vision — Agent Notes

Architectural constraints and module contracts. Do not deviate without updating this file.

---

## Overview

Pipeline: video → TrackNet (shuttle) + YOLOv8 (players) + RANSAC hit detection → stroke classification (MediaPipe + Transformer) → JSON artifacts + annotated MP4.

`main.py` is headless SLURM-compatible. `tools/prepare_run.py` (local-only) handles court-corner GUI. `tools/dashboard.py` (local-only Streamlit) reviews results. `config.yaml` is single source of truth for all parameters.

---

## Key Modules

### `models/shuttle_tracknet.py` — `TrackNetTracker`
- Accepts 3 consecutive RGB frames (or 1 frame replicated ×3).
- Maintains timestamp-aware buffer; flushes if gap > 2/(fps).
- Resizes to (288, 512) before inference.
- Returns `{"shuttle": (timestamp, x, y, w, h)}` if confidence > threshold, else `{}`.
- Applies MOG2 foreground filter (reject if white-pixel ratio < 5%, after warmup).

### `models/player_yolo.py` — `PlayerDetector`
- Uses `model.track(frame, persist=True)` for stable ByteTrack IDs (person class only).
- Applies MOG2 foreground filter after warmup (reject if foreground < 6% of bbox).
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
Direct port of slayminton. Debounces motion, merges short gaps, detects rally start/end.

### `core/analysis.py` — `Analysis`
Direct port of slayminton. Rally duration statistics only.

### `models/stroke_classifier.py` — `StrokeClassifier`
- MediaPipe Pose (33 joints) + trajectory (pre/post) → `stroke_transformer.py`.
- Output: `{"stroke_type": str, "confidence": float, ...}`.

### `utils/mog.py` — `MOG2Manager`
- Stateful wrapper for cv2.BackgroundSubtractorMOG2.
- `get_foreground_ratio(frame, box)` → white-pixel fraction.

### `tools/prepare_run.py` (local-only)
- OpenCV GUI on first frame; click 6 court points (undo=z, quit=q).
- Always saves to `data/input/court_points.json` keyed by video stem.

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
  annotated_video.mp4      — rendered output
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
| MOG2 cold-start false positives | 150-frame warmup before enabling filter |
| Player ID swaps | ByteTrack persistent IDs |
| SLURM logs to submit directory | `#SBATCH -o data/output/logs/slurm-%j.out` |
| Homography + kinematics + viz in one 900-line file | Split into `core/`, `utils/visualization.py` |
| Constants scattered across 8+ files | `config.yaml` as single source |

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
