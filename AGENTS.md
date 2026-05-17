# badminton_vision — Agent Notes

This file is the authoritative source of architectural constraints, module contracts,
and implementation rules for the badminton_vision clean-slate system.
Do not deviate from these rules without updating this file first.

---

## Overview

badminton_vision is a computer-vision pipeline for badminton match analysis.
It ingests a video, tracks the shuttlecock (TrackNet) and players (YOLOv8 + ByteTrack),
detects hits and rally boundaries, classifies stroke types (MediaPipe pose +
trajectory → Transformer), and writes structured JSON artifacts + an annotated MP4.

A separate Streamlit dashboard (`dashboard.py`) consumes the JSON artifacts for
interactive review. The pipeline itself (`main.py`) is 100% headless and
OSCAR/SLURM-compatible. A local setup helper (`prepare_run.py`) handles
court-corner selection via GUI and saves the result to JSON before any SLURM job runs.

`slayminton/` is a read-only reference implementation. Do not modify it.

---

## Repository Layout

```
badminton_vision/
  config.yaml                   # single source of truth for all tunable parameters
  main.py                       # headless CLI entrypoint
  prepare_run.py                # local-only: court-corner GUI → config JSON
  dashboard.py                  # local-only: Streamlit results viewer
  core/
    homography.py               # CourtMapper: pixel ↔ real-world transform
    hit_detector.py             # quadratic fit + RANSAC + Kalman interface + proximity gate
    game_state.py               # rally active/inactive logic (from slayminton)
    analysis.py                 # rally statistics (from slayminton)
  models/
    shuttle_tracknet.py         # TrackNetTracker wrapper (from slayminton) + timestamp buffer
    player_yolo.py              # YOLOv8 + ByteTrack + MOG2 confidence filter
    stroke_classifier.py        # MediaPipe pose + trajectory → stroke type
    TrackNet.py                 # TrackNet model architecture (from slayminton, unchanged)
  utils/
    mog.py                      # MOG2 background subtraction helpers
    video_io.py                 # frame extraction, VideoIOHandler
    visualization.py            # rendering only: boxes, court insert, heatmap overlays
    data_download.py            # Roboflow + FineBadminton download scripts
  training/
    train_tracknet.py           # TrackNet fine-tuning (DDP-ready)
    train_yolo.py               # YOLO fine-tuning (DDP-ready)
    train_stroke.py             # stroke classifier training
    evaluate.py                 # shared IoU / mAP / distance-error metrics
  scripts/
    augment_data_reflect.py     # horizontal-flip augmentation (ported from slayminton)
  tests/
    test_hit_detector.py        # Tier 1 smoke tests
    test_court_mapper.py
    test_tracknet.py
    test_yolo.py
    test_game_state.py
    test_pipeline.py            # Tier 2 integration test (requires test_clip.mp4)
  data/
    input/
      test_clip.mp4             # short clip for integration tests (check in)
      court_points.json         # per-video court corners, keyed by video stem
      train/                    # Roboflow COCO dataset (person + shuttle)
      train_mog_frames/         # MOG2-masked version of train/
      train_mog_reflect/        # horizontal-flip augmented version
      finebadminton/            # FineBadminton dataset (all 3 annotation levels)
    output/
      logs/                     # SLURM stdout/stderr → #SBATCH -o data/output/logs/slurm-%j.out
      <video_name>_<timestamp>/ # one folder per pipeline run (see Output Schema)
  weights/
    tracknet.pt                 # best TrackNet weights (from slayminton/models/tracknet.pt)
    yolo_badminton.pt           # fine-tuned YOLOv8 weights
    stroke_classifier.pt        # trained stroke classifier weights
  slurm_train.sh                # OSCAR launcher: train-tracknet | train-yolo | train-stroke
  slurm_track.sh                # OSCAR launcher: track-video
```

---

## Configuration (`config.yaml`)

All tunable parameters live here. Modules import via a `load_config()` helper.
CLI flags and SLURM `--export` overrides take precedence at runtime.

```yaml
# MOG2
mog2_warmup_frames: 150          # frames before MOG2 filter is enabled
mog2_foreground_thresh_player: 0.06  # min foreground coverage in player bbox
mog2_foreground_thresh_shuttle: 0.05 # min white-pixel ratio in shuttle bbox

# TrackNet
tracknet_buffer_size: 3
tracknet_box_size: 16
tracknet_expected_h: 288
tracknet_expected_w: 512
tracknet_conf_threshold: 0.001

# Hit detection
hit_trajectory_n: 6             # frames for quadratic fit (pre + post hit)
hit_cooldown_s: 0.2
hit_proximity_cm: 200           # max player-shuttle distance to attribute a hit
hit_prediction_error_threshold: 20.0  # pixels; deviation to declare a hit
hit_ransac_iterations: 10

# Rally
rally_inactive_timeout_s: 1.0
rally_min_period_s: 0.5         # short gaps below this are merged (from slayminton)
rally_motion_required_streak: 3
rally_max_displacement_fraction: 0.1667  # 1/6 of frame dimension

# Player tracking
player_bytetrack_history: 1000
player_mog2_warmup_frames: 150  # same value, explicit alias for clarity

# Court
court_real_width_cm: 610
court_real_length_cm: 1340

# Stroke classifier
stroke_pose_joints: 33          # MediaPipe Pose joint count
stroke_trajectory_n: 6         # pre + post shuttle positions

# Visualization
court_insert_h: 300
court_insert_alpha: 0.82
player_heatmap_blur: 7
player_p1_color_bgr: [57, 255, 20]
player_p2_color_bgr: [0, 165, 255]

# Training
train_val_split: 0.8
train_iou_threshold: 0.5
train_epochs_tracknet: 50
train_epochs_yolo: 50
train_epochs_stroke: 30
train_batch_size: 16
train_learning_rate: 3.0e-4
train_num_workers: 4

# Runtime
fps: 30.0
device: cpu                     # overridden to cuda by SLURM jobs
```

---

## Module Contracts

### `models/shuttle_tracknet.py` — `TrackNetTracker`

Source: port of `slayminton/models/tracknet.py` `TrackNetTracker`.

- Accepts a list/tuple of 3 consecutive RGB frames OR a single frame (replicated ×3).
- Maintains a **timestamp-aware frame buffer**: if the gap between two consecutive
  frames exceeds `2 × (1/fps)` seconds, flush the buffer before appending.
- Loads checkpoint by inspecting the final conv layer weight shape to determine
  `out_channels` — never hardcode channel index.
- Resizes input frames to `(tracknet_expected_h, tracknet_expected_w)` before inference.
- Returns `{"shuttle": (timestamp, x, y, w, h)}` or `{}` if confidence < threshold.
- Shuttle bbox filter: reject detections where white-pixel ratio in bbox <
  `mog2_foreground_thresh_shuttle` (only applied after warmup period).

### `models/player_yolo.py` — `PlayerDetector`

- Uses `model.track(frame, persist=True)` (ByteTrack, not `predict`) for stable IDs.
- Classes filtered to `person` (COCO ID 0) only.
- After `mog2_warmup_frames`, apply MOG2 foreground filter: compute foreground
  coverage fraction within each YOLO bbox. Reject if below
  `mog2_foreground_thresh_player`.
- Returns `[{"id": int, "box": [x1,y1,x2,y2], "feet": (cx, y2)}]` per frame.
- MOG2 instance maintained internally; call `update_mog2(frame)` before `detect()`.

### `core/homography.py` — `CourtMapper`

- `calibrate(image_corners)`: takes 6 pixel points (4 court corners + 2 midline
  points) and computes the homography matrix. Court corners order:
  Bottom-Left, Bottom-Right, Top-Right, Top-Left, Midline-Bottom, Midline-Top.
- `transform_point(pixel_xy)`: returns real-world (cm) coordinate or `None`.
- `get_player_feet(box)`: returns bottom-center of bbox as pixel coordinate.
- Never opens a GUI window. Court corners come from `data/input/court_points.json`.

### `core/hit_detector.py` — `HitDetector`

- Maintains a deque of `(timestamp, x, y)` shuttle positions, maxlen = `hit_trajectory_n`.
- `update(timestamp, shuttle_pos, player_feet_real)` → `(is_hit: bool, player_id: int | None)`.
- Hit detection pipeline:
  1. RANSAC quadratic fit over last N positions (10 iterations, min 3 inliers).
  2. Predict next position from fitted curve.
  3. If prediction error > `hit_prediction_error_threshold`, declare hit candidate.
  4. Gate: shuttle must be within `hit_proximity_cm` of a player's feet (real-world cm).
  5. Cooldown: reject if within `hit_cooldown_s` of last confirmed hit.
- Exposes `set_kalman_model(model)` no-op stub for future Kalman swap-in.
- On confirmed hit: clear trajectory buffer, return `(True, player_id)`.

### `core/game_state.py` — `GameState`

Direct port of `slayminton/core/game_state.py`. No functional changes.
Key behaviors preserved:
- Motion debounce: `motion_required_streak` consecutive motion frames to start rally.
- Large-displacement filter: reject shuttle jumps > `max_displacement_fraction` of frame.
- Stability detector: end rally after `stable_frame_threshold` frames of zero movement.
- `_record_rally_segment`: structured dict with `rally_id`, `start_time`, `end_time`, `duration_s`.
- `build_rally_status_per_frame`: consolidates short inactive gaps < `rally_min_period_s`.

### `core/analysis.py` — `Analysis`

Direct port of `slayminton/core/analysis.py`. No functional changes.
Scope limited to rally duration statistics and histogram generation.
`analyze_player_movements` and `analyze_shuttle_trajectories` remain stubs for future work.

### `models/stroke_classifier.py` — `StrokeClassifier`

- Runs **MediaPipe Pose** on the hit keyframe to extract 33 joint (x, y, visibility) tuples.
- Concatenates with `hit_trajectory_n` pre-hit + `hit_trajectory_n` post-hit shuttle positions.
- Feature vector fed into `stroke_transformer.py` sequence classifier.
- Output: `{"stroke_type": str, "confidence": float, "tactical_semantic": str, "decision_eval": str}`.
- All three FineBadminton annotation levels stored; downstream consumers can ignore
  `tactical_semantic` and `decision_eval` if not needed.
- `classify(hit_event: dict) -> dict`. Input hit event must contain `keyframe` (np.ndarray)
  and `trajectory` (list of (timestamp, x, y)).

### `utils/mog.py`

Port of `slayminton/scripts/mog.py`. Adds:
- `MOG2Manager` class: stateful wrapper holding one `cv2.BackgroundSubtractorMOG2`
  instance per video, with `apply(frame) -> mask` method.
- Parameters: `varThreshold=200`, `history=1000` (matching slayminton defaults).
- `get_foreground_ratio(frame, box) -> float`: computes white-pixel fraction inside
  a bounding box on the current mask — used by `PlayerDetector` and `TrackNetTracker`.
- `coco_mog2()` and `process_video()` batch utilities preserved for offline preprocessing.

### `utils/visualization.py`

Rendering only. No game logic. Sources:
- Court drawing: port of `draw_court_background()` from slayminton `visualizations.py`.
- Heatmap overlay: port of `precompute_player_court_heatmap()` from same.
- Box drawing: new, uses ByteTrack IDs and P1/P2 color assignments.
- Rally indicator: overlay text + colored border based on `rally_active` bool.
- `render_frame(frame, detections, rally_active, court_insert) -> np.ndarray`.
- `render_court_insert(player_feet_real_history) -> np.ndarray`.

### `prepare_run.py`

Local-only. Never imported by `main.py`.
- Opens an OpenCV window on the first frame of the input video.
- Prompts user to click 6 court points in order:
  Bottom-Left, Bottom-Right, Top-Right, Top-Left, Midline-Bottom, Midline-Top.
- Supports `z` to undo last point, `q` to quit.
- **Always saves** to `data/input/court_points.json` keyed by video stem name.
- Prints the absolute path of the saved JSON so it can be passed to OSCAR.

### `dashboard.py`

Local-only Streamlit app. Reads output JSONs, renders:
- Rally timeline scrubber.
- Per-shot stroke classification table with confidence.
- Player coverage heatmaps.
- Kinematics plots (shuttle speed over time).
- Does NOT re-run the pipeline. All data comes from `events.json`, `rally_data.json`,
  `analytics.json`.

---

## Output Schema (per run)

All artifacts written to `data/output/<video_stem>_<timestamp>/`:

```
tracking_results.json   — list of per-frame dicts:
  { frame_idx, timestamp, shuttle: {x,y,w,h} | null,
    players: [{id, box, feet_px, feet_real}], rally_active }

events.json             — list of hit events:
  { timestamp, frame_idx, player_id, stroke_type, confidence,
    tactical_semantic, decision_eval,
    trajectory_pre: [(t,x,y) ×6], trajectory_post: [(t,x,y) ×6],
    prediction_error }

rally_data.json         — list of rally segments:
  { rally_id, start_time, end_time, duration_s }

analytics.json          — summary statistics:
  { rally_count, mean_rally_duration_s, min/max_rally_duration_s,
    total_rally_duration_s, per_player_hit_counts,
    rally_duration_histogram_path }

court_points.json       — copy of the 6 court corners used for this run

annotated_video.mp4     — rendered output with boxes, court insert, rally overlay
```

SLURM stdout/stderr → `data/output/logs/slurm-<jobid>.out` / `.err`

---

## Data Sources

### Roboflow badminton dataset
- URL: https://universe.roboflow.com/badminton-rojkf/badminton-hehp8
- License: CC BY 4.0
- ~10,259 annotated frames, COCO format.
- Categories: `badminton`, `person`, `racket`, `shuttle`.
- Canonical training categories mapped to: `player` (from `person`), `shuttle`.
- Download via `utils/data_download.py` using the `roboflow` Python package.
- MOG2-masked version generated by `utils/mog.py` → `data/input/train_mog_frames/`.
- Horizontal-flip augmented version → `data/input/train_mog_reflect/`.

### FineBadminton dataset
- URL: https://ilearn-lab.github.io/MM25-FineBadminton/
- 3-level annotation hierarchy per hit:
  - **Foundational Actions**: shot type (smash, drop, clear, net, drive, lift, …)
  - **Tactical Semantics**: tactical intent of the shot
  - **Decision Evaluation**: correctness/quality of the decision
- Download via `utils/data_download.py`.
- Used exclusively for stroke classifier training (`training/train_stroke.py`).
- Store under `data/input/finebadminton/`.

---

## Training Pipelines

### TrackNet fine-tuning (`training/train_tracknet.py`)
- Input: Roboflow shuttle annotations + MOG2-augmented frames.
- Model: `TrackNet` from `models/TrackNet.py`.
- DDP: wrap with `SyncBatchNorm.convert_sync_batchnorm(model)` then
  `DistributedDataParallel`. SLURM launcher uses `torchrun --nproc_per_node=$NGPUS`.
- Loss: BCE on heatmap output vs Gaussian-blurred GT center.
- Metrics: **distance error** (px from heatmap peak to GT center) + mAP@0.5
  (treating heatmap peak as a fixed-size box). Both reported per epoch.
- Dataset split: 80/20 train/val (per slayminton convention).
- Augmentation: horizontal flip via `scripts/augment_data_reflect.py`.
- Checkpoints → `data/output/logs/<run>/checkpoints/`.

### YOLO fine-tuning (`training/train_yolo.py`)
- Input: Roboflow `person` + `shuttle` annotations, converted to YOLO format.
- Model: YOLOv8n (nano) base; fine-tune all layers.
- DDP: Ultralytics handles natively with `device="0,1"` syntax.
- Metrics: Ultralytics built-in mAP@0.5, mAP@0.5:0.95, precision, recall per class.
- Output: `weights/yolo_badminton.pt`.

### Stroke classifier training (`training/train_stroke.py`)
- Input: FineBadminton Foundational Actions labels + extracted pose + trajectory features.
- Model: `models/stroke_transformer.py` (existing stub, implement as Transformer encoder
  over the concatenated feature sequence).
- Metrics: per-class accuracy + macro F1. All 3 annotation levels stored as separate
  output heads or labels; only Foundational Actions is the primary training objective.
- Output: `weights/stroke_classifier.pt`.

---

## SLURM Rules

- All `#SBATCH -o` and `-e` directives → `data/output/logs/slurm-%j.out` / `.err`.
- `mkdir -p data/output/logs` must precede any Python invocation in the script.
- Training launchers use `torchrun --nproc_per_node=$NGPUS` (default NGPUS=1).
- `track-video` mode uses single-process Python (no DDP needed at inference).
- Pass `--config config.yaml` to all `main.py` invocations; override individual
  parameters with `--set key=value` or SLURM `--export`.
- Prefer `uv run` if available; fall back to venv `python`.

---

## Known Issues Fixed in This Rewrite

| Issue | Source | Fix |
|---|---|---|
| TrackNet extracts `heatmap_tensor[0,2,:,:]` (wrong channel) | `badminton_vision/models/shuttle_tracknet.py` | Adopt slayminton `TrackNetTracker` which detects `out_channels` from checkpoint |
| Court corner GUI never saves to JSON (`save=False`) | `slayminton/scripts/visualizations.py:551` | `prepare_run.py` always saves; `main.py` only reads |
| MOG2 false positives in first ~100 frames (cold start) | Both | 150-frame warmup before enabling MOG2 filter |
| Player ID swaps (DINOv3 + blob centroid) | `slayminton` | ByteTrack persistent IDs |
| SLURM logs written to submit directory | `slurm_train.sh`, `slurm_track.sh` | `#SBATCH -o data/output/logs/slurm-%j.out` |
| DINO multi-GPU fails silently (BatchNorm on single GPU) | `slayminton/models/dino.py` | DDP + SyncBatchNorm + torchrun (TrackNet/YOLO training) |
| Homography / kinematics / visualization all in one 900-line file | `slayminton/scripts/visualizations.py` | Split into `core/`, `utils/visualization.py` |
| No config file — constants scattered across 8+ files | Both | `config.yaml` as single source of truth |

---

## Testing

### Tier 1 — Module Smoke Tests (`tests/test_*.py`)
Run with `pytest tests/ -k "not integration"`. No GPU, no video required.

- `test_tracknet.py`: feed 3 blank (288×512×3) frames → assert return is dict with
  key `shuttle` or empty dict; assert no exception.
- `test_yolo.py`: feed one blank frame → assert return is list; assert no exception.
- `test_hit_detector.py`: construct a perfect synthetic parabola of 6 points with one
  deliberate discontinuity → assert exactly 1 hit fired; assert correct player attributed.
- `test_court_mapper.py`: feed known pixel corners → assert transformed real-world
  coordinates match expected cm values within 1cm tolerance.
- `test_game_state.py`: simulate shuttle detections at controlled timestamps →
  assert rally starts and ends at expected times.

### Tier 2 — Integration Test (`tests/test_pipeline.py`)
Run with `pytest tests/test_pipeline.py`. Requires `data/input/test_clip.mp4`.

- Runs `main.py` programmatically on `test_clip.mp4` with a pre-saved `court_points.json`.
- Asserts: all 5 output JSON files exist; `rally_data.json` contains ≥1 rally;
  `tracking_results.json` has ≥1 frame with non-null shuttle; no Python exceptions raised.

---

## Iteration Log

### Initial clean-slate design (May 2026)
- Grilled full architecture against both `badminton_vision` and `slayminton` codebases.
- Settled on: YOLOv8 + ByteTrack (players), slayminton TrackNetTracker (shuttle),
  quadratic RANSAC hit detector (N=6), Kalman interface stub, MOG2 as confidence filter
  (6% player / 5% shuttle, 150-frame warmup), Streamlit dashboard (headless pipeline),
  config.yaml, DDP training for TrackNet + YOLO, FineBadminton stroke classifier
  (MediaPipe pose + trajectory → Transformer), clean-slate rewrite of badminton_vision.
- `slayminton/` kept as read-only reference.
