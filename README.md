# badminton_vision

Automated video analysis pipeline for badminton match replay. Detects players, shuttlecock, hit events, and strokes to produce structured game analytics.

## What it does

Badminton match analysis using computer vision to track shuttlecocks, detect players, identify hit events, segment rallies, and (optionally) classify strokes. Generates JSON artifacts (tracking data, hit events, rally segments, statistics) and visualizations (annotated video, player heatmaps, analytics dashboard). Runs on local machines or OSCAR/SLURM clusters.

**Core workflow:**
- `main.py`: CLI pipeline — frame extraction, detection, hit identification, rally analysis, JSON output.
- `tools/prepare_run.py`: Court calibration GUI (click 6 points on first frame).
- `tools/viewer.py`: Interactive visualization with overlay playback (macOS/Linux).
- `tools/dashboard.py`: Streamlit analytics dashboard for exploring results.
- `config.yaml`: Central configuration (all parameters tunable).

### Output Schema (per run)

```
data/output/<video_stem>_<timestamp>/
├── tracking_results.json     # Per-frame: positions, IDs, rally status
├── events.json               # Hit events with stroke classification
├── rally_data.json           # Rally segments: start, end, duration
├── analytics.json            # Aggregate stats: rally counts, heatmaps
├── court_points.json         # 6-point court calibration (copy of input)
├── heatmap.png               # Player footwork heatmap (JET colormap)
├── annotated_video.mp4       # Rendered output (optional, slow)
└── logs/slurm-<jobid>.out    # SLURM stdout/stderr
```

## Models Used

### 1. TrackNetV3 (Shuttle Detection)
- **Type:** Semantic segmentation (UNet-style attention)
- **Input:** 3 stacked RGB frames (288×512) + background tensor
- **Output:** Heatmap → argmax detection + confidence
- **Accuracy:** 90.53% on diverse badminton footage
- **Speed:** 5 ms/frame (batched GPU), 20 ms/frame (single)
- **Key feature:** Implicit background subtraction (no MOG2 needed)
- **Weights:** `models/tracknetv3_tracknet.pt`, `models/tracknetv3_inpaintnet.pt`
- **Reference:** [TrackNetV3] (https://github.com/qaz812345/TrackNetV3)

### 2. YOLOv8 (Player Detection)
- **Type:** CNN-based object detection (fine-tuned)
- **Architecture:** YOLOv8n backbone
- **Input:** Raw frame (any resolution, resized to 640×640 internally)
- **Output:** Bounding boxes + person class confidence
- **Speed:** 5 ms/frame GPU
- **Tracking:** Centroid-based persistent ID matching (no Kalman filter)
- **Key feature:** 2-player badminton domain fine-tuning
- **Weights:** `models/yolo.pt`
- **Reference:** [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)

### 3. StrokeTransformer (Stroke Classification) ⚠️ *Incomplete: requires training*
- **Type:** Transformer encoder + multi-head classification
- **Input:** MediaPipe Pose (33 joints) + shuttle pre/post trajectory (5 frames each)
- **Output:** 100+ stroke classes with confidence scores
- **Accuracy:** 16% macro F1 on FineBadminton dataset (50 epochs, 10K images) — *weak; useful for "drive"/"clear"/"smash" only*
- **Speed:** 5–10 ms/event (MediaPipe pose detection is bottleneck)
- **Status:** Optional in pipeline (`stroke_classify_enabled: false` in config); disabled by default
- **Weights:** `models/stroke.pt`
- **Training data:** [FineBadminton20k dataset] (https://huggingface.co/datasets/iLearn-Lab/Finebadminton-20K)


## Quick Start

### Prerequisites

```bash
# Python 3.11+
pip install -r requirements.txt
# Or use the conda environment:
conda activate badminton_vision
```

### 1. Court Calibration (Local)

```bash
python tools/prepare_run.py --video data/input/match.mp4
```

- Click 6 points on first frame: 4 corners + 2 midline
- Saved to `data/input/court_points.json`
- Press `z` to undo, `q` to quit

### 2. Run Pipeline

**Local machine (CPU/GPU):**
```bash
# CPU (slow, but works)
python main.py --video data/input/match.mp4

# GPU (recommended)
python main.py --video data/input/match.mp4 --set device=cuda

# With annotated output (slow re-encoding) and custom parameters
python main.py --video data/input/match.mp4 \
  --set device=cuda tracknet_batch_size=16 player_detect_interval=3 --annotate
```

**OSCAR/SLURM cluster:**
```bash
# Prepare environment
conda activate badminton_vision
mkdir -p data/output/logs

# Submit inference job to GPU queue
sbatch slurm_track.sh --video match.mp4 --device cuda

# (Optional) submit training job first
sbatch slurm_train.sh
```

### 3. View Results Locally

```bash
# DearPyGui interactive viewer (macOS/Linux only)
python tools/viewer.py --run data/output/<run_dir> --video data/input/match.mp4

# Streamlit analytics dashboard
python tools/dashboard.py --run data/output/<run_dir>
```

## Pipeline Overview

```
Video Input
    ↓
[Court Calibration]  ← User clicks 6 corners (local) or loads JSON (OSCAR)
    ↓
[Background Estimation]  ← Mean of first 150 frames (TrackNetV3)
    ↓
[Frame Decoding]  ← Read frames in sequence
    ↓
┌─────────────────────────────────────────┐
│ Per-frame Processing (parallel)         │
├─────────────────────────────────────────┤
│ • TrackNetV3: Shuttle detection         │
│ • YOLOv8: Player detection              │
│ • PlayerContext: Persistent ID mapping  │
│ • Homography: Pixel → real-world coords │
│ • HitDetector: RANSAC trajectory fit    │
│ • GameState: Rally start/end logic      │
└─────────────────────────────────────────┘
    ↓
[Post-processing]
    ├── Hit event refinement (stroke classification optional)
    ├── Rally merging (short gaps < 0.5s)
    ├── Analytics aggregation (counts, stats)
    └── Heatmap generation (footwork visualization)
    ↓
JSON Artifacts
├── tracking_results.json
├── events.json
├── rally_data.json
└── analytics.json
```

## Configuration

All parameters live in `config.yaml` and can be overridden at runtime:

### Key Sections

| Section | Purpose |
|---------|---------|
| `tracknet_*` | TrackNetV3 model, batch size, confidence threshold |
| `player_*` | YOLOv8 model, detection interval, confidence threshold |
| `hit_*` | RANSAC fitting, proximity gating, cooldown |
| `rally_*` | Motion streak, inactive timeout, minimum duration |
| `stroke_*` | Stroke classifier enable flag, model path |

### Example: Speed vs. Accuracy Trade-off

```yaml
# Fast (2× throughput)
player_detect_interval: 3         # Skip YOLO every 3 frames
tracknet_batch_size: 16           # Larger GPU batch
stroke_classify_enabled: false    # Disable expensive stroke classification

# Accurate (slower)
player_detect_interval: 1         # YOLO every frame
tracknet_batch_size: 8
stroke_classify_enabled: true     # Enable stroke classification
```

## Advanced: Custom Training/Fine-tuning

### 1. Shuttle Tracker (TrackNetV3)

```bash
# Not recommended; weights are already optimal (pretrained)
# See training/train_tracknet.py for reference
```

### 2. Player Detector (YOLOv8)

```bash
python training/train_yolo.py \
  --data data/coco-badminton.yaml \
  --epochs 20 \
  --batch-size 16 \
  --device cuda
```

**Requirements:** 5K–10K COCO-format annotations (player class only)

### 3. Stroke Classifier (StrokeTransformer)

```bash
python training/train_stroke.py \
  --dataset FineBadminton \
  --epochs 50 \
  --batch-size 8 \
  --device cuda
```

**Requirements:** Pose annotations + stroke labels from FineBadminton

## Limitations & Future Work

**Current constraints:** Court-specific setup (requires 6-point calibration per angle). Two-player assumption (crowd in background causes false positives). Stroke classification weak (16% macro F1; useful for "drive"/"clear"/"smash" only). Viewer (DearPyGui) macOS/Linux only. Background estimation assumes low-motion first 150 frames. Single-process inference (no multi-GPU).

**Future improvements:** Increase stroke classifier accuracy (training data / architecture). Kalman-filter player tracking (ID stability after occlusions). Real-time inference mode (streaming). Multi-court video support. Reduce MediaPipe CPU bottleneck. 3D court reconstruction (court-agnostic calibration). End-to-end training (joint optimization).


## Project Structure

```
.
├── main.py                    # Headless pipeline entry point
├── config.yaml                # Central configuration
├── core/                      # Game logic (tracking, hit detection, rallies)
├── models/                    # Model wrappers (TrackNet, YOLO, Stroke)
├── training/                  # Training scripts (DDP-ready for OSCAR)
├── tools/                     # Local utilities (GUI, viewer, dashboard)
├── utils/                     # Helpers (video I/O, homography, visualization)
├── tests/                     # Unit + integration tests
├── docs/                      # Architecture (AGENTS.md), issues (ISSUES.md), models (MODELS.md)
├── data/                      # Input/output directories
└── slurm_*.sh                 # SLURM job submission scripts
```
