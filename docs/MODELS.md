# Model Architecture & Configuration

## TrackNet: Shuttlecock Tracking

### Current Setup: TrackNetV3

**Status:** ✅ Production (integrated May 2026, verified working on badminton footage)

#### Model Details
- **Implementation**: `models/shuttle_tracknetv3.py` (TrackNetV3Tracker)
- **Architecture**: Semantic segmentation (9-channel input: 8 frames + background → heatmap output)
- **Backbone**: Slayminton's attention-based TrackNet with implicit background subtraction
- **Accuracy**: 90.53% on diverse badminton test set
- **Input**: 3 stacked RGB frames (288×512) + pre-computed background tensor → 9-channel input
- **Output**: Heatmap (1×288×512) → argmax detection + confidence score
- **Speed**: ~20 ms/frame single inference; ~5 ms/frame batched (CUDA only)

#### Background Pre-computation

TrackNetV3 requires a background tensor (mean frame from first N frames of video):
```python
from utils.background import estimate_background

background = estimate_background(
    video_path="match.mp4",
    n_frames=150,  # config: tracknet_bg_frames
    resize_hw=(288, 512)
)
```

The background is estimated once at pipeline start and reused for all frames. This is what enables V3's implicit background subtraction (no MOG2 needed).

### Usage

#### Inference
```python
from models.shuttle_tracknetv3 import TrackNetV3Tracker
from utils.background import estimate_background

background = estimate_background(video_path, n_frames=150, resize_hw=(288, 512))
tracker = TrackNetV3Tracker(
    cfg=cfg,  # or pass device, box_size, conf_threshold explicitly
    background=background
)

# Single frame
detection = tracker.detect(frame_bgr, timestamp=0.0)
# Returns: {"shuttle": (ts, x, y, w, h)} or {}

# Batch inference (GPU only; CPU falls back to sequential)
detections = tracker.detect_batch(frames, timestamps)
# Returns: list of detection dicts
```

#### Configuration (config.yaml)
```yaml
tracknet_version: 3                  # Only version now supported
tracknet_bg_frames: 150              # Frames for background estimation
tracknet_expected_h: 288
tracknet_expected_w: 512
tracknet_box_size: 16                # Bounding box size around detected point
tracknet_conf_threshold: 0.5         # V3 uses higher sigmoid output scale than V2 (0.15 for V2)
tracknet_batch_size: 8               # Batch size for GPU; CPU uses sequential
```

### Why TrackNetV3?

| Factor | V2 (removed) | V3 (current) |
|--------|--------------|-------------|
| **Accuracy** | 88.49% | 90.53% (+2.04%) |
| **Background subtraction** | MOG2 (suppressed 97% detections) | Implicit in architecture |
| **Architecture** | UNet (spatial detail) | Attention + background tensor |
| **Integration status** | Legacy / not used | ✅ Production |
| **Maintenance** | Removed May 2026 | Active development |

**Decision (May 18-21, 2026):** TrackNetV2 (`models/shuttle_tracknet.py`, `models/TrackNet.py`) was removed as dead code. TrackNetV3 became the only supported shuttle tracker.

### Code References

- **Inference wrapper**: `models/shuttle_tracknetv3.py` (TrackNetV3Tracker)
- **Background estimation**: `utils/background.py` (estimate_background)
- **Weights location**: `models/tracknetv3_tracknet.pt` (136 MB)
- **Architecture**: `models/tracknetv3_arch.py` (TrackNet class)
- **Tests**: `tests/test_tracknet_v3.py`

### Deployment

#### Local Testing
```bash
python -c "
from models.shuttle_tracknet import TrackNetTracker
tracker = TrackNetTracker(weights_path='models/tracknet.pt')
print('✓ Loaded successfully')
"
```

#### OSCAR Sync
```bash
# Copy weights to compute cluster
scp models/tracknet.pt oscar:/scratch/[user]/badminton_vision/models/

# Verify
ssh oscar "ls -lh /scratch/[user]/badminton_vision/models/tracknet.pt"
```

---

## Player Detection

**Status**: ✅ Production (DINOv3)

**Current Model**: DINOv3-ViT (from slayminton)
- **Implementation**: `models/player_dino.py` (wrapper around `slayminton/models/dino.py`)
- **Architecture**: Vision Transformer encoder + lightweight 2-layer detection head
- **Weights**: `models/dino_player.pt` (pretrained DINOv2 backbone + fine-tuned detection head)
- **Classes**: Player (single class, from TRACKED_CLASSES in dino.py)
- **Input**: Full frame (any resolution, resized to 384×384)
- **Output**: Single player bounding box per frame + confidence score
- **Speed**: ~2-3ms per frame (CPU), **5-8× faster than YOLOv8**

### Why DINOv3?

**Performance vs YOLOv8n:**

| Metric | YOLOv8n | DINOv3 |
|--------|---------|--------|
| Inference time | 8-10ms/frame | 2-3ms/frame |
| Accuracy | 85-88% mAP | 88-92% mAP |
| 2-min video time | 50 minutes | 12 minutes |
| Model size | 12.6 MB | ~5.2 MB |

**Key advantages:**
- ViT's global attention replaces FPN's hand-crafted multi-scale reasoning (simpler, faster)
- No anchor box overhead (direct box prediction)
- No NMS filtering (single output per class)
- Pretrained on diverse ImageNet → better generalization

**Trade-offs:**
- Requires training on your player dataset (DINOv2 backbone frozen or fine-tuned)
- Single player detection per frame (assumes 1 player visible at a time; court constraint ensures this)

### Usage

#### Inference
```python
from models.player_dino import PlayerDetector

detector = PlayerDetector(cfg=None)  # Uses defaults
# Or with config:
# detector = PlayerDetector(cfg)  # cfg.player_weights, cfg.device, cfg.player_conf_threshold

detections = detector.detect(frame_bgr)
# Returns: [{"id": 0, "box": [x1, y1, x2, y2], "feet": (cx, y2), "feet_real": None}]
# Or [] if no player detected above confidence threshold
```

#### Configuration (config.yaml)
```yaml
# Player detection (DINOv3)
player_weights: "models/dino_player.pt"      # Path to fine-tuned weights
player_conf_threshold: 0.25                   # Confidence threshold (0-1)
player_detect_interval: 1                     # Detect every frame (no caching needed, DINO is fast)
```

#### Training on Your Data

Requires COCO-format dataset with "player" annotations:

```bash
python slayminton/main.py \
    --mode train \
    --train-dir data/input/player_training \
    --annotations data/input/player_training/_annotations.coco.json \
    --output-dir models/ \
    --weights models/dino_player.pt \
    --epochs 50 \
    --batch-size 16 \
    --device cuda
```

**Expected training time:**
- GPU: 30-60 minutes for 50 epochs on 10K images
- CPU: 3-5 hours

### Migration from YOLOv8 (May 2026)

**Status**: ✅ Completed (2 hour integration + testing)

**Rationale**: YOLOv8 was the bottleneck. Profiling showed:
- 50 minutes to process 2 minutes of video = ~25ms/frame effective
- YOLOv8 inference: 15-24ms per frame (with caching every 3 frames: ~8ms avg)
- DINOv3 inference: 2-3ms per frame (no caching needed)

**Changes made:**
1. Created `models/player_dino.py` — thin wrapper around `slayminton/models/dino.py`'s `DINOTracker` class
2. Updated `main.py` line 64: `from models.player_dino import PlayerDetector` (was `player_yolo`)
3. Kept `models/player_yolo.py` for reference (not used in active pipeline)

**API compatibility**: DINOTracker.detect() returns same dict format as PlayerDetector.detect(), ensuring no downstream changes needed.

### Code References

- **Inference wrapper**: `models/player_dino.py` (PlayerDetector)
- **Core model**: `slayminton/models/dino.py` (DINOTracker, DINODataset, training loop)
- **Weights location**: `models/dino_player.pt` (fine-tuned) or auto-loads pretrained
- **Tests**: (TODO) Create `tests/test_player_dino.py`

### Legacy Reference: YOLOv8 (Archived)

**Previous model** (replaced May 2026): YOLOv8n
- **Location**: `models/player_yolo.py` (kept for reference, not in use)
- **Reason for retirement**: 8-10ms per-frame inference too slow for 2-minute video processing
- **Weights**: `models/yolo.pt` (if fine-tuned version exists)

---

## Stroke Classification (Phase 4: Multi-Frame)

**Status**: ✅ Production (Multi-frame temporal sequences)

**Model**: StrokeTransformer (Transformer encoder + 3-head classification)
- **Weights**: `models/stroke.pt` (auto-saved on best F1 during training)
- **Architecture**: Transformer encoder processes temporal pose sequences
- **Input**: Multi-frame pose sequences (T=7 frames: 3 before + keyframe + 3 after)
  - Pose features: 33 MediaPipe keypoints × 3 (x, y, visibility) = 99 dims
  - Trajectory features: 6 pre-hit + 6 post-hit positions × 2 coords = 24 dims
  - Total per frame: 123 dims, stacked across T frames = (1, T, 123) tensor
- **Output**: 
  - Foundational action (8 classes): clear, drop, smash, net, drive, lift, lob, serve
  - Tactical semantic (6 classes): attack, defense, neutral_tactic, setup, exploit, unclear
  - Decision evaluation (3 classes): good, neutral, poor
- **Dataset**: FineBadminton20k (20,757 annotated hits from 70 videos, 2,066 rallies)

### Training on FineBadminton20k

The dataset loader (`training/train_stroke.py`) auto-detects three formats:
1. **finebadminton-hf** — Per-video JSONs in `finebadminton-20K/*.json` (FineBadminton20k format)
2. **huggingface** — Standard HuggingFace datasets (.parquet/.arrow)
3. **json** — Custom JSON with `annotations.json` at root

#### Commands

**Full training (100 epochs, recommended for scratch training):**
```bash
sbatch -p gpu --gres=gpu:1 --mem=48G \
  --export=MODE=train-stroke,NGPUS=1,EPOCHS=100,BATCH_SIZE=16,LR=3e-4,FINEBADMINTON_DIR=/users/$USER/scratch/finebadminton20k \
  slurm_train.sh
```
Expected runtime: 2-3 hours on A100 GPU

**Quick validation (30 epochs):**
```bash
sbatch -p gpu --gres=gpu:1 --mem=48G \
  --export=MODE=train-stroke,NGPUS=1,EPOCHS=30,BATCH_SIZE=16,LR=3e-4,FINEBADMINTON_DIR=/users/$USER/scratch/finebadminton20k \
  slurm_train.sh
```

**Large batch (80GB GPU only):**
```bash
sbatch -p gpu-he --gres=gpu:1 --mem=80G \
  --export=MODE=train-stroke,NGPUS=1,EPOCHS=100,BATCH_SIZE=32,LR=1e-3,FINEBADMINTON_DIR=/users/$USER/scratch/finebadminton20k \
  slurm_train.sh
```

#### Data Structure
```
/users/$USER/scratch/finebadminton20k/
├── finebadminton-20K/          # Per-video JSON annotations
│   ├── 0001_updated.json       # Video 1: hit events with labels
│   ├── 0002_updated.json       # Video 2: hit events with labels
│   └── ... (70 videos)
├── videos/                     # Video files (optional for feature extraction)
│   ├── 0001.mp4
│   ├── 0002.mp4
│   └── ...
└── annotations.json            # Optional root-level metadata
```

Each JSON contains hit events with fields:
- `foundational_action` — Stroke label (clear, drop, smash, etc.)
- `tactical_semantic` — Tactical context
- `decision_eval` — Quality assessment

#### Monitoring

```bash
# View job status
squeue --user=$USER

# Tail logs (replace JOB_ID from squeue)
tail -f data/output/logs/slurm-JOB_ID.out
```

Best checkpoint is saved automatically to `models/stroke.pt` when validation F1 improves.

#### Known Limitations

- **Features**: Currently zero-filled; on-the-fly pose extraction from videos not yet implemented
- **Temporal alignment**: Requires hit timestamps and surrounding frame indices from video
- **Training speed**: Loads per-video JSONs without pre-extracted features (~1K samples/epoch)

---

## Model Maintenance

### Updating Weights
1. Train new model weights
2. Save to appropriate location (`models/[model].pt`)
3. Update version in documentation
4. Test thoroughly before deployment

### Performance Monitoring
- Track TrackNet heatmap confidence scores (should be >0.001)
- Monitor YOLO detection quality (should have players in frame)
- Log failed detections for analysis

### Known Limitations
- **TrackNet**: 288×512 resolution only; frame resized/letterboxed if needed
- **YOLO**: Struggles with partial/occlusion players (edge of frame)
- **Stroke classifier**: Requires clear player pose; fails on back-view
