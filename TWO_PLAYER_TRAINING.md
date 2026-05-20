# Two-Player DINO Training: Complete Setup

## What Changed

### 1. TrackNet Offset Fix ✓
**File**: `models/shuttle_tracknetv3.py`

**Problem**: TrackNet heatmap output was at reduced resolution (e.g., 64×36 from 512×288 input), but code used input dimensions to compute coordinates, causing constant left+up offset.

**Solution**: 
- Compute actual heatmap dimensions
- Calculate upscaling factors: `scale_x = input_w / heatmap_w`, `scale_y = input_h / heatmap_h`
- Upscale argmax position before using as image coordinates

**Result**: Shuttle detection will now be centered on actual shuttle position instead of offset.

---

### 2. DINO Two-Player Architecture ✓
**File**: `models/player_dino.py`

**Changes**:

#### 2.1 Output Configuration
- `TRACKED_CLASSES = ("player_1", "player_2")` (was: `("player",)`)
- `detector_head` now outputs 10 values instead of 5:
  - Player 1: `[conf, cx, cy, w, h]` (5 values)
  - Player 2: `[conf, cx, cy, w, h]` (5 values)

#### 2.2 Detection API
- `detect()` returns `{"players": [(ts, x, y, w, h), (ts, x, y, w, h)]}` instead of `{"player": ...}`
- Returns players sorted by y-coordinate (top player first, then bottom player)
- Both players returned if both exceed confidence threshold; otherwise `None`

#### 2.3 Dataset Annotation Parsing
- `_pick_representative_boxes()` now:
  - Collects all "person" bounding boxes per image
  - Sorts by y-coordinate (top to bottom)
  - Assigns first two to `player_1` and `player_2`
  - Handles images with 1, 2, or 3+ people gracefully

#### 2.4 YOLO-Compatible Output
- `detect_yolo_compat()` returns 2-element list:
  ```python
  [
    {"id": 0, "box": [x1, y1, x2, y2], "feet": (cx, y2), "feet_real": None},
    {"id": 1, "box": [x1, y1, x2, y2], "feet": (cx, y2), "feet_real": None}
  ]
  ```

---

## Dataset Verification

Your dataset has **excellent coverage** for two-player training:
- **Total images**: 10,259
- **Exactly 2 people**: 9,911 (96.6%)
- **1 person**: 196 images (outliers, will be handled)
- **3+ people**: 116 images (will use top 2)

✓ **Ready for training immediately**

---

## Pre-Submission Validation

Before submitting to SLURM, validate locally:

```bash
# Test model architecture and dataset loading
python3 test_2player_arch.py --dataset-dir data/player_dataset
```

This checks:
- ✓ Model creates with two-player architecture (10 outputs)
- ✓ Dataset loads and parses 2 players per image correctly
- ✓ Forward pass works end-to-end
- ✓ Inference returns properly formatted two-player detections

---

## SLURM Training Submission

### Submit Training Job

```bash
sbatch slurm_train_2player.sh
```

**What it does**:
- Allocates 1 GPU, 8 CPU cores, 32GB RAM for 6 hours
- Loads Python 3.11 + PyTorch 2.0 + CUDA 11.8
- Trains DINOTracker with:
  - Two detection heads (player_1, player_2)
  - LoRA fine-tuning (8% of parameters trainable)
  - Multi-crop augmentation (2 global + 6 local crops)
  - Cosine annealing learning rate schedule
  - 80/20 train/val split
- **Output**: `data/output/dino_player_2player.pt`

### Monitor Training

```bash
# Check job status
squeue -u zshen38

# Watch logs in real-time
tail -f logs/train_2player_*.log

# Or slurm output
cat logs/train_2player_*.log
```

### Training Duration
- **Estimated**: 2-3 hours (with LoRA on 1 GPU)
- **Full epochs**: 75
- **Batch size**: 16
- **Dataset**: ~8200 training samples

---

## Post-Training Usage

After training completes, use in inference:

```python
from models.player_dino import DINOTracker

model = DINOTracker(
    weights_path="data/output/dino_player_2player.pt",
    device="cuda"
)

# Detect two players
result = model.detect(frame)  # Returns {"players": [(ts, x, y, w, h), ...]}
players = result["players"]

for i, (ts, x, y, w, h) in enumerate(players):
    print(f"Player {i}: ({x:.0f}, {y:.0f}) size {w:.0f}×{h:.0f}")

# Or YOLO-compatible
detections = model.detect_yolo_compat(frame)  # Returns list of dicts
```

---

## Architecture Summary

```
Input Frame (1280×720 typical)
    ↓
DINOTracker.detect()
    ├─ BGR→RGB conversion
    ├─ Preprocess (resize to 384×384, normalize)
    ├─ ViT-S/14 encoder (DINOv2 backbone)
    │  └─ Output: 384-dim embeddings (cls token)
    ├─ Two detection heads (FC layers):
    │  ├─ Head 1 → [conf₁, cx₁, cy₁, w₁, h₁]
    │  └─ Head 2 → [conf₂, cx₂, cy₂, w₂, h₂]
    ├─ Post-processing:
    │  ├─ Normalize coords to image space
    │  ├─ Sort by y-coordinate
    │  └─ Filter by confidence threshold (0.25)
    └─ Output: {"players": [(ts, x, y, w, h), (ts, x, y, w, h)]}
```

---

## Key Differences from Single-Player

| Aspect | Single-Player | Two-Player |
|--------|---------------|------------|
| Detection heads | 1 | 2 |
| Output size | 5 values/frame | 10 values/frame |
| Players per image | 1 (best) | 2 (top + bottom) |
| Confidence filtering | Per-player | Per-player |
| Coordinate ordering | N/A | By y-coordinate (top first) |
| Training time | ~1.5h | ~2-3h (LoRA) |

---

## Troubleshooting

### Test Fails: "Expected output shape (B, 10)"
**Cause**: Old checkpoint loaded with 5-output head  
**Fix**: Remove checkpoint or explicitly set `weights_path=None`:
```python
model = DINOTracker(weights_path=None, device=device)
```

### Training crashes: "CUDA out of memory"
**Solution**: Reduce batch size in `slurm_train_2player.sh`:
```python
trained_model, history = train_dino(
    ...
    batch_size=8,  # Reduced from 16
    ...
)
```

### Inference returns single player instead of two
**Cause**: Model needs more training, both players below confidence threshold  
**Fix**: Lower threshold or train longer:
```python
result = model.detect(frame, min_confidence=0.15)
```

---

## Next Steps

1. **Validate locally**:
   ```bash
   python3 test_2player_arch.py
   ```

2. **Submit training**:
   ```bash
   sbatch slurm_train_2player.sh
   ```

3. **Monitor completion**:
   ```bash
   tail -f logs/train_2player_*.log
   ```

4. **After completion** (~3 hours):
   - Check `data/output/dino_player_2player.pt` exists
   - Load in main.py with new checkpoint path
   - Re-run badminton tracking with two-player detection

---

## Files Modified

- `models/shuttle_tracknetv3.py` — TrackNet heatmap upscaling fix
- `models/player_dino.py` — Two-player architecture + dataset parsing
- `slurm_train_2player.sh` — **NEW** — Ready-to-submit training job
- `test_2player_arch.py` — **NEW** — Validation script
- `TWO_PLAYER_TRAINING.md` — **NEW** — This document
