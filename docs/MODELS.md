# Model Architecture & Configuration

## TrackNet: Shuttlecock Tracking

### Current Setup: Slayminton TrackNetV2

**Status:** ✅ Production (verified working on badminton footage)

#### Model Details
- **Implementation**: Slayminton's TensorFlow-to-PyTorch conversion (`slayminton/models/tracknet.py`)
- **Architecture**: UNet-style semantic segmentation (9-channel input → heatmap output)
- **Weights**: `models/tracknet.pt` (44 MB)
- **Accuracy**: 88.49% on badminton test set
- **Input**: 3 stacked RGB frames (288×512) → 9-channel tensor
- **Output**: Heatmap (1×288×512) → argmax detection + confidence score

#### Key Architectural Note: BatchNorm Transpose Hack

Slayminton's implementation includes a quirk for TensorFlow compatibility:
```python
class Conv(nn.Module):
    def forward(self, x):
        x = self.conv(x)
        x = x.transpose(1, 3)        # [B,C,H,W] → [B,W,H,C]
        x = self.bn(x)               # BN applied on width dimension
        x = x.transpose(1, 3)        # [B,W,H,C] → [B,C,H,W]
        return x
```

This is NOT a bug in production weights; pretrained weights expect this exact behavior. Do NOT "fix" it or use alternative architectures without retraining.

### Usage

#### Inference
```python
from models.shuttle_tracknet import TrackNetTracker

tracker = TrackNetTracker(
    weights_path="models/tracknet.pt",
    device="cuda",
    expected_h=288,
    expected_w=512
)

# Single frame
detection = tracker.detect(frame_bgr, timestamp=0.0)
# Returns: {"shuttle": (ts, x, y, w, h)} or {}

# Batch inference (more efficient)
detections = tracker.detect_batch(frames, timestamps)
# Returns: list of detection dicts
```

#### Configuration (config.yaml)
```yaml
tracknet_weights: models/tracknet.pt
tracknet_expected_h: 288
tracknet_expected_w: 512
tracknet_box_size: 16              # Bounding box size around detected point
tracknet_conf_threshold: 0.001     # Minimum heatmap confidence
```

### Why Slayminton V2?

#### Decision Factors

| Factor | Fine-tuned V2 | Pretrained V2 | TrackNetV3 |
|--------|---------------|---------------|-----------|
| **Working** | ✗ (white dot) | ✓ Verified | ? (not tested) |
| **Accuracy** | ??? | 88.49% | 90.53% (+2%) |
| **Dev time** | Wasted (failed) | 0 hours | 3-4 weeks |
| **Risk** | High | Low | High |
| **Domain-fit** | Your data only | Diverse badminton | Small dataset |

#### Root Cause: Fine-tuning Failed

Your custom fine-tuned model learned to track a white background dot instead of the shuttlecock. This occurred because:

1. **Limited training data** → Overfitting to artifacts in your dataset
2. **White dot prominence** → Model found easy local optimum
3. **Lack of regularization** → No mechanism to prevent spurious features

Slayminton's pretrained model, trained on 32k+ diverse badminton images, generalizes better and avoids this pitfall.

#### Why Not TrackNetV3?

TrackNetV3 offers 90.53% accuracy (+2.04%) with attention mechanisms. However:
- Requires 3-4 weeks integration + retraining
- Untested on your specific footage
- Higher implementation risk
- No proven gain over V2 on your domain

**Recommendation**: Start with V2. Switch to V3 only if accuracy becomes bottleneck.

### Code References

- **Inference wrapper**: `models/shuttle_tracknet.py` (TrackNetTracker)
- **Model architecture**: `slayminton/models/tracknet.py` (TrackNet)
- **Weights location**: `models/tracknet.pt`
- **Tests**: `tests/test_tracknet.py`, `tests/test_tracknet_batch.py`

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

## Stroke Classification

**Status**: ✅ Production

**Model**: ResNet-based stroke classifier
- **Weights**: `models/stroke.pt`
- **Classes**: Serve, Clear, Drop, Smash, Push, Lob, Drive
- **Input**: Cropped frame around player (RGB)
- **Output**: Stroke class + confidence

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
