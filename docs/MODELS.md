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

## YOLO: Player Detection

**Status**: ✅ Production

**Model**: YOLOv8n (nano variant)
- **Weights**: `models/yolo.pt`
- **Classes**: Player (single class)
- **Input**: Full frame (any resolution, auto-resized to 640×480)
- **Output**: Bounding boxes + confidence scores

### Usage
```python
from models.player_yolo import PlayerDetector

detector = PlayerDetector(
    weights_path="models/yolo.pt",
    device="cuda"
)

detections = detector.detect(frame_bgr)
# Returns: list of {"player": (x, y, w, h, conf)}
```

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
