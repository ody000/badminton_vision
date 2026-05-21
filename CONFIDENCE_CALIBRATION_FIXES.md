# DINO Confidence Calibration Fixes (May 21, 2026)

## Problem Identified After First Retrain
After retraining with area-based sorting, diagnostic showed confidence values were **WORSE**:
- OLD checkpoint: Player_1 mean=0.9965, std=0.0004
- NEW checkpoint (first retrain): Player_1 mean=0.9981, std=0.0002 ← **MORE locked!**

**Root Cause:** The model was still outputting very large logits (5-10), which when passed through sigmoid become 0.99999.

## Root Cause Analysis

The 0.99 confidence lockout occurs because:

1. **Training targets are all 1.0** - All images have 2 players, so conf_target=1.0 for both
2. **BCE-with-logits has no inherent regularization** - Loss decreases as logits grow larger
3. **Model learns unbounded logits** - To minimize loss, outputs logits of magnitude 5-10
4. **Sigmoid saturates at large logits** - sigmoid(5)=0.9933, sigmoid(10)≈0.99999
5. **Result:** Model appears "locked" at 0.99 confidence

## Fixes Applied (May 21, 2026)

### FIX 1: Label Smoothing (Training)
**File:** `models/player_dino.py` line 835-837

**What:** Change training targets from hard labels (1.0/0.0) to soft labels (0.95/0.05)
```python
LABEL_SMOOTH = 0.05
conf_target[conf_target > 0.5] = 1.0 - LABEL_SMOOTH  # Positive: 1.0 → 0.95
conf_target[conf_target < 0.5] = LABEL_SMOOTH        # Negative: 0.0 → 0.05
```

**Why:** Prevents model from learning to output unbounded logits. With smooth targets, loss increases if logits become too extreme, discouraging excessive magnitude.

**Effect:**
- Old: Logits ≈ ±10 → sigmoid(±10) ≈ 0.99999 → apparent lockout
- New: Logits ≈ ±2.5 → sigmoid(±2.5) ≈ 0.92/0.08 → diverse values

### FIX 2: Logit Regularization (Training)
**File:** `models/player_dino.py` line 845-847

**What:** Add L1 penalty on logit magnitude to loss function
```python
logit_reg = 0.01 * (conf_logits.abs().mean())
loss = conf_loss + BOX_LOSS_WEIGHT * box_loss + logit_reg
```

**Why:** Explicitly penalizes large-magnitude logits, preventing unbounded growth during training.

**Effect:** Keeps logits compact, preventing saturation of sigmoid

### FIX 3: Logit Clipping at Inference (Deployment)
**File:** `models/player_dino.py` line 172-186

**What:** Clip confidence logits to [-3, 3] before applying sigmoid in forward_detect()
```python
raw_conf_clipped = torch.clamp(raw[..., :1], min=-3.0, max=3.0)
conf = torch.sigmoid(raw_conf_clipped)
```

**Why:** Safety measure at deployment. Even if training produces large logits, clipping ensures sigmoid output stays in [0.047, 0.953] range.

**Effect:** Provides hard guarantee on confidence value range regardless of training outcome

## Expected Behavior After Retraining

With all three fixes active:
1. **Training**: Model learns bounded logits (~2-3 range)
2. **Inference**: Logits clipped to [-3, 3]
3. **Output**: Confidence values spread across [0.1, 0.9] range instead of locked at 0.99

**Expected diagnostic results:**
```
Player_1: min=0.15, max=0.88, mean=0.45, std=0.15  ← DIVERSE!
Player_2: min=0.12, max=0.91, mean=0.48, std=0.17  ← DIVERSE!
```

## Why Previous Fixes Weren't Enough

- **Area-based sorting** (FIX 1 on May 20): Fixed label flipping but didn't address logit saturation
- **Version metadata** (FIX 2 on May 20): Documentation/safety but didn't fix loss function
- **BCE-with-logits** (existing): Numerically stable but doesn't prevent unbounded logits

These new fixes (label smoothing + logit regularization + clipping) address the fundamental mechanism causing saturation.

## Implementation Details

### Label Smoothing Parameters
- Positive class (player exists): 0.95 (was 1.0)
- Negative class (no player): 0.05 (was 0.0)
- Strength: 5% smoothing (tuneable if needed)

### Logit Regularization
- Coefficient: 0.01 (can be adjusted)
- Applied to: Mean absolute value of confidence logits
- Added directly to: Total training loss

### Logit Clipping at Inference
- Min value: -3.0 (sigmoid ≈ 0.047)
- Max value: 3.0 (sigmoid ≈ 0.953)
- Applied only to: Confidence logits (not box coordinates)
- When applied: In forward_detect() method

## Files Modified

1. **models/player_dino.py**
   - Line 172-186: Added logit clipping in forward_detect()
   - Line 835-837: Added label smoothing in training
   - Line 845-847: Added logit regularization to loss
   - Line 720-760: Updated train_dino() docstring with all fixes

## Next Steps

### 1. Retrain DINO with New Fixes
```bash
sbatch --gres=gpu:1 --mem=32G --export=MODE=train-dino-2player,EPOCHS=50 slurm_train.sh
```

**Important:** Use more epochs (50 instead of 30) since label smoothing requires additional training to converge.

### 2. Deploy New Checkpoint
```bash
cp data/output/dino_player_2player.pt models/dino_player.pt
```

### 3. Run Diagnostic Validation
```bash
sbatch slurm_dino_diag.sh
```

**Expected output:**
- Confidence values should be DIVERSE (not 0.99 lockout)
- Both players should have varied confidence across frames
- Box predictions should follow actual players

### 4. Test Tracking Pipeline
```bash
sbatch --export=VIDEO_PATH=data/input/match_clip5.mp4 slurm_track.sh
```

## Verification

All code changes verified:
```
[VERIFY] Code imports successfully ✅
[VERIFY] Model creates successfully ✅
[VERIFY] All fixes verified in code! ✅
```

## Technical Notes

- Label smoothing is standard technique in computer vision for preventing overconfidence
- Logit regularization prevents pathological optimization (unbounded logits)
- Logit clipping is deployment safety measure (guarantees output range)
- Combination of three approaches provides multi-layered defense against confidence saturation

## Timeline

- **May 20, ~11:30:** Discovered 0.99 confidence lockout in diagnostic
- **May 20, ~23:30:** Identified area-based sorting as partial fix, added version metadata
- **May 21, 00:01:** First retrain completed (area-based sorting only)
- **May 21, 00:10:** Diagnostic showed confidence WORSE (0.9981 instead of 0.9965)
- **May 21, 00:30-01:00:** Identified true root cause (unbounded logits during training)
- **May 21, 01:00:** Applied three-layered fix (smoothing + regularization + clipping)
- **NOW:** Ready for second retrain with full fixes
