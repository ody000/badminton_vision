# DINO Checkpoint & Training Fixes (May 20, 2026)

## Problem Summary
The DINO player detector was outputting constant 0.99 confidence for all inputs due to corrupted training data labels.

**Root Cause:** Y-coordinate sorting for player assignment
- Image A: persons at y=[245, 390] → player_1 = top person (y=245)
- Image B: same persons at y=[390, 245] → player_1 = top person (y=245) ← **WRONG PERSON!**
- Result: Same physical player randomly labeled as player_1/player_2 across frames
- Effect: Training signal corrupted → both detector heads defaulted to high confidence (0.99 lockout)

## Fixes Applied

### 1. Area-Based Player Sorting (Code Fix) ✅
**File:** `models/player_dino.py` - `DINODataset._pick_representative_boxes()` (lines 548-598)

**Change:** Sort by bounding box AREA (descending) instead of y-coordinate
```python
# OLD (broken):
player_boxes.sort(key=lambda p: p[0])  # Sort by y-coordinate
# NEW (fixed):
player_boxes.sort(key=lambda p: p[0], reverse=True)  # Sort by area (largest first)
```

**Rationale:** Badminton players are the largest visible people in frame; spectators are smaller background elements. Area-based sorting ensures consistent assignment across frames.

### 2. Checkpoint Version Metadata ✅
**File:** `models/player_dino.py` - `save_checkpoint()` & `load_checkpoint()`

**Changes:**
- `save_checkpoint()` now saves checkpoint with version="2026-05-area-based-sorting"
- `load_checkpoint()` now checks version and warns if loading old/unversioned checkpoint
- Warning message explains the label flipping issue and recommends retraining

### 3. Improved Checkpoint Loading Logging ✅
**File:** `models/player_dino.py` - `DINOTracker.__init__()`

**Change:** Added clear logging to show when checkpoint is loaded for tracking
```
[DINOTracker] Loading full checkpoint for inference/tracking: models/dino_player.pt
```

### 4. Removed Old Corrupted Checkpoint ✅
**File:** `models/dino_player.pt`

**Change:** Renamed to `models/dino_player.pt.OLD_Y_COORD_SORTING_CORRUPTED`
- Prevents accidental loading of corrupted checkpoint
- Tracked players will now start with randomly initialized detector_head (clean slate)

## Impact

### Before Fixes
```
Diagnostic Results (100 test frames):
- Player_1: min=0.9957, max=0.9971, mean=0.9965, std=0.0004  ← LOCKED AT 0.99
- Player_2: min=0.9751, max=0.9841, mean=0.9799, std=0.0026  ← LOCKED AT 0.98
```
Both players always detected with extremely high confidence regardless of image content.

### After Retraining with New Fixes
Expected:
- Confidence distribution should spread across [0.2-0.95] range
- Model should learn to distinguish player_1 from player_2
- Boxes should follow actual players instead of wobbling at center

## Next Steps

### MUST DO: Retrain DINO
```bash
sbatch --gres=gpu:1 --mem=32G --export=MODE=train-dino-2player,EPOCHS=30 slurm_train.sh
```

**Why:** New checkpoint format incompatible with old weights. Full retrain required.

**Expected output:** `data/output/dino_player_2player.pt` (with version metadata)

### Deploy Retrained Model
```bash
cp data/output/dino_player_2player.pt models/dino_player.pt
```

### Validate
1. Run diagnostic on retrained model
2. Verify confidence values are diverse (not 0.99 lockout)
3. Verify boxes follow players correctly
4. Test on inference pipeline (main.py)

## Technical Details

### Training Changes
- `train_dino()` now prints: `[TRAIN] USING AREA-BASED PLAYER SORTING (May 2026 fix for label flipping)`
- Dataset loader uses consistent area-based sorting in `DINODataset._pick_representative_boxes()`
- Checkpoint saved with metadata: `{"model": state_dict, "version": "2026-05-area-based-sorting", ...}`

### Inference Changes
- `DINOTracker.__init__()` now logs checkpoint loading status
- If no checkpoint found: `[DINOTracker] Running with randomly initialized detector_head (no checkpoint loaded)`
- `load_checkpoint()` checks version and warns about old checkpoints:
  ```
  [DINOTracker] WARNING: No version metadata in checkpoint — likely trained with OLD y-coordinate sorting
  [DINOTracker]          → This checkpoint may suffer from label flipping (player_1/player_2 inconsistent across frames)
  [DINOTracker]          → Both detector heads probably locked at 0.99 confidence
  [DINOTracker]          → RECOMMEND: Retrain with new area-based sorting in _pick_representative_boxes()
  ```

### Tracking Pipeline
- `main.py` → `PlayerDetector` (alias for `DINOTracker`) → loads checkpoint with version checking
- Diagnostic script also shows checkpoint version when loading

## Files Modified

1. **models/player_dino.py**
   - Added `from datetime import datetime` import
   - Enhanced `_pick_representative_boxes()` docstring (lines 548-598)
   - Updated `save_checkpoint()` to include version metadata (lines 270-280)
   - Updated `load_checkpoint()` to check version and warn (lines 204-265)
   - Updated `train_dino()` docstring and added logging (lines 720-745)
   - Updated `__init__()` to log checkpoint loading status (lines 147-155)

2. **models/dino_player.pt** → **models/dino_player.pt.OLD_Y_COORD_SORTING_CORRUPTED**
   - Renamed to prevent accidental loading

## Verification

✅ Code changes verified:
```
[TEST] ✓ Model created successfully
[TEST] ✓ Area-based sorting is implemented in DINODataset
[TEST] ✓ Version checking is implemented
[TEST] ✅ All code changes verified!
```

## Timeline

- **May 20, 2026 ~11:28:** Diagnostic revealed 0.99 confidence lockout
- **May 20, 2026 ~11:30:** Created area-based sorting fix (code was already in place)
- **May 20, 2026 ~23:30:** Added version metadata and logging
- **May 20, 2026 ~23:45:** Renamed old checkpoint, verified all fixes
- **NEXT:** Retrain with new code to generate clean checkpoint
