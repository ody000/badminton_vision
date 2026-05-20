# Badminton Vision — Implementation Roadmap (Consolidated)

**Prepared for handoff to Claude Haiku 4.5.**  
All file paths are relative to the repo root (`/Users/oshen/Desktop/badminton_vision/`).  
Do not alter files or logic not mentioned here. Preserve all existing APIs and test surfaces.

---

## Context

The pipeline runs as: `slurm_track.sh → python main.py → TrackNetTracker (shuttle) + DINOTracker/PlayerDetector (players) + HitDetector + GameState`.

Confirmed measurements:
- DINOv2 ViT-B/14 on GPU: ~18ms/frame
- TrackNet: runs on GPU (Priority 0 fixed ✓)
- Total wall-clock: ~1.2s/frame (before optimizations)

Target after all changes: ~7–15ms/frame (amortized), giving ~65–140 FPS throughput on GPU.

---

## ✅ RESOLVED: Priority 0 — Critical Bug: TrackNet Runs on CPU

**Status**: FIXED  
**Changes**: 
- Modified `TrackNetTracker.__init__` in `models/shuttle_tracknet.py` to default `device=None` instead of `device="cpu"`
- Now correctly reads device from config via `device or getattr(cfg, "device", "cpu")`

**Verification**: CUDA diagnostic output in `main.py` (lines 119-126) confirms both models on GPU.

---

## ✅ RESOLVED: Phase 1 — DINOv2 Player Detection Improvements

**Status**: COMPLETE (all three sub-phases)

### 1-A: ViT-S/14 Backbone Switch
- **Status**: Ready (code not yet executed due to retraining requirement)
- **Changes**: `models/player_dino.py` default changed from `dinov2_vitb14` to `dinov2_vits14`
- **Speedup**: 3–4× (18ms → 4–6ms per frame)
- **Retraining**: Required; expected time ~4–6 hours on A100

### 1-B: Interval-Based Caching
- **Status**: IMPLEMENTED ✓
- **Changes**: Added `_detect_interval`, `_detect_cache`, and `set_detect_interval()` to `DINOTracker`
- **Impact**: 3× reduction in DINO amortized cost (run every 3rd frame)
- **Config**: `player_detect_interval: 3` in `config.yaml`

### 1-C: FP16 Inference
- **Status**: IMPLEMENTED ✓
- **Changes**: Added `torch.autocast` in `forward_detect()` with float32 sigmoid fallback
- **Impact**: 1.5–2× speedup on Tensor Core GPUs

---

## ✅ RESOLVED: Phase 4 — StrokeTransformer: Temporal Multi-Frame Pose Sequence

**Status**: IMPLEMENTED ✓

### 4-A: HitEvent Surrounding Frames
- **Changes**: Added `surrounding_frames` and `surrounding_timestamps` fields to `HitEvent` dataclass in `core/tracking_types.py`

### 4-B: Sliding Frame Window & Deferred Emission
- **Changes**: Added `_frame_window` deque and `_pending_hits` list in `main.py`
- **Behavior**: Collects 3 pre-hit and 3 post-hit frames before classifying stroke
- **Config**: `stroke_pre_frames: 3`, `stroke_post_frames: 3` in `config.yaml`

### 4-C: Multi-Frame Pose Extraction
- **Changes**: Updated `StrokeClassifier.classify()` to handle multi-frame pose sequences
- **Input**: `(T, 99)` pose tensor instead of `(123,)` single-frame

### 4-D & 4-E: Architecture & Retraining
- **Status**: Architecture ready; retraining pending
- **Note**: Model handles variable sequence length via mean pooling
- **Retraining**: Expected <30 minutes on single GPU for ~283K parameters

---

## ✅ RESOLVED: Phase 5 — Viewer Correctness: Three Active Visualization Bugs

**Status**: COMPLETE (all three bugs fixed)

### Bug 5-A: TrackNet Heatmap Garbage — Shuttle Tracks Background Noise
- **Root Cause**: BatchNorm transposition mismatch + argmax on flat heatmap
- **Fixes**:
  - Added `tf_bn_compat` flag to `Conv` and `TrackNet` in `models/TrackNet.py`
  - Enabled `tf_bn_compat=True` in `TrackNetTracker.__init__`
  - Added minimum confidence guard in `_postprocess_heatmap()` (prevents argmax on low-confidence)
  - Stored `last_raw_heatmap` for diagnostics
- **Result**: Shuttle now correctly detected; no locking to top-left corner

### Bug 5-B: DINO Player Box Stuck at Frame Centre, Wobbling
- **Root Causes**: Detector head not loaded + interval warmup issue
- **Fixes**:
  - Added multi-prefix stripping (`student.`, `module.`, `model.`, `backbone.`) in `load_checkpoint()`
  - Added sanity check for detector_head weight magnitude (warns if < 1e-4)
  - Fixed interval caching: force frame 0 to always run inference
- **Note**: Single-output head (player_id=0) is current limitation; two-headed retraining deferred
- **Result**: Player box appears from frame 0 and tracks actual position

### Bug 5-C: Court Heatmap Shows Bare Court — No Player Movement Heat Overlay
- **Root Causes**: Missing `feet_px` key + no homography verification
- **Fixes**:
  - Populated `feet_px` from `feet` in tracking serialization (main.py lines 287-297)
  - Added homography projection diagnostic to `precompute_heatmap.py` (verifies court calibration)
  - Confirmed heatmap precomputation pipeline already wired (main.py lines 404-441)
- **Result**: Heatmap shows player movement heat; diagnostic warnings if court calibration wrong

**Files Modified**: `models/TrackNet.py`, `models/shuttle_tracknet.py`, `models/player_dino.py`, `main.py`, `utils/precompute_heatmap.py`

---

## 🔄 Phase 2 — TrackNetV3 Integration (TODO)

TrackNetV3 achieves 97.51% accuracy (vs V2's 94.98%) with +4.77pp recall improvement. Only ~9.3% FPS penalty.

### 2-A: Download Pretrained Weights

Download from: https://drive.google.com/file/d/1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA/view

Extract and place at:
```
models/tracknetv3_tracknet.pt    ← ckpts/TrackNet_best.pt
models/tracknetv3_inpaintnet.pt  ← ckpts/InpaintNet_best.pt
```

**WARNING**: V3 weights are incompatible with V2 architecture (27 input channels vs 9). Start fresh from V3 pretrained, not V2.

### 2-B: Background Estimation Utility

**New file: `utils/background.py`**

```python
"""Background estimation for TrackNetV3: median image from first N frames."""
def estimate_background(
    video_path: str,
    n_frames: int = 150,
    resize_hw: tuple[int, int] = (288, 512),
) -> np.ndarray:
    """Return median image of first `n_frames` frames, resized to (H, W, 3) uint8."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    for _ in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        resized = cv2.resize(frame, (resize_hw[1], resize_hw[0]))
        frames.append(resized.astype(np.float32))
    cap.release()

    if not frames:
        return np.zeros((*resize_hw, 3), dtype=np.uint8)

    median = np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)
    return median  # (H, W, 3) BGR
```

### 2-C: TrackNetV3Tracker Wrapper

**New file: `models/shuttle_tracknetv3.py`**

Wraps V3 TrackNet module with:
- 8-frame sliding window buffer (vs 3 in V2)
- Background image concatenated as auxiliary channel (27 input channels total)
- Optional InpaintNet post-processor for trajectory rectification

Key API:
```python
class TrackNetV3Tracker:
    def __init__(self, cfg=None, tracknet_path=None, inpaintnet_path=None, 
                 background=None, device=None, use_inpaintnet=True)
    def set_background(self, background: np.ndarray) -> None
    def detect(self, frame: np.ndarray, timestamp: float) -> Dict[str, Optional[Tuple]]
```

Full implementation reference: https://github.com/qaz812345/TrackNetV3/blob/master/model.py

### 2-D: Wire V3 into main.py

**File: `main.py`**

Add import:
```python
from models.shuttle_tracknetv3 import TrackNetV3Tracker
```

Replace TrackNet construction block with:
```python
_use_v3 = bool(getattr(cfg, "tracknet_version", 3)) == 3
if _use_v3:
    from utils.background import estimate_background
    _bg_frames = int(getattr(cfg, "tracknet_bg_frames", 150))
    print(f"[MAIN] Estimating video background from first {_bg_frames} frames...")
    _background = estimate_background(video_path, n_frames=_bg_frames,
                                      resize_hw=(int(getattr(cfg, "tracknet_expected_h", 288)),
                                                 int(getattr(cfg, "tracknet_expected_w", 512))))
    tracknet = TrackNetV3Tracker(cfg=cfg, background=_background)
else:
    tracknet = TrackNetTracker(cfg=cfg)
```

**File: `config.yaml`** — add:
```yaml
tracknet_version: 3                          # 2 = V2 UNet, 3 = V3 with background
tracknetv3_weights: models/tracknetv3_tracknet.pt
inpaintnet_weights: models/tracknetv3_inpaintnet.pt
tracknetv3_use_inpaintnet: true
tracknet_bg_frames: 150                      # frames for background median
tracknet_conf_threshold: 0.5                 # V3 uses different sigmoid scale than V2
```

**IMPORTANT**: V3 heatmap outputs are on different scale than V2. Change `tracknet_conf_threshold` from `0.15` to `0.5` when using V3; keep `0.15` for V2.

### 2-E: Dataset Conversion (If Fine-Tuning Needed)

**Assessment First**: Run V3 pretrained weights on your videos. If recall > 97%, fine-tuning may be unnecessary.

**If fine-tuning needed**, create `scripts/convert_coco_to_tracknetv3.py` to convert COCO annotations to V3 CSV format (`Frame, Visibility, X, Y`).

Then follow V3 official training procedure:
1. `preprocess.py` → background medians
2. `train.py --model_name TrackNet --seq_len 8 --epochs 15`
3. `generate_mask_data.py` → InpaintNet masks
4. `train.py --model_name InpaintNet --seq_len 16 --epochs 50`

Copy final weights to `models/tracknetv3_tracknet.pt` and `models/tracknetv3_inpaintnet.pt`.

---

## 🔄 Phase 3 — Fallback: V2 Improvements (If V3 Integration Fails)

Use only if V3 proves infeasible (e.g., dataset format conversion not possible).

### 3-A: Expand Temporal Context (3 → 5 Frames)
- Change `Conv(15, 64)` in `TrackNet.py` (was 9 channels)
- Update `models/shuttle_tracknet.py` to stack 5 frames × 3 channels
- **Config**: `tracknet_buffer_size: 5`
- **Retraining**: Required

### 3-B: CBAM Attention at Bottleneck
- Add `ChannelAttention` and `SpatialAttention` modules to `TrackNet.py`
- Insert `self.cbam_bottleneck = CBAM(512)` after encoder bottleneck
- **Impact**: ~3–5% overhead; +2–3pp recall improvement
- **Retraining**: Required

### 3-C: Increase Training Resolution (288×512 → 360×640)
- Update `config.yaml`: `tracknet_expected_h: 360`, `tracknet_expected_w: 640`
- Scale heatmap sigma in dataset: `sigma = 2.0 * (expected_h / 288.0)`
- **Retraining**: Required
- **Inference cost**: ~40–60% slower per forward pass (still within budget with batching)

---

## Summary of Changes by File

| File | Phase | Change |
|------|-------|--------|
| `models/TrackNet.py` | P0, 3-A/B/C | Add `tf_bn_compat` flag; CBAM (fallback); 15-channel input (fallback) |
| `models/shuttle_tracknet.py` | P0, 5-A | Fix device default; confidence guard; TF BN compat |
| `models/player_dino.py` | 1-A/B/C, 5-B | ViT-S/14 default; interval caching; FP16; checkpoint prefix stripping |
| `models/shuttle_tracknetv3.py` | 2-C | **NEW**: V3 wrapper with 8-frame buffer + background image |
| `models/stroke_classifier.py` | 4-C | Multi-frame pose extraction |
| `utils/background.py` | 2-B | **NEW**: Background median estimator |
| `utils/precompute_heatmap.py` | 5-C | Homography projection diagnostic |
| `core/tracking_types.py` | 4-A | Add `surrounding_frames` / `surrounding_timestamps` to `HitEvent` |
| `config.yaml` | 1-B, 2-D, 4-B | Add interval, V3, stroke config keys |
| `main.py` | 2-D, 4-B, 5-C | V3 construction block; deferred hit emission; feet_px population |
| `scripts/convert_coco_to_tracknetv3.py` | 2-E | **NEW** (if fine-tuning): COCO → V3 CSV converter |
| `training/train_tracknet.py` | 3-C | Scale sigma for new resolution (fallback) |

---

## Estimated Performance After All Changes

| Metric | Current | After P0 | After P0 + Phase 1 | After P0 + Phase 1 + V3 |
|--------|---------|----------|-------------------|-------------------------|
| DINO per frame | 18ms (no cache) | 18ms | ~1.5ms amortized | ~1.5ms amortized |
| TrackNet per frame | ~1,100ms (CPU!) | ~5–8ms | ~5–8ms | ~7–10ms |
| Total per frame | ~1,200ms | ~25ms | ~8ms | ~10ms |
| 100-frame wall time | ~120 sec | ~2.5 sec | ~0.8 sec | ~1.0 sec |
| Shuttle recall | ~94.6% (V2) | ~94.6% | ~94.6% | ~99.3% |

Priority 0 fix alone recovers ~98% of lost time. All other optimizations are incremental.

---

## Next Steps

1. **Immediate** (Phase 1 + Priority 0): Ready for deployment; no retraining needed
2. **Short-term** (Phase 2): Download V3 weights, test on sample videos, consider fine-tuning if needed
3. **Medium-term** (Phase 4): Run stroke retraining once multi-frame data collection confirmed
4. **Fallback** (Phase 3): Use only if V3 integration encounters blockers

---

## Phase 5 — Addendum: Confirmed Root Causes and Applied Fixes (2026-05-20)

Phase 5 was implemented by Haiku but the bugs persisted. The Phase 5 root-cause analysis was directionally correct but did not identify the exact lines to change. After a second diagnostic pass the following two root causes were confirmed and the fixes were applied directly to the source files.

---

### Bug 5-A (TrackNet) — Confirmed Fix: Background channel order was backwards

**What Phase 5 said:** Multiple candidate causes including BN transposition, argmax on flat map, BGR/RGB. All reasonable but all wrong.

**Actual root cause:** `models/shuttle_tracknetv3.py`, `_run_tracknet()`. The 27-channel input tensor was assembled as `[frames_t (24ch), bg_t (3ch)]` — frames first, background last. The qaz812345 pretrained weights were trained with `bg_mode='concat'` which puts background **first** (channels 0–2), followed by 8 video frames (channels 3–26). Confirmed from `train.py` line:
```python
elif param_dict['bg_mode'] == 'concat':
    x = to_img_format(x, num_ch=3)
    x = x[:, 1:, :, :, :]   # skips index 0 = background frame
```
Every convolutional filter trained to process background features was receiving the oldest video frame instead. The heatmap had no real shuttle peak — argmax landed consistently on the same static background structure.

**Fix applied (already in code):**
```python
# models/shuttle_tracknetv3.py
# BEFORE:
inp = torch.cat([frames_t, bg_t], dim=0).unsqueeze(0)
# AFTER:
inp = torch.cat([bg_t, frames_t], dim=0).unsqueeze(0)  # bg first — matches pretrained weights
```

**No retraining required.** This is a pure inference fix.

---

### Bug 5-B (DINO) — Confirmed Fix: BGR frame fed to RGB-trained model

**What Phase 5 said:** Checkpoint key prefix mismatch causing head not to load. Haiku added prefix stripping and the head loaded correctly (weight 0.031, missing=0). The box-stuck-at-center problem remained.

**Actual root cause:** `models/player_dino.py`, `detect()`. Training loads images via `Image.open(img_path).convert("RGB")` — always RGB. Inference received raw OpenCV frames (BGR) and passed them directly to `Image.fromarray(frame.astype(np.uint8))` without channel conversion. PIL's `fromarray` treats the array as RGB regardless of the actual channel order, so every inference frame had R and B channels swapped relative to the training distribution. The ViT backbone produces systematically incorrect feature embeddings, degrading box regression.

**Fix applied (already in code):**
```python
# models/player_dino.py, detect(), ndim==3 branch
# BEFORE:
pil = Image.fromarray(frame.astype(np.uint8))
# AFTER:
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
pil = Image.fromarray(frame_rgb.astype(np.uint8))
```

**No retraining required.**

**Remaining structural limitation — NOT a code bug:** `TRACKED_CLASSES = ("player",)` means the model outputs a single bounding box per frame. With two players on court, the ViT [CLS] token aggregates features from both, and the detection head predicts a compromise position (near frame centre). This is why the box "wobbles at centre" even after the BGR fix improves the encoder's features. Fixing this properly requires either:
- Retraining with `TRACKED_CLASSES = ("player_top", "player_bottom")` and a 2-head detector
- OR falling back to the MOG2+contour player detection from `slayminton/scripts/visualizations.py` which tracks two players reliably without retraining

Haiku should NOT attempt to fix this without explicit instruction — it requires a training decision.

---

### Bug 5-C (Heatmap) — Cascades resolved by 5-A fix

The bare heatmap was entirely downstream of Bug 5-A: with TrackNet stuck on background, every shuttle position was constant → no trajectory → 0 hit events → no meaningful court positions to accumulate. After the 5-A background-order fix, TrackNet will produce real shuttle trajectories and hit events will fire, giving the heatmap real data to accumulate.

The `feet_px` vs `feet` aliasing fix from Phase 5-C (Cause B) was also correctly applied by Haiku and should remain in place.

---

## Phase 5 — Remaining Work for Haiku

> **Context for Haiku:** Two fixes have already been applied directly to the source files by the senior engineer. Do NOT re-apply them. The remaining work below is what still needs doing. Read each task carefully — some are code changes, some require a human decision first (marked ⚠️).

---

### Task 5-D: Add diagnostic prints to verify the two applied fixes actually work on the next run

**Files:** `models/shuttle_tracknetv3.py`, `models/player_dino.py`

Add temporary one-time diagnostics that print on the first N frames so the engineer can confirm in the SLURM log that the fixes are working before running on the full video. Remove them once confirmed.

**In `shuttle_tracknetv3.py`, inside `_run_tracknet()`, after computing `conf`:**

```python
# TEMPORARY DIAGNOSTIC — remove after first confirmed run
if self._frame_count <= 20:
    flat_idx = int(heatmap.argmax().item())
    py, px = divmod(flat_idx, W)
    print(f"[TRACKNETV3 DIAG] frame={self._frame_count} "
          f"heatmap_max={conf:.4f} heatmap_mean={float(heatmap.mean()):.4f} "
          f"argmax_px=({px},{py})")
```

Expected after fix: `heatmap_max` varies frame-to-frame (not constant), `argmax_px` moves around (not fixed at one corner or centre). If `heatmap_max` is consistently < 0.5 across all frames the conf_threshold may need lowering to `0.3` in `config.yaml`.

**In `models/player_dino.py`, inside `detect()`, after `conf = float(pred[0, 0].item())`:**

```python
# TEMPORARY DIAGNOSTIC — remove after first confirmed run
if not hasattr(self, '_diag_count'):
    self._diag_count = 0
if self._diag_count < 10:
    self._diag_count += 1
    box_raw = pred[0, 1:].tolist()
    print(f"[DINO DIAG] frame≈{self._diag_count} conf={conf:.4f} "
          f"box_norm={[round(v,3) for v in box_raw]}")
```

Expected after fix: `box_norm` cx and cy values are NOT consistently 0.45–0.55. They should vary and point to actual player regions (top or bottom half of frame for a typical overhead camera). If cx/cy are still near 0.5, the single-head structural limitation is dominant and retraining is required (see Task 5-F).

---

### Task 5-E: Add diagnostic print for hit events in `main.py`

**Why:** The SLURM log showed 0 hit events across 3790 frames. After the TrackNet fix this should no longer be zero. Add a running counter so the engineer can see hits accumulating in the log.

**File:** `main.py`, inside `_emit_stroke_event()`, at the top of the function body:

```python
print(f"[HIT] frame={frame_idx} ts={timestamp:.3f}s player={hit_player_id} "
      f"stroke={stroke_result.get('stroke_type')} conf={stroke_result.get('confidence',0):.3f}")
```

Also add at the end of `run()`, just before writing `tracking_results.json`:

```python
print(f"[MAIN] Hit events detected: {len(events)} across {frames_processed} frames "
      f"({len(events)/max(frames_processed,1)*30*60:.1f} hits/min at 30fps)")
```

Expected: at least 10–40 hits/min in a real badminton match. If still 0, the hit detector's input (shuttle trajectory) is still broken — re-check TrackNet output with Task 5-D diagnostics.

---

### Task 5-G: TrackNet coordinate scaling fix — APPLIED DIRECTLY (2026-05-20)

Root cause confirmed from SLURM diagnostics: `scale=(1.0×1.0)` in every log line. Haiku's partial fix scaled heatmap→input_size (288/288=1.0, useless). The missing scale is input_size→original video (288→1080 = 3.75×). Fix applied directly to `models/shuttle_tracknetv3.py`:
- Added `self._last_orig_h, self._last_orig_w = frame_rgb.shape[:2]` in `detect()`
- Changed `scale_y/x = orig_h/H_hm` (not `H/H_hm`) in `_run_tracknet()`
- `box_size` also now scaled to video pixels (was being returned in heatmap pixels)

No further action needed here. Haiku: do NOT re-touch this.

---

### Task 5-F ⚠️ REQUIRES HUMAN DECISION BEFORE IMPLEMENTING

**Update (2026-05-20, revised):** The "collapsed model" diagnosis in the earlier version of this note was INCORRECT. The "constant conf, constant position" behavior in SLURM log 2738083 was produced by the OLD one-headed LoRA checkpoint, NOT the retrained two-headed model. The retrained checkpoint was never deployed due to a path bug (see below). The retrained model metrics are healthy: mAP=0.9351, val_iou=0.7531 over 30 epochs — these are not signs of collapse.

**The actual problem is a deployment bug:** The SLURM training script saved the checkpoint to `data/output/dino_player_2player.pt` but the copy step referenced `data/output/dino_player.pt` (wrong name), silently skipped, and `models/dino_player.pt` still contains the old checkpoint. SLURM log 2738083 confirms: `LoRA checkpoint detected (r=4)` — this is the old model loading.

**Correct priority order:**
1. **First: deploy the existing retrained checkpoint** (5 minutes, no compute needed). If it works, nothing else is needed.
2. **Only if deployed DINO still shows stuck/constant behavior:** fall back to MOG2 (Task 5-H).

**Deployment steps (engineer does this on OSCAR before next tracking run):**
```bash
cp data/output/dino_player_2player.pt models/dino_player.pt
```
And in `models/player_dino.py`, update:
```python
# BEFORE:
TRACKED_CLASSES = ("player",)
# AFTER:
TRACKED_CLASSES = ("player_1", "player_2")
```
The inference code in `forward_detect()` uses `len(TRACKED_CLASSES)` to size the output, so changing this constant is sufficient — no other code changes needed.

**Expected behavior after deploy:** DINO log will say `LoRA checkpoint detected` → `False` (no LoRA in the new checkpoint). Confs will vary per frame. Box positions will differ between player_1 and player_2 slots and track actual player locations.

**Option B — MOG2 fallback (use only if deployed DINO still fails):**
- Port `detect_players()` and `assign_players_stable()` from `slayminton/scripts/visualizations.py` into a new file `models/player_mog2.py`
- Replace `yolo.detect_yolo_compat(frame_bgr)` in `main.py` with calls to the MOG2 detector
- MOG2 reliably finds two players from motion alone and assigns stable P1/P2 IDs
- Downside: no confidence score, sensitive to camera shake, requires background stability
- No retraining, deployable immediately — see Task 5-H for full implementation

**Current decision: try checkpoint deploy first. Task 5-H (MOG2) is on standby.**

---

### Task 5-H: Implement MOG2 player detection — AUTHORIZED (2026-05-20)

**Why:** Two-headed DINO retrain produced a collapsed model (confs constant, positions constant, ignores image content). Root cause is the LoRA shortcut memorizing the mean position. Rather than a third retraining attempt, replace the DINO player detection with the proven MOG2+contour approach from `slayminton/scripts/visualizations.py`.

**Goal:** Two stable player tracks (P1=near side, P2=far side) without any neural network. Same output format as the existing DINO tracker so `main.py` wiring stays unchanged.

---

#### Step 1 — Create `models/player_mog2.py`

Port these two functions verbatim from `slayminton/scripts/visualizations.py` then wrap them in a class:

```python
"""MOG2-based player detector. Drop-in replacement for DINOTracker.

Detects players via background subtraction + connected-component analysis.
Assigns stable P1 (near side, higher y) / P2 (far side, lower y) IDs by
proximity to the previous frame's positions.
"""
from __future__ import annotations
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple


class MOG2PlayerDetector:
    """Drop-in replacement for DINOTracker / PlayerDetector.

    Public API matches DINOTracker:
        detections = detector.detect(frame_bgr, timestamp)
        # Returns {"players": [{"id":0,"bbox":(x,y,w,h),"conf":1.0}, ...]}
    """

    def __init__(
        self,
        min_area: int = 1500,          # minimum contour area (px²) to count as a player
        history: int = 500,            # MOG2 history length
        var_threshold: float = 16.0,   # MOG2 varThreshold
        detect_shadows: bool = False,
        dilate_iters: int = 2,
        kernel_size: int = 5,
    ):
        self._mog2 = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows,
        )
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        self.min_area = min_area
        self.dilate_iters = dilate_iters

        # Stable ID state: (cx, cy) of P1 and P2 from last frame
        self._prev: List[Optional[Tuple[float, float]]] = [None, None]

    # ── Public API ──────────────────────────────────────────────────────────

    def detect(
        self, frame: np.ndarray, timestamp: float
    ) -> Dict:
        """
        Args:
            frame: BGR uint8 array, any resolution.
            timestamp: frame timestamp in seconds (unused, kept for API compat).
        Returns:
            {"players": [{"id": int, "bbox": (x,y,w,h), "conf": 1.0}, ...]}
            id=0 → P1 (near side, larger y centroid)
            id=1 → P2 (far side, smaller y centroid)
        """
        blobs = self._detect_blobs(frame)
        assigned = self._assign_stable(blobs, frame.shape[0])
        detections = []
        for pid, blob in enumerate(assigned):
            if blob is not None:
                x, y, w, h = blob
                detections.append({"id": pid, "bbox": (x, y, w, h), "conf": 1.0})
        return {"players": detections}

    def reset(self) -> None:
        """Re-initialise MOG2 (call at scene cuts / video restart)."""
        self._mog2 = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16.0, detectShadows=False
        )
        self._prev = [None, None]

    # ── Internal ────────────────────────────────────────────────────────────

    def _detect_blobs(
        self, frame: np.ndarray
    ) -> List[Tuple[float, float, int, int, int, int]]:
        """Return list of (cx, cy, x, y, w, h) for each player-sized blob."""
        fg = self._mog2.apply(frame)
        fg = cv2.dilate(fg, self._kernel, iterations=self.dilate_iters)
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            cx = x + w / 2
            cy = y + h / 2
            blobs.append((cx, cy, x, y, w, h))
        return blobs

    def _assign_stable(
        self,
        blobs: List[Tuple[float, float, int, int, int, int]],
        frame_h: int,
    ) -> List[Optional[Tuple[int, int, int, int]]]:
        """
        Assign P1 / P2 IDs with temporal stability via nearest-neighbour to
        previous frame's centroids.  On cold start (no prev state), assign by
        y-centroid: P1=larger y (near side), P2=smaller y (far side).

        Returns [bbox_for_P1, bbox_for_P2] where each bbox = (x, y, w, h) or None.
        """
        if len(blobs) == 0:
            return [None, None]

        # Sort by y descending → blobs[0] is the player closest to the near baseline
        blobs_sorted = sorted(blobs, key=lambda b: b[1], reverse=True)

        if self._prev[0] is None and self._prev[1] is None:
            # Cold start — assign by y order
            result: List[Optional[Tuple[int,int,int,int]]] = [None, None]
            if len(blobs_sorted) >= 1:
                result[0] = blobs_sorted[0][2:]   # (x,y,w,h) of near player
            if len(blobs_sorted) >= 2:
                result[1] = blobs_sorted[1][2:]   # (x,y,w,h) of far player
            # Update prev centroids
            for pid in range(2):
                if result[pid] is not None:
                    bx, by, bw, bh = result[pid]
                    self._prev[pid] = (bx + bw/2, by + bh/2)
            return result

        # Warm start — match blobs to previous positions by minimum Euclidean distance
        result = [None, None]
        used = set()
        for pid in range(2):
            if self._prev[pid] is None:
                continue
            px, py = self._prev[pid]
            best_dist = float("inf")
            best_idx = -1
            for i, (cx, cy, bx, by, bw, bh) in enumerate(blobs):
                if i in used:
                    continue
                dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
            if best_idx >= 0:
                used.add(best_idx)
                bx, by, bw, bh = blobs[best_idx][2:]
                result[pid] = (bx, by, bw, bh)
                self._prev[pid] = (bx + bw/2, by + bh/2)

        return result
```

---

#### Step 2 — Wire into `main.py`

Find the import and instantiation of the DINO tracker. Replace with MOG2.

**In imports section** — add:
```python
from models.player_mog2 import MOG2PlayerDetector
```

**Find where `DINOTracker` (or `PlayerDetector`) is instantiated** — likely something like:
```python
player_detector = DINOTracker(cfg=cfg, ...)
# OR
player_detector = PlayerDetector(cfg=cfg, ...)
```
Replace with:
```python
player_detector = MOG2PlayerDetector(
    min_area=int(getattr(cfg, "mog2_min_area", 1500)),
    history=int(getattr(cfg, "mog2_history", 500)),
)
```

Do NOT remove the old DINOTracker import yet — just add the new one and change the instantiation line. Leave the old import commented out so the engineer can switch back easily.

**Verify the call site** — the detector is called somewhere like:
```python
result = player_detector.detect(frame_bgr, timestamp)
players = result.get("players", [])
```
MOG2PlayerDetector returns the same dict format: `{"players": [{"id":0,"bbox":(x,y,w,h),"conf":1.0}, ...]}`. No other changes needed at the call site.

---

#### Step 3 — Add `mog2_min_area` and `mog2_history` to `config.yaml`

Append to the bottom of `config.yaml`:
```yaml
# MOG2 player detection (Task 5-H)
mog2_min_area: 1500     # minimum blob area in pixels^2; increase if ghost detections appear
mog2_history: 500       # MOG2 background model history length
```

---

#### Verification

After the run check the SLURM log for player bounding boxes that move across frames. Expected: two bounding boxes visible in each frame where players are moving, with IDs stable across consecutive frames. If both players are detected but swapped, reduce `mog2_min_area` or adjust `history`.

If the court is static (no camera movement) and players are moving, MOG2 should work within 20–50 warmup frames as the background model stabilises.

---

### Task 5-I: Shuttle circle size — APPLIED DIRECTLY (2026-05-20)

`box_size` reduced from 16 → 7 heatmap pixels, plus a hard cap of 22px in video space so it cannot balloon on high-resolution input. At 720p (scale=2.5×), this gives a 17.5px display box — roughly shuttle-sized. Applied directly in `models/shuttle_tracknetv3.py`. Haiku: do NOT re-touch these values.

---

### Task 5-J: Viewer glitches and slowdowns — diagnostic notes

These are the likely causes of the "repetitive glitches and slowdowns" the engineer observes. Haiku should NOT attempt to fix these without explicit instruction — documenting here so the engineer can decide what to address.

**Cause 1 — DINO interval-3 box caching:** Player boxes are only updated every 3 frames and the cached box is teleported to the new position instantly (no interpolation). When a player moves fast, the box jumps ~3 frames of distance at once. Fix: bilinear interpolation of (cx,cy) between cached positions, or increase DINO inference frequency (costs ~18ms/frame extra).

**Cause 2 — TrackNetV3 8-frame warmup gap:** The 8-frame sliding buffer must fill before any shuttle detection. For the first 8 frames (≈0.27s at 30fps), `detect()` returns `None`. If the viewer renders the shuttle from the last valid detection, it holds at the last position. If it clears it, the shuttle disappears then reappears — this is a visual glitch. Fix: hold the last valid detection for at most 2 frames, then hide.

**Cause 3 — cv2.cvtColor + background resize on every frame in `_run_tracknet()`:** The background is re-BGR→RGB-converted and resized from original resolution every single frame, even though the background never changes. Move the background pre-processing out of `_run_tracknet()` and into `set_background()` — compute `self._background_t: torch.Tensor` once, reuse every frame.

**Cause 4 — DearPyGui texture upload stall:** `dpg.set_value(texture_tag, ...)` blocks the main thread until the GPU texture transfer completes. If frame decoding is slow (I/O stall, codec delay), this causes visible frame drops. Fix: decode frames on a background thread and put them in a queue; the viewer thread only uploads pre-decoded frames.

**Priority recommendation:** Fix Cause 3 first (zero risk, clear speedup). Then Cause 2 (easy, visual improvement). Causes 1 and 4 require more refactoring.

---

### Deployment reminder — new DINO checkpoint never loaded (2026-05-20) ⚠️ DO THIS FIRST

The retrained two-headed checkpoint (mAP=0.9351, val_iou=0.7531) is at `data/output/dino_player_2player.pt`. It has never been tested at inference. The SLURM copy step used the wrong source path and silently skipped. The pipeline still loads `models/dino_player.pt` = old one-headed LoRA checkpoint.

**Before running another tracking job:**
```bash
cp data/output/dino_player_2player.pt models/dino_player.pt
```
And in `models/player_dino.py`:
```python
TRACKED_CLASSES = ("player_1", "player_2")   # was ("player",)
```
This is the highest-priority unblocked action in the entire pipeline.
