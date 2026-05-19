# Badminton Vision — Implementation Roadmap

**Prepared for handoff to Claude Haiku 4.5.**  
All file paths are relative to the repo root (`/Users/oshen/Desktop/badminton_vision/`).  
Do not alter files or logic not mentioned here. Preserve all existing APIs and test surfaces.

---

## Context

The pipeline runs as: `slurm_track.sh → python main.py → TrackNetTracker (shuttle) + DINOTracker/PlayerDetector (players) + HitDetector + GameState`.

Confirmed measurements:
- DINOv2 ViT-B/14 on GPU: ~18ms/frame (reasonable, encoder confirmed on `cuda:0`)
- TrackNet: currently running on **CPU** (bug — see Priority 0 below)
- Total wall-clock: ~1.2s/frame (~2 min per 100 frames) — almost entirely TrackNet-on-CPU

Target after all changes: ~7–15ms/frame (amortized), giving ~65–140 FPS throughput on GPU.

---

## Priority 0 — Critical Bug: TrackNet Runs on CPU

### Root cause

In `models/shuttle_tracknet.py`, `TrackNetTracker.__init__` has this signature:

```python
def __init__(self, cfg=None, weights_path=None, device: str = "cpu", ...):
    if cfg is not None:
        device = device or getattr(cfg, "device", "cpu")
```

`"cpu" or getattr(...)` short-circuits immediately because `"cpu"` is a truthy string. The config's `device: cuda` is **never read**. TrackNet therefore always initialises on CPU regardless of SLURM flags.

### Fix — `models/shuttle_tracknet.py`

Change the signature so `device` defaults to `None`:

```python
def __init__(
    self,
    cfg=None,
    weights_path: str | None = None,
    device: str | None = None,        # ← was: device: str = "cpu"
    box_size: int = 16,
    conf_threshold: float = 0.001,
    expected_h: int = 288,
    expected_w: int = 512,
    fps: float = 30.0,
):
    if cfg is not None:
        weights_path = weights_path or getattr(cfg, "tracknet_weights", "models/tracknet.pt")
        device = device or getattr(cfg, "device", "cpu")   # now works: None or "cuda" → "cuda"
        ...
    else:
        device = device or "cpu"
```

No other changes needed. `self.device = torch.device(device if isinstance(device, str) else ...)` already handles the rest.

### Verification

After the fix, add a one-time print to `main.py` inside `run()` after both models are initialised:

```python
print(f"[MAIN] CUDA diagnostic: "
      f"torch.cuda.is_available()={torch.cuda.is_available()}, "
      f"yolo.encoder.device={next(yolo.encoder.parameters()).device}, "
      f"tracknet.model.device={next(tracknet.model.parameters()).device}")
```

Both should show `cuda:0`.

---

## Phase 1 — DINOv2 Player Detection Improvements

Three independent improvements. Apply all three; they compose without conflict.

### 1-A: Switch backbone from ViT-B/14 to ViT-S/14

**Why:** ViT-S/14 shares `patch_size=14` with ViT-B/14 (same token grid, same 392×392 adjusted input), but has `embed_dim=384` vs `768`. This reduces MLP flops by 4× and attention flops by 2×, giving ~3–4× wall-clock speedup (from ~18ms → ~4–6ms per frame). File size drops from ~300MB to ~87MB. DINOv2-pretrained ViT-S/14 weights are available from the same `facebookresearch/dinov2` hub.

**File: `models/player_dino.py`**

In `_create_vit_tiny` (lines ~901–912), change the default model name:

```python
def _create_vit_tiny(pretrained_weights_path=None):
    import timm
    dinov2_model = os.environ.get("DINOV2_MODEL", "dinov2_vits14")  # ← was "dinov2_vitb14"
    try:
        print(f"[DINOTracker] Loading DINOv2: {dinov2_model}")
        encoder = torch.hub.load("facebookresearch/dinov2", dinov2_model)
        embed_dim = getattr(encoder, "embed_dim", None) or getattr(encoder, "num_features", None) or 384
        print(f"[DINOTracker] Loaded DINOv2 (embed_dim={embed_dim})")
        return encoder, int(embed_dim)
    except Exception as e:
        print(f"[DINOTracker] DINOv2 load failed ({e}), using timm ViT-small")
        encoder = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=0, dynamic_img_size=True)
        embed_dim = getattr(encoder, "embed_dim", 384)
        return encoder, int(embed_dim)
```

In `_create_vit_by_embed_dim`, update the `embed_dim == 384` branch to prefer ViT-S/14:

```python
elif embed_dim == 384:
    try:
        encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        ed = getattr(encoder, "embed_dim", None) or 384
        return encoder, int(ed)
    except Exception:
        model_name = "vit_small_patch16_224"
```

**Retraining required:** Yes. The existing `models/dino_player.pt` was trained with ViT-B/14 (embed_dim=768). After switching to ViT-S/14 (embed_dim=384), the detection head shape changes and the checkpoint is incompatible. Run:

```bash
sbatch --gres=gpu:1 --mem=64G \
  --export=MODE=train-dino,TRAIN_DIR=data/input/train/player,EPOCHS=75 \
  slurm_train.sh
```

The new checkpoint will be ~87MB. Expected training time on 30k images: ~4–6 hours on a single A100.

---

### 1-B: Interval-based caching for player detection

**Why:** Players move slowly relative to framerate (~5–15px/frame at 30fps broadcast angle). Running a ViT forward pass on every frame is wasteful. Caching every 3 frames costs negligible accuracy (bounding box is stale by ~15–30px maximum) while reducing DINO's amortized cost by 3×.

**File: `models/player_dino.py`**

Add instance variables to `DINOTracker.__init__`, after `self.eval()` at the end of `__init__`:

```python
# Interval caching for detect_yolo_compat
self._detect_interval: int = 1       # overridden by set_detect_interval()
self._detect_frame_count: int = 0
self._detect_cache: list = []        # last result from detect_yolo_compat
```

Add a setter method to `DINOTracker` (after `__init__`):

```python
def set_detect_interval(self, interval: int) -> None:
    """Set how often to run a real forward pass vs return cached result."""
    self._detect_interval = max(1, int(interval))
```

Replace the body of `detect_yolo_compat` with:

```python
@torch.no_grad()
def detect_yolo_compat(
    self,
    frame,
    timestamp: float = 0.0,
    min_confidence: float = MIN_CONFIDENCE,
) -> List[dict]:
    """Detect player, returning YOLO-compatible format with optional interval caching."""
    run_inference = (self._detect_frame_count % self._detect_interval == 0)
    self._detect_frame_count += 1

    if run_inference:
        result = self.detect(frame, timestamp, min_confidence)
        player_det = result.get("player")

        if player_det is None:
            self._detect_cache = []
        else:
            ts, x, y, w, h = player_det
            if isinstance(frame, np.ndarray):
                orig_h, orig_w = frame.shape[:2]
            elif isinstance(frame, torch.Tensor):
                orig_h, orig_w = int(frame.shape[1]), int(frame.shape[2])
            else:
                orig_h, orig_w = 1080, 1920
            x1, y1 = x, y
            x2, y2 = x + w, y + h
            x1 = max(0.0, min(x1, orig_w - 1.0))
            y1 = max(0.0, min(y1, orig_h - 1.0))
            x2 = max(x1 + 1.0, min(x2, float(orig_w)))
            y2 = max(y1 + 1.0, min(y2, float(orig_h)))
            cx = (x1 + x2) / 2.0
            self._detect_cache = [
                {"id": 0, "box": [x1, y1, x2, y2], "feet": (cx, y2), "feet_real": None}
            ]

    return self._detect_cache
```

**File: `main.py`**

After `yolo = PlayerDetector(...)` is constructed (around line 100), read the interval from config and apply it:

```python
_player_interval = int(getattr(cfg, "player_detect_interval", 1))
yolo.set_detect_interval(_player_interval)
print(f"[MAIN] DINOTracker interval caching: every {_player_interval} frame(s)")
```

**File: `config.yaml`**

```yaml
# ─── Player tracking ─────────────────────────────────────────────────────────
player_conf_threshold: 0.25
player_weights: models/dino_player.pt
player_detect_interval: 3          # ← was 1; run DINO every 3rd frame
```

---

### 1-C: FP16 (half-precision) inference for DINO

**Why:** ViT-B/14 and ViT-S/14 are attention-heavy. On Tensor Core–capable GPUs (V100, A100, RTX series), FP16 gives ~1.5–2× wall-clock speedup with negligible precision loss for detection.

**File: `models/player_dino.py`**

Replace `forward_detect` with:

```python
def forward_detect(self, x: torch.Tensor) -> torch.Tensor:
    """Detection forward pass with optional FP16 autocast.

    Args:
        x: Tensor (B, 3, H, W)
    Returns:
        Tensor (B, num_classes, 5) with [conf, cx, cy, w, h] normalized
    """
    use_amp = (self.device.type == "cuda")
    with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
        feat = self.encode(x)
        raw = self.detector_head(feat).view(x.size(0), len(TRACKED_CLASSES), 5)
    # Sigmoid in float32 to avoid precision loss in probability outputs
    raw = raw.float()
    conf = torch.sigmoid(raw[..., :1])
    box  = torch.sigmoid(raw[..., 1:])
    return torch.cat([conf, box], dim=-1)
```

No other changes required. `detect()` is already decorated with `@torch.no_grad()`.

---

## Phase 2 — TrackNetV3 Integration

### Overview

TrackNetV3 is a two-module system:
1. **TrackNet module**: detects shuttle position per frame using 8-frame temporal context + a precomputed background image as auxiliary input. Input shape: `(B, 27, 288, 512)` — 8 RGB frames (24ch) + 1 RGB background (3ch).
2. **InpaintNet module**: trajectory rectifier that operates on *predicted coordinate sequences* (not raw pixels) to fill in frames where the shuttle was missed or occluded. Input: sequence of (x, y, visibility) predictions over 16-frame windows. Very low per-frame cost.

Performance vs V2 (on Shuttlecock Trajectory Dataset benchmark):

| | Accuracy | Precision | Recall | F1 | FPS |
|---|---|---|---|---|---|
| TrackNetV2 | 94.98% | 99.64% | 94.56% | 97.03% | 27.70 |
| TrackNetV3 | 97.51% | 97.79% | **99.33%** | **98.56%** | 25.11 |

FPS penalty: only ~9.3%. Recall gain: +4.77pp — eliminates most shuttle-miss frames that currently break hit detection.

### 2-A: Download pretrained weights

Pretrained weights are available from Google Drive (linked in the TrackNetV3 README):
`https://drive.google.com/file/d/1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA/view`

Download and unzip. Place files at:
```
models/tracknetv3_tracknet.pt    ← ckpts/TrackNet_best.pt from the zip
models/tracknetv3_inpaintnet.pt  ← ckpts/InpaintNet_best.pt from the zip
```

**Note on V2→V3 weight transfer:** Do NOT attempt to load `models/tracknet.pt` (V2) into V3. V3's first conv layer has 27 input channels vs V2's 9. They are incompatible. Always start from the V3 pretrained weights.

### 2-B: Background estimation utility

V3 requires a per-video background estimate (typically the median image of a rally or match). For a fixed overhead camera this is trivial. Add a new utility:

**New file: `utils/background.py`**

```python
"""Background estimation for TrackNetV3: median image from first N frames."""
from __future__ import annotations
import cv2
import numpy as np


def estimate_background(
    video_path: str,
    n_frames: int = 150,
    resize_hw: tuple[int, int] = (288, 512),
) -> np.ndarray:
    """Return median image of first `n_frames` frames, resized to (H, W, 3) uint8.

    Args:
        video_path: Path to input video.
        n_frames:   Number of frames to sample for median (150 = 5s at 30fps).
        resize_hw:  Target (H, W) matching TrackNet input resolution.

    Returns:
        np.ndarray of shape (H, W, 3) dtype uint8.
    """
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

### 2-C: New TrackNetV3Tracker wrapper

This replaces `TrackNetTracker` in the pipeline. It wraps the V3 TrackNet module with an 8-frame sliding window buffer and an optional InpaintNet post-processor.

**New file: `models/shuttle_tracknetv3.py`**

```python
"""TrackNetV3Tracker: wrapper for TrackNetV3 shuttle detection.

Uses 8-frame temporal context + background image as auxiliary input.
Optionally applies InpaintNet trajectory rectification.

References:
    - Architecture: https://github.com/qaz812345/TrackNetV3
    - Pretrained weights: models/tracknetv3_tracknet.pt, models/tracknetv3_inpaintnet.pt
"""
from __future__ import annotations

import os
from collections import deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Import V3 model architecture ───────────────────────────────────────────────
# The TrackNetV3 model.py defines TrackNet and InpaintNet classes.
# Copy model.py from https://github.com/qaz812345/TrackNetV3/blob/master/model.py
# into models/tracknetv3_arch.py (do not rename functions inside it).
from models.tracknetv3_arch import TrackNet as _TrackNetV3Arch
from models.tracknetv3_arch import InpaintNet as _InpaintNetArch


SEQ_LEN = 8            # V3 uses 8-frame temporal window
INPUT_H  = 288
INPUT_W  = 512


class TrackNetV3Tracker:
    """Drop-in replacement for TrackNetTracker using TrackNetV3 architecture.

    Key differences from V2:
    - 8-frame input window (vs 3 in V2)
    - Background image concatenated as auxiliary channel
    - Optional InpaintNet for trajectory rectification
    """

    def __init__(
        self,
        cfg=None,
        tracknet_path: str | None = None,
        inpaintnet_path: str | None = None,
        background: np.ndarray | None = None,
        device: str | None = None,
        box_size: int = 16,
        conf_threshold: float = 0.5,
        expected_h: int = INPUT_H,
        expected_w: int = INPUT_W,
        fps: float = 30.0,
        use_inpaintnet: bool = True,
    ):
        if cfg is not None:
            tracknet_path  = tracknet_path  or getattr(cfg, "tracknetv3_weights",   "models/tracknetv3_tracknet.pt")
            inpaintnet_path = inpaintnet_path or getattr(cfg, "inpaintnet_weights",  "models/tracknetv3_inpaintnet.pt")
            device         = device         or getattr(cfg, "device", "cpu")
            box_size       = int(getattr(cfg, "tracknet_box_size",       box_size))
            conf_threshold = float(getattr(cfg, "tracknet_conf_threshold", conf_threshold))
            expected_h     = int(getattr(cfg, "tracknet_expected_h",     expected_h))
            expected_w     = int(getattr(cfg, "tracknet_expected_w",     expected_w))
            fps            = float(getattr(cfg, "fps", fps))
            use_inpaintnet = bool(getattr(cfg, "tracknetv3_use_inpaintnet", use_inpaintnet))
        else:
            device = device or "cpu"

        self.device = torch.device(device)
        self.box_size = box_size
        self.conf_threshold = conf_threshold
        self.expected_size = (expected_h, expected_w)
        self.fps = fps
        self._frame_count = 0

        # 8-frame sliding buffer: deque of (timestamp, frame_rgb_resized) tuples
        self._buffer: deque[tuple[float, np.ndarray]] = deque(maxlen=SEQ_LEN)

        # Background image (H, W, 3) uint8 — set via set_background() or constructor
        self._background: np.ndarray | None = background

        # Load TrackNet V3
        self.tracknet = _TrackNetV3Arch(in_channels=SEQ_LEN * 3 + 3, out_channels=1)
        if tracknet_path and os.path.exists(tracknet_path):
            state = torch.load(tracknet_path, map_location="cpu")
            sd = state.get("state_dict", state) if isinstance(state, dict) else state
            self.tracknet.load_state_dict(sd, strict=False)
            print(f"[TRACKNETV3] Loaded TrackNet from {tracknet_path}")
        else:
            print(f"[TRACKNETV3] Warning: TrackNet weights not found at {tracknet_path}")
        self.tracknet.to(self.device).eval()

        # Load InpaintNet (optional)
        self.inpaintnet: Optional[nn.Module] = None
        if use_inpaintnet:
            if inpaintnet_path and os.path.exists(inpaintnet_path):
                self.inpaintnet = _InpaintNetArch()
                state = torch.load(inpaintnet_path, map_location="cpu")
                sd = state.get("state_dict", state) if isinstance(state, dict) else state
                self.inpaintnet.load_state_dict(sd, strict=False)
                self.inpaintnet.to(self.device).eval()
                print(f"[TRACKNETV3] Loaded InpaintNet from {inpaintnet_path}")
            else:
                print(f"[TRACKNETV3] InpaintNet weights not found at {inpaintnet_path}; running without rectification.")

        # Trajectory buffer for InpaintNet post-processing
        # Stores (timestamp, x, y, visibility) tuples from TrackNet raw output
        self._traj_buffer: List[Tuple[float, float, float, float]] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def set_fps(self, fps: float) -> None:
        self.fps = float(fps)

    def set_background(self, background: np.ndarray) -> None:
        """Set the background estimate (H, W, 3) uint8 BGR."""
        self._background = background

    @torch.no_grad()
    def detect(
        self,
        frame: np.ndarray,
        timestamp: float,
    ) -> Dict[str, Optional[Tuple[float, float, float, float, float]]]:
        """Detect shuttle in frame using V3 TrackNet.

        Returns {"shuttle": (ts, x, y, w, h)} or {"shuttle": None}.
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized   = cv2.resize(frame_rgb, (self.expected_size[1], self.expected_size[0]))
        self._buffer.append((timestamp, resized))
        self._frame_count += 1

        if len(self._buffer) < SEQ_LEN:
            return {"shuttle": None}

        return self._run_tracknet(timestamp)

    @torch.no_grad()
    def detect_batch(
        self,
        frames: List[np.ndarray],
        timestamps: List[float],
    ) -> List[Dict[str, Optional[Tuple[float, float, float, float, float]]]]:
        """Batch detect over a list of frames."""
        results = []
        for frame, ts in zip(frames, timestamps):
            results.append(self.detect(frame, ts))
        return results

    # ── Internal ──────────────────────────────────────────────────────────────

    def _run_tracknet(
        self, timestamp: float
    ) -> Dict[str, Optional[Tuple[float, float, float, float, float]]]:
        """Run V3 TrackNet forward pass on the current 8-frame buffer."""
        H, W = self.expected_size

        # Stack 8 frames: (8, H, W, 3) → (24, H, W) normalised
        frames_np = np.stack([f for _, f in self._buffer], axis=0)  # (8, H, W, 3)
        frames_t  = torch.from_numpy(frames_np).float().permute(0, 3, 1, 2) / 255.0  # (8, 3, H, W)
        frames_t  = frames_t.reshape(-1, H, W)  # (24, H, W)

        # Background: (3, H, W)
        if self._background is not None:
            bg_rgb   = cv2.cvtColor(self._background, cv2.COLOR_BGR2RGB)
            bg_rs    = cv2.resize(bg_rgb, (W, H))
            bg_t     = torch.from_numpy(bg_rs).float().permute(2, 0, 1) / 255.0  # (3, H, W)
        else:
            bg_t = torch.zeros(3, H, W)

        # Concatenate: (27, H, W)
        inp = torch.cat([frames_t, bg_t], dim=0).unsqueeze(0).to(self.device)  # (1, 27, H, W)

        use_amp = (self.device.type == "cuda")
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            heatmap = self.tracknet(inp)  # (1, 1, H, W)
        heatmap = heatmap.float().squeeze()  # (H, W)

        conf = float(heatmap.max().item())
        if conf < self.conf_threshold:
            self._traj_buffer.append((timestamp, -1.0, -1.0, 0.0))
            return {"shuttle": None}

        # Argmax → pixel position
        flat_idx = int(heatmap.argmax().item())
        py, px   = divmod(flat_idx, W)
        bs       = self.box_size // 2
        x0 = max(0, px - bs)
        y0 = max(0, py - bs)

        self._traj_buffer.append((timestamp, float(px), float(py), 1.0))
        return {"shuttle": (timestamp, float(x0), float(y0), float(self.box_size), float(self.box_size))}
```

**Note to Haiku:** The `_TrackNetV3Arch` and `_InpaintNetArch` classes must be sourced from the official TrackNetV3 repository's `model.py`. Copy that file verbatim from `https://github.com/qaz812345/TrackNetV3/blob/master/model.py` to `models/tracknetv3_arch.py`. Do not modify the architecture classes inside it — only the wrapper above interacts with them.

### 2-D: Wire V3 into `main.py`

**File: `main.py`**

At the top, add the new import alongside the existing one:

```python
from models.shuttle_tracknetv3 import TrackNetV3Tracker
```

Inside `run()`, replace the TrackNet construction block (currently just `tracknet = TrackNetTracker(cfg=cfg)`) with:

```python
# Determine which TrackNet version to use
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

**File: `config.yaml`** — add new keys:

```yaml
# ─── TrackNetV3 ──────────────────────────────────────────────────────────────
tracknet_version: 3                          # 2 = original UNet, 3 = V3 with background
tracknetv3_weights: models/tracknetv3_tracknet.pt
inpaintnet_weights: models/tracknetv3_inpaintnet.pt
tracknetv3_use_inpaintnet: true
tracknet_bg_frames: 150                      # frames sampled for background median
tracknet_conf_threshold: 0.5                 # V3 uses higher sigmoid output scale than V2
```

**Important:** V3's heatmap outputs are on a different scale than V2 (different final activation). The `tracknet_conf_threshold` default must be changed from `0.15` to `0.5` when using V3. Keep `0.15` if falling back to V2.

### 2-E: Dataset conversion for fine-tuning V3 (if needed)

**Assess first:** Run the V3 pretrained weights on your actual match videos. If recall is already satisfactory (>97%), fine-tuning may not be necessary — V3 was trained on professional badminton footage and generalises well.

**If fine-tuning is needed**, a conversion script is required to transform your COCO annotations into V3 CSV format (`Frame, Visibility, X, Y`).

**New file: `scripts/convert_coco_to_tracknetv3.py`**

```python
"""Convert COCO-format shuttle annotations to TrackNetV3 CSV format.

V3 expects per-rally CSVs with columns: Frame, Visibility, X, Y
This script assumes frames were extracted from video with temporal ordering
preserved in the filename (e.g., frame_000001.jpg, frame_000002.jpg, ...).

Usage:
    python scripts/convert_coco_to_tracknetv3.py \
        --coco-json data/input/train/_annotations.coco.json \
        --image-dir data/input/train \
        --out-dir data/v3_dataset/train/match1/csv
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd


def convert(coco_json: str, image_dir: str, out_dir: str, rally_id: str = "1_01_00") -> None:
    os.makedirs(out_dir, exist_ok=True)

    with open(coco_json) as f:
        coco = json.load(f)

    # Build image_id → file_name map
    id_to_file = {im["id"]: im["file_name"] for im in coco["images"]}
    id_to_idx  = {im["id"]: i for i, im in enumerate(sorted(coco["images"], key=lambda x: x["file_name"]))}

    # Identify shuttle category
    shuttle_cat_ids = {
        c["id"] for c in coco.get("categories", [])
        if c["name"].lower() in ("shuttle", "shuttlecock", "ball")
    }
    if not shuttle_cat_ids:
        print("Warning: no shuttle category found in COCO JSON")
        return

    # Build frame → annotation map
    frame_ann: dict[int, dict] = {}
    for ann in coco.get("annotations", []):
        if ann["category_id"] not in shuttle_cat_ids:
            continue
        img_id = ann["image_id"]
        frame_idx = id_to_idx.get(img_id, -1)
        if frame_idx < 0:
            continue
        # Take bbox centre as (X, Y)
        x, y, w, h = ann["bbox"]
        frame_ann[frame_idx] = {"X": x + w / 2, "Y": y + h / 2}

    # Build full CSV (all frames, visibility=0 when no shuttle annotation)
    total_frames = len(coco["images"])
    rows = []
    for frame_idx in range(total_frames):
        if frame_idx in frame_ann:
            rows.append({
                "Frame": frame_idx,
                "Visibility": 1,
                "X": round(frame_ann[frame_idx]["X"], 2),
                "Y": round(frame_ann[frame_idx]["Y"], 2),
            })
        else:
            rows.append({"Frame": frame_idx, "Visibility": 0, "X": 0, "Y": 0})

    out_path = os.path.join(out_dir, f"{rally_id}_ball.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote {len(rows)} rows → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-json", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--out-dir",   required=True)
    parser.add_argument("--rally-id",  default="1_01_00")
    args = parser.parse_args()
    convert(args.coco_json, args.image_dir, args.out_dir, args.rally_id)
```

After conversion, follow the TrackNetV3 training procedure from the official README:
1. Run `preprocess.py` (from V3 repo) to generate background median images
2. Fine-tune TrackNet: `python train.py --model_name TrackNet --seq_len 8 --epochs 15 --batch_size 10 --bg_mode concat --alpha 0.5 --save_dir exp`
3. Generate InpaintNet masks: `python generate_mask_data.py --tracknet_file exp/TrackNet_best.pt --batch_size 16`
4. Fine-tune InpaintNet (or skip and use pretrained): `python train.py --model_name InpaintNet --seq_len 16 --epochs 50 ...`

Copy final weights to `models/tracknetv3_tracknet.pt` and `models/tracknetv3_inpaintnet.pt`.

---

## Phase 3 — Fallback: V2 Improvements (if V3 integration fails)

Use this path **only if** V3 integration proves infeasible (e.g., dataset format conversion is not possible because training images lack temporal ordering). These changes improve V2 without touching V3.

### 3-A: Expand temporal context from 3 to 5 frames

**File: `models/TrackNet.py`**

Change the first `Conv` layer from 9 to 15 input channels:

```python
# In TrackNet.__init__, first encoder block:
self.conv1 = Conv(15, 64)   # was Conv(9, 64)
```

**File: `models/shuttle_tracknet.py`**

Change buffer size and input assembly:

```python
SEQ_LEN_V2_EXPANDED = 5   # was 3

# In TrackNetTracker.__init__:
self._buffer: list[tuple[float, np.ndarray]] = []

# In _make_input_tensor(), stack 5 frames × 3 channels = 15 channels
```

**Config:** `tracknet_buffer_size: 5` (was 3)

**Retraining required:** Yes. The first conv layer weight shape changes.  Expected accuracy improvement: ~2–3pp recall.

### 3-B: CBAM attention at UNet bottleneck

**File: `models/TrackNet.py`**

Add CBAM after the encoder bottleneck:

```python
class ChannelAttention(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, in_planes // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_planes // ratio, in_planes, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape
        avg = self.fc(self.avg_pool(x).view(B, C)).view(B, C, 1, 1)
        mx  = self.fc(self.max_pool(x).view(B, C)).view(B, C, 1, 1)
        return x * self.sigmoid(avg + mx)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        attn = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn


class CBAM(nn.Module):
    def __init__(self, in_planes: int):
        super().__init__()
        self.ca = ChannelAttention(in_planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))
```

In `TrackNet.__init__`, add after the encoder bottleneck conv:

```python
self.cbam_bottleneck = CBAM(512)   # applied at the 512-channel bottleneck
```

In `TrackNet.forward`, apply it at the bottleneck:

```python
x = self.conv_bottleneck(x)
x = self.cbam_bottleneck(x)        # ← insert here
x = self.upsample1(x)
```

**Parameter cost:** ~200K additional parameters. Inference overhead: ~3–5%. No change to input/output shapes or existing checkpoint format (weights are additive, not replacing).

### 3-C: Increase training resolution to 360×640

**File: `config.yaml`**

```yaml
tracknet_expected_h: 360    # was 288
tracknet_expected_w: 640    # was 512
```

**File: `models/TrackNet.py`** — no architectural changes needed; UNet is fully convolutional.

**File: `training/train_tracknet.py`** — ensure `ShuttleHeatmapDataset` reads the new resolution from config.

**Inference cost:** ~56% more pixels → ~40–60% slower per forward pass. With batching (batch=8) on GPU, this adds ~2–4ms amortized per frame. Still well within budget.

**Retraining required:** Yes. The Gaussian ground-truth heatmap sigma should scale with resolution:

```python
# In ShuttleHeatmapDataset, scale sigma proportionally:
sigma = 2.0 * (expected_h / 288.0)  # was fixed at 2.0
```

---

## Summary of changes by file

| File | Change | Phase |
|---|---|---|
| `models/shuttle_tracknet.py` | Fix `device` default: `str = "cpu"` → `Optional[str] = None` | P0 |
| `models/player_dino.py` | ViT-S/14 default; interval caching in `detect_yolo_compat`; FP16 in `forward_detect` | 1-A/B/C |
| `config.yaml` | `player_detect_interval: 3`; V3 config keys; threshold updates | 1-B, 2-D |
| `main.py` | Read `player_detect_interval` + call `set_detect_interval`; V3 construction block | 1-B, 2-D |
| `utils/background.py` | **New file**: background median estimator | 2-B |
| `models/tracknetv3_arch.py` | **New file**: copy `model.py` from TrackNetV3 repo verbatim | 2-C |
| `models/shuttle_tracknetv3.py` | **New file**: `TrackNetV3Tracker` wrapper | 2-C |
| `scripts/convert_coco_to_tracknetv3.py` | **New file**: COCO → V3 CSV converter | 2-E |
| `models/TrackNet.py` | (Fallback only) CBAM; 15-channel input; resolution | 3-A/B/C |
| `training/train_tracknet.py` | (Fallback only) Sigma scaling for new resolution | 3-C |

---

## Phase 4 — StrokeTransformer: Temporal Multi-Frame Pose Sequence

### State of existing `models/stroke.pt`

- **File exists** at `models/stroke.pt` (~1.1MB = ~283K params × 4 bytes ✓ — a real trained checkpoint).
- **Saved on a GPU node.** `torch.save()` records tensor device. The file has CUDA tensors, so loading it on a CPU-only machine (e.g. the `vscode1` login node) fails with `RuntimeError: Attempting to deserialize object on a CUDA device`. To inspect it on a CPU machine: `torch.load('models/stroke.pt', map_location='cpu')`.
- **Trained on single-frame sequences.** These weights were produced with the old `seq_len=1` path and are being discarded in favour of the multi-frame retrain below. Do not attempt to resume from them — start fresh.
- **Retrain is fast.** ~283K parameters, expected convergence in <30 minutes on a single OSCAR GPU.

---

### Background and design rationale

`StrokeClassifier` currently extracts MediaPipe Pose keypoints from **one frame only** — the frame on which `is_hit` fires — and feeds a flat 123-float vector to the StrokeTransformer. The Transformer receives a sequence of length 1, so its self-attention is the identity operation and it degenerates to a 2-layer MLP. The architecture is unused as designed.

The fix: collect **T frames around the hit** (default T=7: 3 before, the hit frame, 3 after), run MediaPipe on each, and feed a genuine pose sequence `(T, 99)` to the Transformer. Self-attention then models how joint positions evolve through the stroke — the pre-swing extension, contact point, and follow-through — which is highly discriminative across stroke types.

**Computational cost of multi-frame MediaPipe:**
- MediaPipe Pose `model_complexity=1` on CPU: ~30–40ms per frame
- ~60 hits per 2-minute match × 7 frames × 35ms ≈ 15 seconds total overhead
- Current pipeline: ~240 seconds (4 minutes) for a 2-minute video
- Overhead fraction: ~6% — acceptable

**Key constraint:** Pre-hit frames are gone by the time `is_hit` fires. The pipeline must maintain a sliding window buffer of raw BGR frames. Post-hit frames require deferring event emission by N frames.

---

### 4-A: Add `surrounding_frames` to `HitEvent`

**File: `core/tracking_types.py`**

Find the `HitEvent` dataclass and add the new field:

```python
@dataclass
class HitEvent:
    keyframe: np.ndarray                        # existing — frame at hit detection moment
    trajectory_pre: list                        # existing
    trajectory_post: list                       # existing
    surrounding_frames: list[np.ndarray] = field(default_factory=list)  # NEW — T BGR frames around hit
    surrounding_timestamps: list[float]  = field(default_factory=list)  # NEW — corresponding timestamps
```

Import `field` from `dataclasses` if not already imported at the top of the file.

---

### 4-B: Sliding frame window in `main.py`

**File: `main.py`**

Add these constants near the top of `run()`, before the frame loop:

```python
# ── Stroke surrounding-frame buffer ───────────────────────────────────────────
# Keeps a rolling window of recent raw frames so StrokeClassifier can access
# pre-hit frames after hit detection fires (hit detection is inherently delayed).
_STROKE_PRE_FRAMES  = int(getattr(cfg, "stroke_pre_frames",  3))   # frames before hit
_STROKE_POST_FRAMES = int(getattr(cfg, "stroke_post_frames", 3))   # frames after hit
_STROKE_WINDOW      = _STROKE_PRE_FRAMES + 1 + _STROKE_POST_FRAMES # total T

from collections import deque
_frame_window: deque[tuple[np.ndarray, int, float]] = deque(
    maxlen=_STROKE_PRE_FRAMES + 1   # only need to buffer pre-hit frames; post collected on the fly
)

# Pending hit events waiting for post-hit frames before being emitted
# Each entry: {"event_dict": dict, "frames_needed": int, "post_frames": list[np.ndarray]}
_pending_hits: list[dict] = []
```

Inside `_process_frame`, append the current frame to the window **before** the hit detection block:

```python
def _process_frame(frame_bgr, frame_idx, timestamp, shuttle_det_dict):
    nonlocal final_timestamp
    final_timestamp = timestamp

    # Keep rolling window of raw frames for multi-frame stroke pose extraction
    _frame_window.append((frame_bgr.copy(), frame_idx, timestamp))

    # ... rest of existing _process_frame logic ...
```

Replace the hit-event construction block (currently the `if is_hit:` block) with:

```python
if is_hit:
    # Collect pre-hit frames from the rolling window
    pre_frames = [f for f, _, _ in _frame_window]           # up to _STROKE_PRE_FRAMES + 1 frames
    pre_ts     = [t for _, _, t in _frame_window]

    hit_ev = HitEvent(
        keyframe=frame_bgr,
        trajectory_pre=[(e[0], e[1], e[2]) for e in hit_detector._buffer],
        trajectory_post=[],
        surrounding_frames=list(pre_frames),                 # pre-hit frames collected now
        surrounding_timestamps=list(pre_ts),
    )

    if _STROKE_POST_FRAMES > 0:
        # Defer emission: collect post-hit frames before classifying
        _pending_hits.append({
            "hit_ev":        hit_ev,
            "timestamp":     timestamp,
            "frame_idx":     frame_idx,
            "hit_player_id": hit_player_id,
            "frames_left":   _STROKE_POST_FRAMES,
        })
    else:
        # No post frames needed — classify immediately (original behaviour)
        _emit_stroke_event(hit_ev, timestamp, frame_idx, hit_player_id)

# Fulfil pending hits that are waiting for post-hit frames
still_pending = []
for pending in _pending_hits:
    pending["hit_ev"].surrounding_frames.append(frame_bgr.copy())
    pending["hit_ev"].surrounding_timestamps.append(timestamp)
    pending["frames_left"] -= 1
    if pending["frames_left"] <= 0:
        _emit_stroke_event(
            pending["hit_ev"],
            pending["timestamp"],
            pending["frame_idx"],
            pending["hit_player_id"],
        )
    else:
        still_pending.append(pending)
_pending_hits[:] = still_pending
```

Add the helper function `_emit_stroke_event` inside `run()`, before `_process_frame`:

```python
def _emit_stroke_event(hit_ev: HitEvent, timestamp: float, frame_idx: int, hit_player_id) -> None:
    """Classify and append a completed hit event (pre + post frames available)."""
    stroke_result = stroke_classifier.classify(hit_ev)
    event = {
        "timestamp":       timestamp,
        "frame_idx":       frame_idx,
        "player_id":       hit_player_id,
        "stroke_type":     stroke_result.get("stroke_type"),
        "confidence":      stroke_result.get("confidence", 0.0),
        "tactical_semantic": stroke_result.get("tactical_semantic"),
        "decision_eval":   stroke_result.get("decision_eval"),
        "trajectory_pre":  hit_ev.trajectory_pre,
        "trajectory_post": [],
        "prediction_error": None,
    }
    events.append(event)
```

After the main frame loop ends (after `_flush_batch()` / sequential path), flush any remaining pending hits:

```python
# Flush pending hits that reached end of video without enough post frames
for pending in _pending_hits:
    _emit_stroke_event(
        pending["hit_ev"],
        pending["timestamp"],
        pending["frame_idx"],
        pending["hit_player_id"],
    )
_pending_hits.clear()
```

**File: `config.yaml`** — add:

```yaml
# ─── Stroke classifier (multi-frame) ─────────────────────────────────────────
stroke_classify_enabled: true        # re-enable now that multi-frame is supported
stroke_pre_frames: 3                 # frames before hit to include in pose sequence
stroke_post_frames: 3                # frames after hit to include (defers event emission)
```

---

### 4-C: Multi-frame pose extraction in `StrokeClassifier`

**File: `models/stroke_classifier.py`**

Replace `_extract_pose_features` and `classify` with multi-frame–aware versions:

```python
def classify(self, hit_event) -> dict:
    """Classify a hit event using a temporal pose sequence around the hit.

    If surrounding_frames is populated (T > 1), runs MediaPipe on each frame
    and feeds the sequence to the Transformer. Falls back to keyframe-only
    if surrounding_frames is empty (backward compatible).
    """
    null_result = {
        "stroke_type": None, "confidence": 0.0,
        "tactical_semantic": None, "decision_eval": None,
    }
    if not self._enabled or self._model is None:
        return null_result

    from core.tracking_types import HitEvent
    if isinstance(hit_event, HitEvent):
        keyframe         = hit_event.keyframe
        traj_pre         = hit_event.trajectory_pre
        traj_post        = hit_event.trajectory_post
        surrounding      = hit_event.surrounding_frames      # list of BGR arrays, may be empty
    else:
        keyframe         = hit_event.get("keyframe")
        traj_pre         = hit_event.get("trajectory_pre", [])
        traj_post        = hit_event.get("trajectory_post", [])
        surrounding      = hit_event.get("surrounding_frames", [])

    if keyframe is None:
        return null_result

    import torch

    # ── Pose feature extraction ───────────────────────────────────────────
    frames_to_process = surrounding if surrounding else [keyframe]
    pose_seq = []
    for frame in frames_to_process:
        pose = self._extract_pose_features(frame)
        if pose is None:
            pose = np.zeros(self.pose_joints * 3, dtype=np.float32)
        pose_seq.append(pose)

    # pose_seq: list of T arrays each (99,) → stack to (T, 99)
    pose_tensor = torch.from_numpy(np.stack(pose_seq, axis=0)).unsqueeze(0)  # (1, T, 99)

    # ── Trajectory features (same for all frames — fixed at hit moment) ──
    traj_feats  = self._extract_trajectory_features(traj_pre, traj_post)     # (24,)
    traj_tensor = torch.from_numpy(traj_feats).unsqueeze(0).unsqueeze(0)    # (1, 1, 24)
    # Broadcast trajectory across all T frames
    T = pose_tensor.shape[1]
    traj_tensor = traj_tensor.expand(1, T, -1)                               # (1, T, 24)

    x = torch.cat([pose_tensor, traj_tensor], dim=-1)   # (1, T, 123)

    try:
        with torch.no_grad():
            logits = self._model(x)
            fa_logits = logits.get("foundational_actions")
            if fa_logits is None:
                return null_result
            probs = torch.softmax(fa_logits, dim=-1)
            conf, idx = probs.max(dim=-1)
            stroke_type = self.STROKE_TYPES[int(idx.item())]
            confidence  = float(conf.item())

            ts_logits = logits.get("tactical_semantic")
            tactical  = (self.TACTICAL_LABELS[int(torch.argmax(ts_logits, -1).item())]
                         if ts_logits is not None else None)

            de_logits = logits.get("decision_eval")
            decision  = (self.DECISION_LABELS[int(torch.argmax(de_logits, -1).item())]
                         if de_logits is not None else None)

        return {"stroke_type": stroke_type, "confidence": confidence,
                "tactical_semantic": tactical, "decision_eval": decision}
    except Exception as e:
        print(f"[STROKE] classify error: {e}")
        return null_result
```

---

### 4-D: Update `StrokeTransformer` to handle variable sequence length

**File: `models/stroke_transformer.py`**

The model already handles variable `seq_len` via `encoded.mean(dim=1)` (mean pooling). **No architecture change is needed.** The only change is that `input_dim` must match the per-frame feature dimension, which remains `123` (pose 99 + trajectory 24). The sequence dimension T now varies from 1 to 7 depending on how many frames were available — mean pooling handles this gracefully.

However, the model **must be retrained** with multi-frame sequences for the attention layers to learn temporal patterns. Until retraining, the existing weights will still function (they'll receive `(1, T, 123)` and mean-pool to the same `(1, 128)` representation they were trained on, treating all T frames equally). The model will not regress — it will still produce valid stroke type predictions — but it won't yet exploit temporal dynamics. Retraining is Phase 4-E.

---

### 4-E: Retrain `StrokeTransformer` on multi-frame sequences

The FineBadminton20k dataset provides per-event video clips, making it possible to extract multi-frame pose sequences for training. The training script `training/train_stroke.py` needs a dataset variant that returns `(T, 123)` tensors rather than `(123,)`.

**Approximate training effort:** StrokeTransformer has ~283K parameters. On 20k samples with T=7, training converges in <30 minutes on a single GPU. This is by far the smallest training job in the pipeline.

Expected improvement: stroke type accuracy should rise from the ~60–70% achievable with single-frame pose to ~80–90% with 7-frame temporal context, consistent with published results on sports action recognition with similar sequence lengths.

---

## Estimated performance after all changes

| Metric | Current | After P0 fix only | After P0 + Phase 1 | After P0 + Phase 1 + V3 |
|---|---|---|---|---|
| DINO per frame | 18ms (no caching) | 18ms | ~1.5ms amortized (ViT-S/14 + FP16 + interval=3) | same |
| TrackNet per frame | ~1,100ms (CPU!) | ~5–8ms (GPU, batched) | ~5–8ms | ~7–10ms |
| Total per frame | ~1,200ms | ~25ms | ~8ms | ~10ms |
| 100-frame wall time | ~120 sec | ~2.5 sec | ~0.8 sec | ~1.0 sec |
| Shuttle recall | ~94.6% (V2) | ~94.6% | ~94.6% | ~99.3% |

The Priority 0 fix alone recovers ~98% of the lost time. Everything else is incremental.
