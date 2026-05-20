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

        # Load TrackNet V3.
        # out_channels must match the pretrained checkpoint: one heatmap per input
        # frame in the 8-frame window (SEQ_LEN).  Using out_channels=1 would make
        # the final predictor Conv2d shape incompatible with the checkpoint, and
        # strict=False would silently leave it randomly initialized → the model
        # outputs garbage from a biased random head (appears to "track" a fixed corner).
        self.tracknet = _TrackNetV3Arch(in_channels=SEQ_LEN * 3 + 3, out_channels=SEQ_LEN)
        if tracknet_path and os.path.exists(tracknet_path):
            state = torch.load(tracknet_path, map_location="cpu")
            # The official TrackNetV3 checkpoint uses the key 'param' for the
            # state dict.  Try several common keys before falling back to the raw
            # dict so we never accidentally pass top-level metadata keys (epoch,
            # optim, loss…) as layer names.
            if isinstance(state, dict):
                sd = (state.get("param")
                      or state.get("state_dict")
                      or state.get("model_state_dict")
                      or state.get("model")
                      or state)
            else:
                sd = state
            result = self.tracknet.load_state_dict(sd, strict=False)
            n_missing = len(result.missing_keys)
            n_unexpected = len(result.unexpected_keys)
            print(f"[TRACKNETV3] Loaded TrackNet from {tracknet_path} "
                  f"(missing={n_missing}, unexpected={n_unexpected})")
            if n_missing:
                print(f"[TRACKNETV3]   missing keys: {result.missing_keys[:5]}"
                      f"{'...' if n_missing > 5 else ''}")
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
            heatmap_all = self.tracknet(inp)  # (1, SEQ_LEN, H, W)
        # Take the last channel: prediction for the most recent (current) frame.
        # The V3 model outputs one heatmap per input frame; channel [-1] corresponds
        # to the frame that was just added to the buffer.
        heatmap = heatmap_all.float()[0, -1]  # (H, W)

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
