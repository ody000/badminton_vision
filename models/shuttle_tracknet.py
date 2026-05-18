"""TrackNetTracker - port from slayminton/models/tracknet.py with timestamp-aware buffer.

Key additions vs slayminton:
- Timestamp-aware buffer: flush if gap > 2 * (1/fps) between frames
- set_fps() method
- Accepts cfg (SimpleNamespace) for config-driven init
"""

from __future__ import annotations

import os
import cv2
import numpy as np
import torch

from models.TrackNet import TrackNet


class TrackNetTracker:
    """Lightweight wrapper around TrackNet for a detect(frame, timestamp) API.

    Assumptions/behavior:
    - Maintains a timestamp-aware buffer of (timestamp, frame) tuples (maxlen 3).
    - If gap between consecutive timestamps > 2 * (1/fps), buffer is flushed.
    - Loads checkpoint by inspecting the final conv layer weight shape for out_channels.
    - Resizes input frames to (expected_h, expected_w) before inference.
    - Returns {"shuttle": (timestamp, x, y, w, h)} or {} if confidence < threshold.
    """

    def __init__(
        self,
        cfg=None,
        weights_path: str | None = None,
        device: str = "cpu",
        box_size: int = 16,
        conf_threshold: float = 0.001,
        expected_h: int = 288,
        expected_w: int = 512,
        fps: float = 30.0,
    ):
        # Read params from cfg if provided, use explicit args as overrides.
        if cfg is not None:
            weights_path = weights_path or getattr(cfg, "tracknet_weights", "models/tracknet.pt")
            device = device or getattr(cfg, "device", "cpu")
            box_size = int(getattr(cfg, "tracknet_box_size", box_size))
            conf_threshold = float(getattr(cfg, "tracknet_conf_threshold", conf_threshold))
            expected_h = int(getattr(cfg, "tracknet_expected_h", expected_h))
            expected_w = int(getattr(cfg, "tracknet_expected_w", expected_w))
            fps = float(getattr(cfg, "fps", fps))

        self.device = torch.device(
            device if isinstance(device, str) else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.box_size = int(box_size)
        self.conf_threshold = float(conf_threshold)
        self.expected_size = (int(expected_h), int(expected_w))  # (H, W)
        self.fps = float(fps)
        self._frame_count = 0

        # Timestamp-aware buffer: list of (timestamp, frame_rgb)
        self._buffer: list[tuple[float, np.ndarray]] = []

        # Determine out_channels from checkpoint.
        out_ch = 1
        state = None
        if weights_path and os.path.exists(weights_path):
            try:
                state = torch.load(weights_path, map_location="cpu")
                sd = state.get("state_dict", state) if isinstance(state, dict) else state
                for key in reversed(list(sd.keys())):
                    if isinstance(sd[key], torch.Tensor) and sd[key].ndim == 4:
                        out_ch = int(sd[key].shape[0])
                        break
                print(f"[TRACKNET] Loaded checkpoint suggests out_channels={out_ch}")
            except Exception as e:
                print(f"[TRACKNET] Warning: failed to inspect weights ({e}), using default out_channels=1")

        self.model = TrackNet(out_channels=out_ch)
        if state is not None:
            try:
                sd = state.get("state_dict", state) if isinstance(state, dict) else state
                self.model.load_state_dict(sd, strict=False)
                print(f"[TRACKNET] Loaded weights from {weights_path}")
            except Exception as e:
                print(f"[TRACKNET] Warning: failed to load weights ({e}), using random init")
        else:
            if weights_path:
                print(f"[TRACKNET] Warning: weights not found at {weights_path}; using random init")

        self.model.to(self.device).eval()

    def set_fps(self, fps: float) -> None:
        """Update the fps used for timestamp-gap flush detection."""
        self.fps = float(fps)

    def _flush_buffer(self) -> None:
        self._buffer.clear()

    def _maybe_flush_for_gap(self, timestamp: float) -> None:
        """Flush buffer if gap between new timestamp and last buffered is > 2*(1/fps)."""
        if not self._buffer:
            return
        last_ts = self._buffer[-1][0]
        gap_threshold = 2.0 / max(self.fps, 1e-6)
        if (timestamp - last_ts) > gap_threshold:
            self._flush_buffer()

    def _preprocess(self, frames: list[np.ndarray]) -> torch.Tensor:
        """Stack 3 RGB frames into 9-channel tensor."""
        arrays = []
        for f in frames:
            img = f.astype(np.float32) / 255.0
            chw = np.transpose(img, (2, 0, 1))  # H,W,C -> C,H,W
            arrays.append(chw)
        stacked = np.concatenate(arrays, axis=0)  # 9 x H x W
        return torch.from_numpy(stacked).unsqueeze(0).to(self.device)

    def _postprocess_heatmap(
        self, heatmap: np.ndarray, frame_w: int, frame_h: int
    ) -> tuple[int, int, int, int, float]:
        """Find argmax of heatmap and return bounding box + confidence.

        Returns (x0, y0, w, h, conf).
        """
        eh, ew = self.expected_size
        if heatmap.shape[0] != frame_h or heatmap.shape[1] != frame_w:
            heatmap = cv2.resize(heatmap, (frame_w, frame_h))
        _, maxv, _, maxloc = cv2.minMaxLoc(heatmap.astype(np.float32))
        cx, cy = int(maxloc[0]), int(maxloc[1])
        half = self.box_size // 2
        x0 = max(0, cx - half)
        y0 = max(0, cy - half)
        w = min(self.box_size, frame_w - x0)
        h = min(self.box_size, frame_h - y0)
        return x0, y0, w, h, float(maxv)

    def detect(self, frame: np.ndarray, timestamp: float = 0.0) -> dict:
        """Run TrackNet on one BGR frame and return detection dict.

        Args:
            frame: BGR frame (numpy uint8).
            timestamp: Frame timestamp in seconds.

        Returns:
            {"shuttle": (timestamp, x, y, w, h)} or {}.
        """
        self._frame_count += 1

        # Convert BGR to RGB for model
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]

        # Flush buffer if large timestamp gap
        self._maybe_flush_for_gap(timestamp)

        # Append to buffer
        self._buffer.append((timestamp, frame_rgb))
        if len(self._buffer) > 3:
            self._buffer.pop(0)

        # Build 3-frame input (replicate if not enough frames yet)
        if len(self._buffer) < 3:
            frames_for_model = [frame_rgb] * 3
        else:
            frames_for_model = [entry[1] for entry in self._buffer]

        # Resize to expected model resolution
        eh, ew = self.expected_size
        if (h, w) != (eh, ew):
            resized = [cv2.resize(f, (ew, eh)) for f in frames_for_model]
        else:
            resized = frames_for_model

        inp = self._preprocess(resized)
        with torch.no_grad():
            out = self.model(inp)
            out_np = out.squeeze(0).cpu().numpy()

        heat = out_np[0] if out_np.ndim == 3 else out_np

        x0, y0, bw, bh, conf = self._postprocess_heatmap(heat, w, h)

        if conf < self.conf_threshold:
            return {}

        return {
            "shuttle": (
                float(timestamp),
                float(x0),
                float(y0),
                float(bw),
                float(bh),
            )
        }
