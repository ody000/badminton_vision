"""StrokeClassifier: MediaPipe Pose + shuttle trajectory -> stroke type.

Uses a StrokeTransformer model (Transformer encoder + classification heads).
If weights file does not exist, classify() returns null fields gracefully.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np


class StrokeClassifier:
    """Classify badminton strokes from pose keyframe and shuttle trajectory.

    Args:
        cfg: SimpleNamespace from load_config().
    """

    # Stroke type labels (foundational actions)
    STROKE_TYPES = ["clear", "drop", "smash", "net", "drive", "lift", "lob", "serve"]
    TACTICAL_LABELS = ["attack", "defense", "neutral_tactic", "setup", "exploit", "unclear"]
    DECISION_LABELS = ["good", "neutral", "poor"]

    def __init__(self, cfg=None):
        if cfg is not None:
            self.trajectory_n = int(getattr(cfg, "stroke_trajectory_n", 6))
            self.pose_joints = int(getattr(cfg, "stroke_pose_joints", 33))
            weights_path = getattr(cfg, "stroke_weights", "weights/stroke_classifier.pt")
        else:
            self.trajectory_n = 6
            self.pose_joints = 33
            weights_path = "weights/stroke_classifier.pt"

        # Feature dimension: 33*3 (pose joints x,y,visibility) + trajectory pre+post
        self.feature_dim = self.pose_joints * 3 + self.trajectory_n * 2 * 2

        self._model = None
        self._mp_pose = None
        self._mp_pose_obj = None

        # Initialize MediaPipe Pose
        try:
            import mediapipe as mp
            self._mp = mp
            self._mp_pose = mp.solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=1,
            )
        except ImportError:
            print("[STROKE] MediaPipe not available; pose features will be zero-filled.")
            self._mp_pose = None

        # Load transformer weights if available
        if os.path.exists(weights_path):
            try:
                import torch
                from models.stroke_transformer import StrokeTransformer

                device = "cpu"
                if cfg is not None:
                    device = getattr(cfg, "device", "cpu")

                model = StrokeTransformer(
                    input_dim=self.feature_dim,
                    cfg=cfg,
                )
                state = torch.load(weights_path, map_location=device)
                sd = state.get("state_dict", state) if isinstance(state, dict) else state
                model.load_state_dict(sd, strict=False)
                model.eval()
                self._model = model
                self._device = device
                print(f"[STROKE] Loaded weights from {weights_path}")
            except Exception as e:
                print(f"[STROKE] Could not load weights ({e}); running in null mode.")
        else:
            print(f"[STROKE] Weights not found at {weights_path}; running in null mode.")

    def classify(self, hit_event) -> dict:
        """Classify a hit event into stroke type and tactical labels.

        Args:
            hit_event: HitEvent dataclass (preferred) or legacy dict with keys
                "keyframe", "trajectory_pre", "trajectory_post".

        Returns:
            dict with keys: stroke_type, confidence, tactical_semantic, decision_eval
        """
        null_result = {
            "stroke_type": None,
            "confidence": 0.0,
            "tactical_semantic": None,
            "decision_eval": None,
        }

        # Accept both HitEvent dataclass and legacy dict
        from core.tracking_types import HitEvent
        if isinstance(hit_event, HitEvent):
            keyframe   = hit_event.keyframe
            traj_pre   = hit_event.trajectory_pre
            traj_post  = hit_event.trajectory_post
        else:
            keyframe   = hit_event.get("keyframe")
            traj_pre   = hit_event.get("trajectory_pre", [])
            traj_post  = hit_event.get("trajectory_post", [])

        if keyframe is None:
            return null_result

        if self._model is None:
            return null_result

        import torch

        # Extract features
        pose_feats = self._extract_pose_features(keyframe)
        if pose_feats is None:
            pose_feats = np.zeros(self.pose_joints * 3, dtype=np.float32)
        traj_feats  = self._extract_trajectory_features(traj_pre, traj_post)

        feature_vec = np.concatenate([pose_feats, traj_feats]).astype(np.float32)
        x = torch.from_numpy(feature_vec).unsqueeze(0)  # (1, feature_dim)

        try:
            with torch.no_grad():
                logits = self._model(x)
                # logits["foundational_actions"]: (1, 8)
                fa_logits = logits.get("foundational_actions")
                if fa_logits is None:
                    return null_result

                probs = torch.softmax(fa_logits, dim=-1)
                conf, idx = probs.max(dim=-1)
                stroke_type = self.STROKE_TYPES[int(idx.item())]
                confidence = float(conf.item())

                # Tactical semantic
                ts_logits = logits.get("tactical_semantic")
                if ts_logits is not None:
                    ts_idx = int(torch.argmax(ts_logits, dim=-1).item())
                    tactical = self.TACTICAL_LABELS[ts_idx] if ts_idx < len(self.TACTICAL_LABELS) else None
                else:
                    tactical = None

                # Decision eval
                de_logits = logits.get("decision_eval")
                if de_logits is not None:
                    de_idx = int(torch.argmax(de_logits, dim=-1).item())
                    decision = self.DECISION_LABELS[de_idx] if de_idx < len(self.DECISION_LABELS) else None
                else:
                    decision = None

            return {
                "stroke_type": stroke_type,
                "confidence": confidence,
                "tactical_semantic": tactical,
                "decision_eval": decision,
            }
        except Exception as e:
            print(f"[STROKE] classify error: {e}")
            return null_result

    def _extract_pose_features(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Run MediaPipe Pose on a BGR frame, return flat array of 33*(x,y,visibility).

        Returns None if no pose detected.
        """
        if self._mp_pose is None:
            return None

        import cv2
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._mp_pose.process(frame_rgb)

        if not results.pose_landmarks:
            return None

        landmarks = results.pose_landmarks.landmark
        feats = np.array(
            [[lm.x, lm.y, lm.visibility] for lm in landmarks],
            dtype=np.float32,
        ).flatten()  # 33*3 = 99 floats
        return feats

    def _extract_trajectory_features(
        self,
        traj_pre: list,
        traj_post: list,
    ) -> np.ndarray:
        """Flatten pre+post (x,y) pairs, zero-pad to fixed length.

        Returns flat array of trajectory_n * 2 * 2 floats.
        """
        n = self.trajectory_n
        total_floats = n * 2 * 2  # pre: n*(x,y), post: n*(x,y)

        def extract_xy(traj, max_n):
            out = np.zeros((max_n, 2), dtype=np.float32)
            for i, entry in enumerate(traj[:max_n]):
                if len(entry) >= 3:
                    out[i, 0] = float(entry[1])
                    out[i, 1] = float(entry[2])
                elif len(entry) >= 2:
                    out[i, 0] = float(entry[0])
                    out[i, 1] = float(entry[1])
            return out

        pre_arr = extract_xy(traj_pre, n)
        post_arr = extract_xy(traj_post, n)
        return np.concatenate([pre_arr.flatten(), post_arr.flatten()])
