"""StrokeTransformer: Transformer encoder + multi-head classification.

Architecture:
  - Linear projection: input_dim -> d_model (128)
  - TransformerEncoder: 2 layers, 4 heads, d_model=128
  - Three classification heads:
      foundational_actions: 8 classes (clear, drop, smash, net, drive, lift, lob, serve)
      tactical_semantic:    6 classes
      decision_eval:        3 classes (good, neutral, poor)

Input: (batch, input_dim) feature vector
Output: dict of logits tensors
"""

from __future__ import annotations

import torch
import torch.nn as nn


# Class counts
N_FOUNDATIONAL = 8   # clear, drop, smash, net, drive, lift, lob, serve
N_TACTICAL = 6
N_DECISION = 3       # good, neutral, poor


class StrokeTransformer(nn.Module):
    """Transformer-based stroke classifier with three output heads.

    Args:
        input_dim: Feature vector dimension = 33*3 + stroke_trajectory_n*2*2
        cfg: Optional config SimpleNamespace (reads stroke_trajectory_n, stroke_pose_joints).
    """

    def __init__(self, input_dim: int = 99 + 48, cfg=None):
        super().__init__()

        if cfg is not None:
            trajectory_n = int(getattr(cfg, "stroke_trajectory_n", 6))
            pose_joints = int(getattr(cfg, "stroke_pose_joints", 33))
            input_dim = pose_joints * 3 + trajectory_n * 2 * 2

        d_model = 128
        nhead = 4
        num_layers = 2
        dim_feedforward = 256
        dropout = 0.1

        self.input_proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification heads
        self.head_foundational = nn.Linear(d_model, N_FOUNDATIONAL)
        self.head_tactical = nn.Linear(d_model, N_TACTICAL)
        self.head_decision = nn.Linear(d_model, N_DECISION)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: (batch, input_dim) or (batch, seq_len, input_dim) feature tensor.
               If 2D, adds a sequence dimension of 1.

        Returns:
            dict with keys "foundational_actions", "tactical_semantic", "decision_eval".
            Each value is a (batch, n_classes) logits tensor.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, input_dim)

        projected = self.input_proj(x)   # (B, seq, d_model)
        encoded = self.encoder(projected)  # (B, seq, d_model)
        pooled = encoded.mean(dim=1)       # (B, d_model)

        return {
            "foundational_actions": self.head_foundational(pooled),
            "tactical_semantic": self.head_tactical(pooled),
            "decision_eval": self.head_decision(pooled),
        }
