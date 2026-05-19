"""Stroke classifier training on FineBadminton dataset.

Supports two data formats, auto-detected from the directory:

  HuggingFace format (FineBadminton20k, 43 GB):
    Directory contains dataset_info.json / *.parquet / *.arrow files
    as written by datasets.save_to_disk() or huggingface-cli download.
    Loaded with datasets.load_from_disk(data_dir).

  Custom JSON format (legacy / small test sets):
    data_dir/annotations.json with an "events" list, each event having
    "foundational_action" and optionally "features".

In both cases the three FineBadminton label levels are used:
  - foundational_actions (8 classes) — primary training target
  - tactical_semantic    (6 classes) — auxiliary head
  - decision_evaluation  (3 classes) — auxiliary head

Metrics: per-class accuracy, macro F1.

Usage:
    # HuggingFace dataset in /oscar/scratch
    python training/train_stroke.py \\
        --data-dir /oscar/scratch/$USER/finebadminton20k

    # Custom JSON
    python training/train_stroke.py \\
        --data-dir data/input/finebadminton
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


FOUNDATIONAL_ACTIONS = ["clear", "drop", "smash", "net", "drive", "lift", "lob", "serve"]
TACTICAL_LABELS      = ["attack", "defense", "neutral_tactic", "setup", "exploit", "unclear"]
DECISION_LABELS      = ["good", "neutral", "poor"]

ACTION_TO_IDX   = {a: i for i, a in enumerate(FOUNDATIONAL_ACTIONS)}
TACTICAL_TO_IDX = {t: i for i, t in enumerate(TACTICAL_LABELS)}
DECISION_TO_IDX = {d: i for i, d in enumerate(DECISION_LABELS)}


# ─────────────────────────────────────────────────────────────────────────────
# Format detection
# ─────────────────────────────────────────────────────────────────────────────

def _detect_format(data_dir: str) -> str:
    """Return 'finebadminton-hf' (per-video JSONs), 'huggingface', or 'json'.

    FineBadminton20k uses per-video JSON annotations in finebadminton-20K/*.json.
    HuggingFace datasets.save_to_disk() always writes dataset_info.json.
    Arrow / parquet files are also a reliable signal.
    """
    if not os.path.isdir(data_dir):
        return "json"
    contents = os.listdir(data_dir)

    # Check for FineBadminton20k per-video format (finebadminton-20K/*.json)
    finebadminton_subdir = os.path.join(data_dir, "finebadminton-20K")
    if os.path.isdir(finebadminton_subdir):
        json_files = [f for f in os.listdir(finebadminton_subdir) if f.endswith(".json")]
        if json_files:
            return "finebadminton-hf"

    # Standard HuggingFace signals
    hf_signals = {"dataset_info.json", "state.json"}
    if hf_signals & set(contents):
        return "huggingface"
    if any(f.endswith(".parquet") or f.endswith(".arrow") for f in contents):
        return "huggingface"
    # Check one level down (HF sometimes nests train/test splits as sub-dirs)
    for sub in contents:
        sub_path = os.path.join(data_dir, sub)
        if os.path.isdir(sub_path):
            sub_contents = os.listdir(sub_path)
            if hf_signals & set(sub_contents):
                return "huggingface"
            if any(f.endswith(".parquet") or f.endswith(".arrow") for f in sub_contents):
                return "huggingface"
    return "json"


# ─────────────────────────────────────────────────────────────────────────────
# FineBadminton20k per-video JSON format loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_finebadminton_hf(data_dir: str, split: str, cfg):
    """Load FineBadminton20k from per-video JSON format (finebadminton-20K/*.json).

    Structure:
      data_dir/finebadminton-20K/*.json (per-video annotations)
      data_dir/videos/*.mp4 (video files, optional for feature extraction)
      data_dir/annotations.json (optional, root-level annotations)

    Each per-video JSON contains hit events with stroke labels.
    """
    finebadminton_subdir = os.path.join(data_dir, "finebadminton-20K")
    json_files = sorted([
        f for f in os.listdir(finebadminton_subdir)
        if f.endswith(".json")
    ])

    feature_dim = (
        int(getattr(cfg, "stroke_pose_joints", 33)) * 3
        + int(getattr(cfg, "stroke_trajectory_n", 6)) * 2 * 2
    )

    samples = []
    skipped = 0

    for json_file in json_files:
        json_path = os.path.join(finebadminton_subdir, json_file)
        try:
            with open(json_path, "r") as f:
                video_data = json.load(f)
        except Exception as e:
            print(f"[TRAIN_STROKE] Warning: Could not load {json_file}: {e}")
            continue

        # Handle different JSON structures
        # Case 1: Direct list of hits
        if isinstance(video_data, list):
            hits = video_data
        # Case 2: Dict with "hits", "events", "rallies" keys
        elif isinstance(video_data, dict):
            hits = (
                video_data.get("hits") or
                video_data.get("events") or
                video_data.get("rallies") or
                []
            )
        else:
            hits = []

        for hit_idx, hit in enumerate(hits):
            # Extract stroke label (try multiple naming conventions)
            raw_label = (
                hit.get("foundational_action") or
                hit.get("stroke_type") or
                hit.get("action") or
                None
            )

            if raw_label is None:
                skipped += 1
                continue

            # Convert to index
            if isinstance(raw_label, int):
                label_idx = raw_label if raw_label < len(FOUNDATIONAL_ACTIONS) else -1
            else:
                label_idx = ACTION_TO_IDX.get(str(raw_label).lower().strip(), -1)

            if label_idx < 0:
                skipped += 1
                continue

            # Extract auxiliary labels
            ta_raw = hit.get("tactical_semantic") or hit.get("tactic")
            ta_idx = (TACTICAL_TO_IDX.get(str(ta_raw).lower().strip(), -1)
                     if isinstance(ta_raw, str) else
                     (ta_raw if isinstance(ta_raw, int) and ta_raw < len(TACTICAL_LABELS) else -1))

            de_raw = hit.get("decision_eval") or hit.get("decision")
            de_idx = (DECISION_TO_IDX.get(str(de_raw).lower().strip(), -1)
                     if isinstance(de_raw, str) else
                     (de_raw if isinstance(de_raw, int) and de_raw < len(DECISION_LABELS) else -1))

            # Store hit metadata for optional feature extraction
            # For now, use zero-filled features (features will be None)
            samples.append({
                "features":      None,  # Will be filled with zeros in __getitem__
                "label":         label_idx,
                "tactical":      ta_idx,
                "decision":      de_idx,
                "video_file":    json_file.replace(".json", ".mp4"),
                "hit_index":     hit_idx,
                "hit_data":      hit,  # Store full hit data for later feature extraction
            })

    if skipped:
        print(f"[TRAIN_STROKE] Skipped {skipped} hits with missing/invalid labels.")

    print(f"[TRAIN_STROKE] Loaded {len(samples)} hits from {len(json_files)} video JSONs")

    # Split into train/val
    n = len(samples)
    cutoff = int(n * 0.8)
    split_samples = samples[:cutoff] if split == "train" else samples[cutoff:]

    return split_samples, feature_dim


# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace dataset loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_hf_dataset(data_dir: str, split: str, cfg):
    """Load FineBadminton20k from a local HuggingFace-format directory.

    Column name mapping (tries several conventions the dataset may use):
      stroke/foundational_action/action → FOUNDATIONAL_ACTIONS index
      tactical_semantic/tactic          → TACTICAL_LABELS index
      decision_eval/decision            → DECISION_LABELS index
      features                          → pre-extracted float vector (optional)
    """
    try:
        from datasets import load_from_disk, DatasetDict
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required for HuggingFace format.\n"
            "Install with:  pip install datasets"
        )

    raw = load_from_disk(data_dir)

    # Handle DatasetDict (has named splits) vs. plain Dataset
    if isinstance(raw, DatasetDict):
        if split in raw:
            ds = raw[split]
        elif "train" in raw and split != "train":
            # fall back: split manually from train
            ds_train = raw["train"]
            n = len(ds_train)
            cutoff = int(n * 0.8)
            ds = ds_train.select(range(cutoff) if split == "train" else range(cutoff, n))
        else:
            ds = raw[list(raw.keys())[0]]
    else:
        # Plain Dataset — split manually
        n = len(raw)
        cutoff = int(n * 0.8)
        ds = raw.select(range(cutoff) if split == "train" else range(cutoff, n))

    print(f"[TRAIN_STROKE] HF dataset '{split}' split: {len(ds)} samples")
    print(f"[TRAIN_STROKE] Columns: {ds.column_names}")

    # Resolve column names flexibly
    col_names = set(ds.column_names)

    def _find_col(*candidates):
        for c in candidates:
            if c in col_names:
                return c
        return None

    fa_col  = _find_col("foundational_action", "stroke", "action", "label", "stroke_type")
    ts_col  = _find_col("tactical_semantic", "tactic", "tactical")
    de_col  = _find_col("decision_eval", "decision_evaluation", "decision")
    feat_col = _find_col("features", "feature_vector", "pose_features")

    if fa_col is None:
        raise ValueError(
            f"[TRAIN_STROKE] Cannot find a foundational-action column in {col_names}.\n"
            "Expected one of: foundational_action, stroke, action, label, stroke_type"
        )

    feature_dim = (
        int(getattr(cfg, "stroke_pose_joints", 33)) * 3
        + int(getattr(cfg, "stroke_trajectory_n", 6)) * 2 * 2
    )

    samples = []
    skipped = 0
    for row in ds:
        raw_label = row[fa_col]
        # Label may be int (already encoded) or string
        if isinstance(raw_label, int):
            label_idx = raw_label if raw_label < len(FOUNDATIONAL_ACTIONS) else -1
        else:
            label_idx = ACTION_TO_IDX.get(str(raw_label).lower().strip(), -1)
        if label_idx < 0:
            skipped += 1
            continue

        feats = row.get(feat_col) if feat_col else None

        ta_raw = row.get(ts_col) if ts_col else None
        ta_idx = (TACTICAL_TO_IDX.get(str(ta_raw).lower().strip(), -1)
                  if isinstance(ta_raw, str) else
                  (ta_raw if isinstance(ta_raw, int) and ta_raw < len(TACTICAL_LABELS) else -1))

        de_raw = row.get(de_col) if de_col else None
        de_idx = (DECISION_TO_IDX.get(str(de_raw).lower().strip(), -1)
                  if isinstance(de_raw, str) else
                  (de_raw if isinstance(de_raw, int) and de_raw < len(DECISION_LABELS) else -1))

        samples.append({
            "features": feats,
            "label":    label_idx,
            "tactical": ta_idx,
            "decision": de_idx,
        })

    if skipped:
        print(f"[TRAIN_STROKE] Skipped {skipped} rows with unrecognised action labels.")

    return samples, feature_dim


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────

class FineBadmintonDataset(Dataset):
    """Unified dataset for both HuggingFace and custom JSON formats.

    Auto-detects format from the directory; no flag needed.
    """

    def __init__(self, data_dir: str, cfg, split: str = "train"):
        fmt = _detect_format(data_dir)
        print(f"[TRAIN_STROKE] Detected format: '{fmt}' in {data_dir}")

        self.feature_dim = (
            int(getattr(cfg, "stroke_pose_joints", 33)) * 3
            + int(getattr(cfg, "stroke_trajectory_n", 6)) * 2 * 2
        )

        if fmt == "finebadminton-hf":
            self.samples, self.feature_dim = _load_finebadminton_hf(data_dir, split, cfg)
        elif fmt == "huggingface":
            self.samples, self.feature_dim = _load_hf_dataset(data_dir, split, cfg)
        else:
            self.samples = self._load_json(data_dir, split, cfg)

    def _load_json(self, data_dir: str, split: str, cfg) -> list[dict]:
        ann_path = os.path.join(data_dir, "annotations.json")
        samples: list[dict] = []

        if os.path.exists(ann_path):
            with open(ann_path, "r") as f:
                data = json.load(f)
            events = data.get("events", data) if isinstance(data, dict) else data
            for ev in events:
                label = ev.get("foundational_action", "").lower()
                if label in ACTION_TO_IDX:
                    samples.append({
                        "features": ev.get("features"),
                        "label":    ACTION_TO_IDX[label],
                        "tactical": TACTICAL_TO_IDX.get(
                            ev.get("tactical_semantic", "").lower(), -1),
                        "decision": DECISION_TO_IDX.get(
                            ev.get("decision_eval", "").lower(), -1),
                    })
        else:
            print(
                f"[TRAIN_STROKE] annotations.json not found in {data_dir} "
                "and directory is not a HuggingFace dataset — using synthetic data."
            )
            for _ in range(200):
                samples.append({
                    "features": None,
                    "label":    int(np.random.randint(0, len(FOUNDATIONAL_ACTIONS))),
                    "tactical": -1,
                    "decision": -1,
                })

        n = len(samples)
        cutoff = int(n * 0.8)
        return samples[:cutoff] if split == "train" else samples[cutoff:]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        feats  = sample.get("features")
        if feats is not None:
            x = np.array(feats, dtype=np.float32)
            if len(x) < self.feature_dim:
                x = np.pad(x, (0, self.feature_dim - len(x)))
            else:
                x = x[: self.feature_dim]
        else:
            x = np.zeros(self.feature_dim, dtype=np.float32)

        return {
            "features": torch.from_numpy(x),
            "label":    torch.tensor(sample["label"],    dtype=torch.long),
            "tactical": torch.tensor(max(sample["tactical"], 0), dtype=torch.long),
            "decision": torch.tensor(max(sample["decision"], 0), dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_class_accuracy(preds: list, labels: list) -> dict[str, float]:
    preds_arr  = np.array(preds)
    labels_arr = np.array(labels)
    per_class  = {}
    for i, name in enumerate(FOUNDATIONAL_ACTIONS):
        mask = labels_arr == i
        if mask.sum() == 0:
            continue
        per_class[name] = float((preds_arr[mask] == i).mean())
    return per_class


def compute_macro_f1(preds: list, labels: list, n_classes: int) -> float:
    preds_arr  = np.array(preds)
    labels_arr = np.array(labels)
    f1s = []
    for c in range(n_classes):
        tp = int(((preds_arr == c) & (labels_arr == c)).sum())
        fp = int(((preds_arr == c) & (labels_arr != c)).sum())
        fn = int(((preds_arr != c) & (labels_arr == c)).sum())
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1   = 2 * prec * rec / (prec + rec + 1e-9)
        f1s.append(f1)
    return float(np.mean(f1s))


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stroke classifier training")
    parser.add_argument(
        "--data-dir",
        default="data/input/finebadminton",
        help=(
            "Path to FineBadminton data. Accepts either a HuggingFace dataset "
            "directory (datasets.save_to_disk output, e.g. "
            "/oscar/scratch/$USER/finebadminton20k) or a directory containing "
            "annotations.json."
        ),
    )
    parser.add_argument("--output-dir",  default="data/output")
    parser.add_argument("--epochs",      type=int,   default=30)
    parser.add_argument("--batch-size",  type=int,   default=16)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--device",      default=None)
    parser.add_argument("--config",      default="config.yaml")
    args = parser.parse_args()

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from utils.config_loader import load_config
    from models.stroke_transformer import StrokeTransformer

    cfg        = load_config(args.config)
    device_str = args.device or getattr(cfg, "device", "cpu")
    device     = torch.device(device_str)

    train_ds   = FineBadmintonDataset(args.data_dir, cfg, split="train")
    val_ds     = FineBadmintonDataset(args.data_dir, cfg, split="val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    model     = StrokeTransformer(input_dim=train_ds.feature_dim, cfg=cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    weights_out = "models/stroke.pt"
    best_f1 = 0.0

    for epoch in range(1, args.epochs + 1):
        # ── Train ──────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        n_batches  = 0
        for batch in train_loader:
            x = batch["features"].to(device)
            y = batch["label"].to(device)
            optimizer.zero_grad()
            logits = model(x)["foundational_actions"]
            loss   = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches  += 1

        # ── Validate ───────────────────────────────────────────────────
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["features"].to(device)
                y = batch["label"].tolist()
                logits = model(x)["foundational_actions"]
                preds  = logits.argmax(dim=-1).tolist()
                all_preds.extend(preds)
                all_labels.extend(y)

        macro_f1  = compute_macro_f1(all_preds, all_labels, len(FOUNDATIONAL_ACTIONS))
        per_class = compute_class_accuracy(all_preds, all_labels)

        print(
            f"[TRAIN_STROKE] epoch={epoch}/{args.epochs}  "
            f"loss={train_loss / max(n_batches, 1):.4f}  "
            f"macro_F1={macro_f1:.4f}  "
            f"per_class={per_class}"
        )

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save({"epoch": epoch, "state_dict": model.state_dict()}, weights_out)
            print(f"[TRAIN_STROKE] Saved best weights (F1={best_f1:.4f}) to {weights_out}")

    print(f"[TRAIN_STROKE] Training complete. Best macro F1={best_f1:.4f}")


if __name__ == "__main__":
    main()
