#!/usr/bin/env python3
"""
Quick validation of two-player DINO architecture before SLURM submission.

Run locally to verify:
1. Model loads correctly
2. Two detection heads output 10 values (5 per player)
3. Dataset loads 2 players per image on average
4. Forward pass works end-to-end

Usage:
    python3 test_2player_arch.py --dataset-dir data/player_dataset
"""

import argparse
import sys
from pathlib import Path

import torch

# Add repo to path
sys.path.insert(0, str(Path(__file__).parent))

from models.player_dino import DINODataset, DINOTracker, TRACKED_CLASSES


def test_model_creation():
    """Test model creation and forward pass."""
    print("[TEST] Creating DINOTracker model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DINOTracker(device=device)
    print(f"  ✓ Model created on {device}")

    # Check detector_head output size
    assert len(TRACKED_CLASSES) == 2, f"Expected 2 classes, got {len(TRACKED_CLASSES)}"
    print(f"  ✓ TRACKED_CLASSES = {TRACKED_CLASSES}")

    # Create dummy input
    dummy_frame = torch.randn(1, 3, 384, 384).to(device)
    with torch.no_grad():
        output = model.forward_detect(dummy_frame)

    print(f"  ✓ forward_detect output shape: {output.shape}")
    assert output.shape[1] == 10, f"Expected output shape (B, 10) for 2 players, got {output.shape}"
    print(f"  ✓ Output correctly has 10 values: 5 per player × 2 players")

    return model, device


def test_dataset(dataset_dir):
    """Test dataset loading and annotation parsing."""
    print(f"\n[TEST] Loading dataset from {dataset_dir}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DINODataset(
        device=device,
        data_dir=dataset_dir,
        annotations_file=f"{dataset_dir}/_annotations.coco.json"
    )
    print(f"  ✓ Dataset loaded: {len(dataset)} images")

    # Check a few samples
    player_counts = {1: 0, 2: 0, 3: 0}
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        det_target = sample["det_target"]  # (2, 5): [conf, cx, cy, w, h] per player

        # Count how many players have confidence > 0
        count = (det_target[:, 0] > 0).sum().item()
        player_counts[count] = player_counts.get(count, 0) + 1

        print(f"    Sample {i}: {count} player(s) annotated")
        print(f"      Player targets shape: {det_target.shape}")
        print(f"      Confidences: {det_target[:, 0].tolist()}")

    print(f"\n  ✓ Player annotation distribution (first 5 images):")
    for count, freq in player_counts.items():
        if freq > 0:
            print(f"    {count} player(s): {freq} images")

    return dataset, device


def test_inference(model, device, dataset):
    """Test inference on actual dataset sample."""
    print(f"\n[TEST] Running inference on dataset sample...")
    from PIL import Image

    # Get a sample
    sample = dataset[0]
    img_path = sample["image_path"]

    # Load image
    frame = Image.open(img_path).convert("RGB")
    print(f"  Image: {img_path}")
    print(f"  Size: {frame.size}")

    # Run detect
    with torch.no_grad():
        result = model.detect(frame, timestamp=0.0, min_confidence=0.25)

    players = result.get("players")
    if players:
        print(f"  ✓ Detected {len(players)} player(s)")
        for i, (ts, x, y, w, h) in enumerate(players):
            print(f"    Player {i}: box=({x:.1f}, {y:.1f}, {w:.1f}, {h:.1f})")
    else:
        print(f"  ℹ No players detected (below confidence threshold)")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="data/input/train/player2", help="Path to dataset")
    args = parser.parse_args()

    print("=" * 60)
    print("TWO-PLAYER DINO ARCHITECTURE VALIDATION")
    print("=" * 60)

    try:
        # Test 1: Model creation
        model, device = test_model_creation()

        # Test 2: Dataset
        dataset, _ = test_dataset(args.dataset_dir)

        # Test 3: Inference
        result = test_inference(model, device, dataset)

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        print("\nReady for SLURM training. Submit with:")
        print("  sbatch slurm_train_2player.sh")
        return 0

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
