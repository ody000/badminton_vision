#!/bin/bash
#SBATCH --job-name=badminton_dino_2player
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/train_2player_%j.log

# Two-player DINO detection training
# Ready for immediate submission: sbatch slurm_train_2player.sh

module load python/3.11
module load pytorch/2.0
module load cuda/11.8

set -e

WORK_DIR="/oscar/home/zshen38/badminton_vision"
cd "$WORK_DIR"

echo "[TRAIN] Starting two-player DINO training..."
echo "[TRAIN] PWD=$PWD"
echo "[TRAIN] CUDA check..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# Train with two-player architecture
# Dataset: _annotations.coco.json with 9911 images (96.6% have exactly 2 people)
# Output: models/dino_player_2player.pt ready for inference

python3 << 'TRAIN_SCRIPT'
import sys
sys.path.insert(0, '/oscar/home/zshen38/badminton_vision')

from models.player_dino import (
    DINODataset, DINOTracker, train_dino, EPOCHS, BATCH_SIZE, LEARNING_RATE
)
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("[TRAIN] Loading dataset...")
dataset = DINODataset(
    device=device,
    data_dir="data/player_dataset",
    annotations_file="data/player_dataset/_annotations.coco.json"
)
print(f"[TRAIN] Dataset size: {len(dataset)}")

print("[TRAIN] Creating model...")
model = DINOTracker(device=device, pretrained_backbone_path=None)

print("[TRAIN] Starting training...")
trained_model, history = train_dino(
    student=model,
    dataset=dataset,
    device=device,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    output_dir="data/output",
    checkpoint_name="dino_player_2player.pt",
    use_lora=True,
    lora_r=4,
    num_workers=4,
)

print("[TRAIN] Training complete!")
print(f"[TRAIN] Final checkpoint: data/output/dino_player_2player.pt")
print(f"[TRAIN] Final val_loss: {history.val_loss[-1]:.4f}")
if history.val_iou:
    print(f"[TRAIN] Final val_iou:  {history.val_iou[-1]:.4f}")
if history.val_map:
    print(f"[TRAIN] Final val_map:  {history.val_map[-1]:.4f}")

TRAIN_SCRIPT

echo "[TRAIN] ✓ Training finished at $(date)"
