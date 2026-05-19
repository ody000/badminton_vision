#!/bin/bash
# SLURM launcher for badminton_vision training.
#
# Modes: train-tracknet | train-yolo | train-stroke | train-dino
# For inference, use slurm_track.sh instead.
# Phase 4-E: train-stroke now supports multi-frame pose sequences (T=7 temporal context)
#
# Brown OSCAR GPU partitions:
#   gpu          — general GPU pool (V100/A100, up to 2 GPUs per job)
#   gpu-he       — high-end GPU (A100 80 GB), limited allocation
#   3090-gcondo  — RTX 3090 condo nodes (if you have allocation)
#
# Default: 2 GPUs on the 'gpu' partition (OSCAR cap per job).
# Common overrides:
#
#   # DINOv3 player detection (1 GPU, recommended)
#   sbatch --gres=gpu:1 --mem=64G \
#     --export=MODE=train-dino,TRAIN_DIR=data/input/train/player,EPOCHS=50,BATCH_SIZE=16,LR=5e-4 \
#     slurm_train.sh
#
#   # DINOv3 with LoRA fine-tuning (faster, lower memory)
#   sbatch --gres=gpu:1 --mem=48G \
#     --export=MODE=train-dino,TRAIN_DIR=data/input/train/player,EPOCHS=50,BATCH_SIZE=16,LR=5e-4,USE_LORA=1,LORA_R=4,LORA_ALPHA=8 \
#     slurm_train.sh
#
#   # 2 GPUs, default partition (TrackNet)
#   sbatch --export=MODE=train-tracknet slurm_train.sh
#
#   # Stroke classifier (multi-frame, Phase 4-E) — 1 GPU, ~30 minutes
#   sbatch -p gpu --gres=gpu:1 --mem=32G \
#     --export=MODE=train-stroke,NGPUS=1,EPOCHS=30,BATCH_SIZE=8,LR=3e-4,FINEBADMINTON_DIR=/users/$USER/scratch/finebadminton20k \
#     slurm_train.sh
#
# NOTE: #SBATCH directives are parsed as literal text — shell variables do NOT
# expand inside them.  Override --partition and --gres on the sbatch command
# line instead of editing these lines.

#SBATCH --job-name=badminton_vision
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH -o data/output/logs/slurm-%j.out
#SBATCH -e data/output/logs/slurm-%j.err
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8

# ── Defaults ──────────────────────────────────────────────────────────────────
MODE="${MODE:-train-tracknet}"
NGPUS="${NGPUS:-2}"   # must match --gres=gpu:N above (or the sbatch override)
DATA_DIR="${DATA_DIR:-data/input/train_mog_reflect}"
OUTPUT_DIR="${OUTPUT_DIR:-data/output}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-3e-4}"
DEVICE="${DEVICE:-cuda}"
VIDEO_PATH="${VIDEO_PATH:-data/input/match_clip.mp4}"
COURT_POINTS="${COURT_POINTS:-data/input/court_points.json}"
WEIGHTS="${WEIGHTS:-}"

# DINOv3 player detection training defaults
TRAIN_DIR="${TRAIN_DIR:-data/input/train/player}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-dino_player.pt}"
USE_LORA="${USE_LORA:-0}"
LORA_R="${LORA_R:-4}"
LORA_ALPHA="${LORA_ALPHA:-8}"
FREEZE_BACKBONE_EPOCHS="${FREEZE_BACKBONE_EPOCHS:-0}"
PRETRAINED_BACKBONE="${PRETRAINED_BACKBONE:-}"
LOG_EVERY="${LOG_EVERY:-10}"
NUM_WORKERS="${NUM_WORKERS:-4}"

# ── Ensure log directory exists ───────────────────────────────────────────────
mkdir -p data/output/logs

# ── OSCAR-specific CUDA setup ───────────────────────────────────────────────
# Load CUDA modules on compute node
module load cuda/11.8.0-kuhf cudnn/8.7.0.84-11.8-kff3

# ── Activate environment ──────────────────────────────────────────────────────
# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate badminton_train
PYTHON_CMD="python -u"

# Sanity-check: abort immediately if torch is missing rather than failing deep
# inside a torchrun subprocess with a cryptic error.
if ! ${PYTHON_CMD} -c "import torch" 2>/dev/null; then
    echo "[SLURM] ERROR: 'import torch' failed for PYTHON='${PYTHON_CMD}'"
    echo "  Attempting to install torch for CUDA 11.8..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --quiet
    if ! ${PYTHON_CMD} -c "import torch" 2>/dev/null; then
        echo "[SLURM] FATAL: 'import torch' still failed after install"
        exit 1
    fi
fi
echo "[SLURM] torch OK — $(${PYTHON_CMD} -c 'import torch; print(torch.__version__)')"
echo "[SLURM] CUDA available: $(${PYTHON_CMD} -c 'import torch; print(torch.cuda.is_available())')"

echo "[SLURM] MODE=${MODE} NGPUS=${NGPUS} JOB_ID=${SLURM_JOB_ID}"
echo "[SLURM] Starting at $(date)"

# ── Dispatch ──────────────────────────────────────────────────────────────────
case "${MODE}" in

    train-tracknet)
        echo "[SLURM] Training TrackNet"
        if [[ "${NGPUS}" -gt 1 ]]; then
            ${PYTHON_CMD} -m torch.distributed.run \
                --nproc_per_node="${NGPUS}" \
                training/train_tracknet.py \
                --config config.yaml \
                --data-dir "${DATA_DIR}" \
                --output-dir "${OUTPUT_DIR}" \
                --epochs "${EPOCHS}" \
                --batch-size "${BATCH_SIZE}" \
                --lr "${LR}" \
                --device "${DEVICE}" \
                --world-size "${NGPUS}" \
                ${WEIGHTS:+--weights "${WEIGHTS}"}
        else
            ${PYTHON_CMD} training/train_tracknet.py \
                --config config.yaml \
                --data-dir "${DATA_DIR}" \
                --output-dir "${OUTPUT_DIR}" \
                --epochs "${EPOCHS}" \
                --batch-size "${BATCH_SIZE}" \
                --lr "${LR}" \
                --device "${DEVICE}" \
                --world-size "${NGPUS}" \
                ${WEIGHTS:+--weights "${WEIGHTS}"}
        fi
        ;;

    train-yolo)
        echo "[SLURM] Training YOLO"
        if [[ "${NGPUS}" -gt 1 ]]; then
            ${PYTHON_CMD} -m torch.distributed.run \
                --nproc_per_node="${NGPUS}" \
                training/train_yolo.py \
                --config config.yaml \
                --data-dir "${DATA_DIR}" \
                --output-dir "${OUTPUT_DIR}" \
                --epochs "${EPOCHS}" \
                --batch-size "${BATCH_SIZE}" \
                --lr "${LR}" \
                --device "${DEVICE}"
        else
            ${PYTHON_CMD} training/train_yolo.py \
                --config config.yaml \
                --data-dir "${DATA_DIR}" \
                --output-dir "${OUTPUT_DIR}" \
                --epochs "${EPOCHS}" \
                --batch-size "${BATCH_SIZE}" \
                --lr "${LR}" \
                --device "${DEVICE}"
        fi
        ;;

    train-stroke)
        echo "[SLURM] Training stroke classifier"
        # FINEBADMINTON_DIR defaults to /oscar/scratch/<user>/finebadminton20k.
        # Override at submission time:
        #   sbatch --export=MODE=train-stroke,FINEBADMINTON_DIR=/path/to/data slurm_train.sh
        _FB_DIR="${FINEBADMINTON_DIR:-/oscar/scratch/${USER}/finebadminton20k}"
        echo "[SLURM] FINEBADMINTON_DIR=${_FB_DIR}"
        if [[ "${NGPUS}" -gt 1 ]]; then
            ${PYTHON_CMD} -m torch.distributed.run \
                --nproc_per_node="${NGPUS}" \
                training/train_stroke.py \
                --config config.yaml \
                --data-dir "${_FB_DIR}" \
                --output-dir "${OUTPUT_DIR}" \
                --epochs "${EPOCHS:-30}" \
                --batch-size "${BATCH_SIZE}" \
                --lr "${LR}" \
                --device "${DEVICE}"
        else
            ${PYTHON_CMD} training/train_stroke.py \
                --config config.yaml \
                --data-dir "${_FB_DIR}" \
                --output-dir "${OUTPUT_DIR}" \
                --epochs "${EPOCHS:-30}" \
                --batch-size "${BATCH_SIZE}" \
                --lr "${LR}" \
                --device "${DEVICE}"
        fi
        ;;

    train-dino)
        echo "[SLURM] Training DINOv3 player detector"
        echo "[SLURM]   Train dir:         ${TRAIN_DIR}"
        echo "[SLURM]   Output dir:        ${OUTPUT_DIR}"
        echo "[SLURM]   Epochs:            ${EPOCHS}"
        echo "[SLURM]   Batch size:        ${BATCH_SIZE}"
        echo "[SLURM]   Learning rate:     ${LR}"
        echo "[SLURM]   Use LoRA:          ${USE_LORA}"
        if [[ "${USE_LORA}" == "1" ]]; then
            echo "[SLURM]   LoRA r:            ${LORA_R}"
            echo "[SLURM]   LoRA alpha:        ${LORA_ALPHA}"
        fi
        echo "[SLURM]   Num workers:       ${NUM_WORKERS}"
        echo "[SLURM]   Log every:         ${LOG_EVERY}"

        # Verify training data exists
        if [[ ! -d "${TRAIN_DIR}" ]]; then
            echo "[SLURM] ERROR: TRAIN_DIR does not exist: ${TRAIN_DIR}"
            exit 1
        fi

        ANNOTATIONS_FILE="${TRAIN_DIR}/_annotations.coco.json"
        if [[ ! -f "${ANNOTATIONS_FILE}" ]]; then
            echo "[SLURM] ERROR: COCO annotations file not found: ${ANNOTATIONS_FILE}"
            exit 1
        fi

        # Create output directory
        mkdir -p "${OUTPUT_DIR}"

        # Build Python training script (inline to avoid shell escaping issues)
        ${PYTHON_CMD} << 'ENDPYTHON'
import sys
import os
from models.player_dino import DINODataset, train_dino

train_dir = os.environ.get("TRAIN_DIR", "data/input/train/player")
annotations_file = os.path.join(train_dir, "_annotations.coco.json")
output_dir = os.environ.get("OUTPUT_DIR", "data/output")
epochs = int(os.environ.get("EPOCHS", "50"))
batch_size = int(os.environ.get("BATCH_SIZE", "16"))
lr = float(os.environ.get("LR", "5e-4"))
checkpoint_name = os.environ.get("CHECKPOINT_NAME", "dino_player.pt")
use_lora = os.environ.get("USE_LORA", "0") == "1"
lora_r = int(os.environ.get("LORA_R", "4"))
lora_alpha = int(os.environ.get("LORA_ALPHA", "8"))
freeze_backbone_epochs = int(os.environ.get("FREEZE_BACKBONE_EPOCHS", "0"))
pretrained_backbone = os.environ.get("PRETRAINED_BACKBONE", "")
log_every = int(os.environ.get("LOG_EVERY", "10"))
num_workers = int(os.environ.get("NUM_WORKERS", "4"))

print(f"[TRAIN] Loading dataset from: {train_dir}")
dataset = DINODataset(
    device="cuda",
    data_dir=train_dir,
    annotations_file=annotations_file
)
print(f"[TRAIN] Loaded {len(dataset)} images")

print("[TRAIN] Starting DINOv3 training...")
model, history = train_dino(
    student=None,
    dataset=dataset,
    device="cuda",
    epochs=epochs,
    batch_size=batch_size,
    learning_rate=lr,
    output_dir=output_dir,
    checkpoint_name=checkpoint_name,
    pretrained_backbone_path=pretrained_backbone if pretrained_backbone else None,
    freeze_backbone_epochs=freeze_backbone_epochs,
    use_lora=use_lora,
    lora_r=lora_r,
    lora_alpha=lora_alpha,
    num_workers=num_workers,
    log_every=log_every,
)

print("[TRAIN] Training complete!")
print(f"[TRAIN] Checkpoint saved to: {os.path.join(output_dir, checkpoint_name)}")
ENDPYTHON
        ;;

    *)
        echo "[SLURM] ERROR: Unknown MODE '${MODE}'"
        echo "  Valid modes: train-tracknet | train-yolo | train-stroke | train-dino"
        echo "  For inference, use slurm_track.sh instead."
        exit 1
        ;;
esac

echo "[SLURM] Finished at $(date)"
