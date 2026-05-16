#!/bin/bash
# SLURM launcher for badminton_vision training and inference.
#
# Modes: train-tracknet | train-yolo | train-stroke | run-main
#
# Default: 2 GPUs (Brown OSCAR cap per job).
# To use 1 GPU, pass --gres=gpu:1 AND NGPUS=1 together:
#
#   sbatch --export=MODE=train-tracknet slurm_train.sh
#   sbatch --export=MODE=train-tracknet,NGPUS=1 --gres=gpu:1 slurm_train.sh
#   sbatch --export=MODE=run-main,VIDEO_PATH=data/input/match.mp4 slurm_train.sh
#
# NOTE: #SBATCH directives are parsed as literal text — shell variables do NOT
# expand inside them.  Override --gres on the sbatch command line instead.

#SBATCH --job-name=badminton_vision
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

# ── Ensure log directory exists ───────────────────────────────────────────────
mkdir -p data/output/logs

# ── Activate environment ──────────────────────────────────────────────────────
if command -v uv &>/dev/null; then
    PYTHON="uv run python"
else
    PYTHON="python"
fi

echo "[SLURM] MODE=${MODE} NGPUS=${NGPUS} JOB_ID=${SLURM_JOB_ID}"
echo "[SLURM] Starting at $(date)"

# ── Dispatch ──────────────────────────────────────────────────────────────────
case "${MODE}" in

    train-tracknet)
        echo "[SLURM] Training TrackNet"
        torchrun \
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
        ;;

    train-yolo)
        echo "[SLURM] Training YOLO"
        torchrun \
            --nproc_per_node="${NGPUS}" \
            training/train_yolo.py \
            --config config.yaml \
            --data-dir "${DATA_DIR}" \
            --output-dir "${OUTPUT_DIR}" \
            --epochs "${EPOCHS}" \
            --batch-size "${BATCH_SIZE}" \
            --lr "${LR}" \
            --device "${DEVICE}"
        ;;

    train-stroke)
        echo "[SLURM] Training stroke classifier"
        # FINEBADMINTON_DIR defaults to /oscar/scratch/<user>/finebadminton20k.
        # Override at submission time:
        #   sbatch --export=MODE=train-stroke,FINEBADMINTON_DIR=/path/to/data slurm_train.sh
        _FB_DIR="${FINEBADMINTON_DIR:-/oscar/scratch/${USER}/finebadminton20k}"
        echo "[SLURM] FINEBADMINTON_DIR=${_FB_DIR}"
        torchrun \
            --nproc_per_node="${NGPUS}" \
            training/train_stroke.py \
            --config config.yaml \
            --data-dir "${_FB_DIR}" \
            --output-dir "${OUTPUT_DIR}" \
            --epochs "${EPOCHS:-30}" \
            --batch-size "${BATCH_SIZE}" \
            --lr "${LR}" \
            --device "${DEVICE}"
        ;;

    run-main)
        echo "[SLURM] Running main pipeline"
        ${PYTHON} main.py \
            --config config.yaml \
            --video "${VIDEO_PATH}" \
            --court-points "${COURT_POINTS}" \
            --output-dir "${OUTPUT_DIR}" \
            --device "${DEVICE}"
        ;;

    *)
        echo "[SLURM] ERROR: Unknown MODE '${MODE}'"
        echo "  Valid modes: train-tracknet | train-yolo | train-stroke | run-main"
        exit 1
        ;;
esac

echo "[SLURM] Finished at $(date)"
