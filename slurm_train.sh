#!/bin/bash
# SLURM launcher for badminton_vision training and inference.
#
# Modes: train-tracknet | train-yolo | train-stroke | run-main
#
# Brown OSCAR GPU partitions:
#   gpu          — general GPU pool (V100/A100, up to 2 GPUs per job)
#   gpu-he       — high-end GPU (A100 80 GB), limited allocation
#   3090-gcondo  — RTX 3090 condo nodes (if you have allocation)
#
# Default: 2 GPUs on the 'gpu' partition (OSCAR cap per job).
# Common overrides:
#
#   # 2 GPUs, default partition
#   sbatch --export=MODE=train-tracknet slurm_train.sh
#
#   # 1 GPU (faster queue time for stroke classifier)
#   sbatch -p gpu --gres=gpu:1 --export=MODE=train-stroke,NGPUS=1,FINEBADMINTON_DIR=/oscar/scratch/$USER/finebadminton20k slurm_train.sh
#
#   # Inference only (no GPU needed beyond 1)
#   sbatch -p gpu --gres=gpu:1 --export=MODE=run-main,VIDEO_PATH=data/input/match_clip.mp4 slurm_train.sh
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

# ── Ensure log directory exists ───────────────────────────────────────────────
mkdir -p data/output/logs

# ── Activate environment ──────────────────────────────────────────────────────
# OSCAR: load CUDA + cuDNN modules, then activate your conda/venv.
# Uncomment and edit the lines that match your setup:
#
#   module load cuda/11.8.0 cudnn/8.6.0          # adjust versions as needed
#   module load python/3.10.12                    # if using OSCAR module python
#
source ~/miniconda3/etc/profile.d/conda.sh    # conda (most common on OSCAR)
conda activate /users/zshen38/ulg_new_env                      # ← your env name here
#
#   source .venv/bin/activate                     # plain venv alternative

# After activation, set PYTHON to whatever interpreter has torch installed.
# uv run python is tried first; falls back to plain python.
if command -v uv &>/dev/null; then
    PYTHON="uv run python"
else
    PYTHON="python"
fi

# Sanity-check: abort immediately if torch is missing rather than failing deep
# inside a torchrun subprocess with a cryptic error.
if ! ${PYTHON} -c "import torch" 2>/dev/null; then
    echo "[SLURM] ERROR: 'import torch' failed for PYTHON='${PYTHON}'"
    echo "  Activate your conda/venv environment before submitting, or"
    echo "  uncomment the module load / conda activate lines above."
    exit 1
fi
echo "[SLURM] torch OK — $(${PYTHON} -c 'import torch; print(torch.__version__)')"
echo "[SLURM] CUDA available: $(${PYTHON} -c 'import torch; print(torch.cuda.is_available())')"

echo "[SLURM] MODE=${MODE} NGPUS=${NGPUS} JOB_ID=${SLURM_JOB_ID}"
echo "[SLURM] Starting at $(date)"

# ── Dispatch ──────────────────────────────────────────────────────────────────
case "${MODE}" in

    train-tracknet)
        echo "[SLURM] Training TrackNet"
        ${PYTHON} -m torch.distributed.run \
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
        ${PYTHON} -m torch.distributed.run \
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
        ${PYTHON} -m torch.distributed.run \
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
