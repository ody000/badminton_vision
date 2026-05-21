#!/bin/bash
# SLURM launcher for DINO diagnostic visualization
# Runs inference on random frames and generates annotated visualizations

#SBATCH --job-name=dino_diag
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH -o data/output/logs/slurm-%j.out
#SBATCH -e data/output/logs/slurm-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

# ── Defaults ──────────────────────────────────────────────────────────────────
NUM_FRAMES="${NUM_FRAMES:-100}"
DATA_DIR="${DATA_DIR:-data/input/train/player2}"
OUTPUT_DIR="${OUTPUT_DIR:-data/output/dino_diag}"
CHECKPOINT="${CHECKPOINT:-models/dino_player.pt}"

# ── Ensure log directory exists ───────────────────────────────────────────────
mkdir -p data/output/logs

# ── OSCAR-specific CUDA setup ───────────────────────────────────────────────
module load cuda/11.8.0-kuhf cudnn/8.7.0.84-11.8-kff3

# ── Activate environment ──────────────────────────────────────────────────────
eval "$(conda shell.bash hook)"
conda activate badminton_train
PYTHON_CMD="python -u"

echo "[SLURM_DIAG] torch version: $(${PYTHON_CMD} -c 'import torch; print(torch.__version__)')"
echo "[SLURM_DIAG] CUDA available: $(${PYTHON_CMD} -c 'import torch; print(torch.cuda.is_available())')"
echo "[SLURM_DIAG] Starting at $(date)"

echo "[SLURM_DIAG] NUM_FRAMES=${NUM_FRAMES}"
echo "[SLURM_DIAG] DATA_DIR=${DATA_DIR}"
echo "[SLURM_DIAG] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[SLURM_DIAG] CHECKPOINT=${CHECKPOINT}"

# ── Run diagnostic ────────────────────────────────────────────────────────────
${PYTHON_CMD} tools/dino_diagnostic.py \
    --num-frames "${NUM_FRAMES}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --checkpoint "${CHECKPOINT}"

echo "[SLURM_DIAG] Finished at $(date)"
