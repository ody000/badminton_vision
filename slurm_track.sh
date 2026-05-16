#!/bin/bash
# SLURM launcher for badminton_vision inference (single-process, no DDP).
#
# Usage:
#   sbatch --export=VIDEO_PATH=data/input/match.mp4 slurm_track.sh
#
# Required env vars:
#   VIDEO_PATH      — path to input video
#
# Optional env vars:
#   SHUTTLE_WEIGHTS — path to TrackNet checkpoint (default: weights/tracknet.pt)
#   COURT_POINTS    — path to court_points.json (default: data/input/court_points.json)
#   OUTPUT_DIR      — output directory (default: data/output)
#   DEVICE          — cpu / cuda (default: cuda)

#SBATCH --job-name=badminton_track
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --mem=16G
#SBATCH -o data/output/logs/slurm-%j.out
#SBATCH -e data/output/logs/slurm-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

# ── Defaults ──────────────────────────────────────────────────────────────────
VIDEO_PATH="${VIDEO_PATH:-data/input/match_clip.mp4}"
SHUTTLE_WEIGHTS="${SHUTTLE_WEIGHTS:-weights/tracknet.pt}"
COURT_POINTS="${COURT_POINTS:-data/input/court_points.json}"
OUTPUT_DIR="${OUTPUT_DIR:-data/output}"
DEVICE="${DEVICE:-cuda}"

# ── Ensure log directory exists ───────────────────────────────────────────────
mkdir -p data/output/logs

# ── Activate environment ──────────────────────────────────────────────────────
cd "$(dirname "$(realpath "$0")")"  # Ensure we're in project root
source .venv/bin/activate

# Use uv run with verbose and unbuffered output flags (like slayminton)
PYTHON="uv run -v python -u"

# Sanity-check: abort immediately if torch is missing rather than failing deep
# inside a subprocess with a cryptic error.
if ! ${PYTHON} -c "import torch" 2>/dev/null; then
    echo "[SLURM_TRACK] ERROR: 'import torch' failed for PYTHON='${PYTHON}'"
    echo "  Activate your conda/venv environment before submitting, or"
    echo "  uncomment the module load / conda activate lines above."
    exit 1
fi
echo "[SLURM_TRACK] torch OK — $(${PYTHON} -c 'import torch; print(torch.__version__)')"
echo "[SLURM_TRACK] CUDA available: $(${PYTHON} -c 'import torch; print(torch.cuda.is_available())')"

echo "[SLURM_TRACK] VIDEO_PATH=${VIDEO_PATH}"
echo "[SLURM_TRACK] SHUTTLE_WEIGHTS=${SHUTTLE_WEIGHTS}"
echo "[SLURM_TRACK] COURT_POINTS=${COURT_POINTS}"
echo "[SLURM_TRACK] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[SLURM_TRACK] DEVICE=${DEVICE}"
echo "[SLURM_TRACK] Starting at $(date)"

# ── Run inference ─────────────────────────────────────────────────────────────
${PYTHON} main.py \
    --config config.yaml \
    --video "${VIDEO_PATH}" \
    --court-points "${COURT_POINTS}" \
    --output-dir "${OUTPUT_DIR}" \
    --device "${DEVICE}" \
    --set tracknet_weights="${SHUTTLE_WEIGHTS}"

echo "[SLURM_TRACK] Finished at $(date)"
