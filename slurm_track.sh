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
#   SHUTTLE_WEIGHTS — path to TrackNet checkpoint (default: models/tracknet.pt)
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
# TRACKNET_VERSION: 2 or 3 (default 3).  Must match config.yaml tracknet_version.
TRACKNET_VERSION="${TRACKNET_VERSION:-3}"
# V2 weight override (only used when TRACKNET_VERSION=2):
SHUTTLE_WEIGHTS="${SHUTTLE_WEIGHTS:-models/tracknet.pt}"
# V3 weight overrides (only used when TRACKNET_VERSION=3); leave empty to use config.yaml values:
TRACKNETV3_WEIGHTS="${TRACKNETV3_WEIGHTS:-}"
INPAINTNET_WEIGHTS="${INPAINTNET_WEIGHTS:-}"
COURT_POINTS="${COURT_POINTS:-data/input/court_points.json}"
OUTPUT_DIR="${OUTPUT_DIR:-data/output}"
DEVICE="${DEVICE:-cuda}"

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
# inside a subprocess with a cryptic error.
if ! ${PYTHON_CMD} -c "import torch" 2>/dev/null; then
    echo "[SLURM_TRACK] ERROR: 'import torch' failed for PYTHON='${PYTHON_CMD}'"
    echo "  Attempting to install torch for CUDA 11.8..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --quiet
    if ! ${PYTHON_CMD} -c "import torch" 2>/dev/null; then
        echo "[SLURM_TRACK] FATAL: 'import torch' still failed after install"
        exit 1
    fi
fi
echo "[SLURM_TRACK] torch OK — $(${PYTHON_CMD} -c 'import torch; print(torch.__version__)')"
echo "[SLURM_TRACK] CUDA available: $(${PYTHON_CMD} -c 'import torch; print(torch.cuda.is_available())')"

echo "[SLURM_TRACK] VIDEO_PATH=${VIDEO_PATH}"
echo "[SLURM_TRACK] TRACKNET_VERSION=${TRACKNET_VERSION}"
if [[ "${TRACKNET_VERSION}" == "3" ]]; then
    echo "[SLURM_TRACK] TRACKNETV3_WEIGHTS=${TRACKNETV3_WEIGHTS:-<from config.yaml>}"
    echo "[SLURM_TRACK] INPAINTNET_WEIGHTS=${INPAINTNET_WEIGHTS:-<from config.yaml>}"
else
    echo "[SLURM_TRACK] SHUTTLE_WEIGHTS=${SHUTTLE_WEIGHTS}"
fi
echo "[SLURM_TRACK] COURT_POINTS=${COURT_POINTS}"
echo "[SLURM_TRACK] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[SLURM_TRACK] DEVICE=${DEVICE}"
echo "[SLURM_TRACK] Starting at $(date)"

# ── Build --set overrides based on TrackNet version ───────────────────────────
_SET_ARGS=("--set" "tracknet_version=${TRACKNET_VERSION}")
if [[ "${TRACKNET_VERSION}" == "3" ]]; then
    # V3: override weight paths only if explicitly provided; otherwise config.yaml wins
    [[ -n "${TRACKNETV3_WEIGHTS}" ]] && _SET_ARGS+=("--set" "tracknetv3_weights=${TRACKNETV3_WEIGHTS}")
    [[ -n "${INPAINTNET_WEIGHTS}" ]] && _SET_ARGS+=("--set" "inpaintnet_weights=${INPAINTNET_WEIGHTS}")
else
    # V2: always pass shuttle weights override
    _SET_ARGS+=("--set" "tracknet_weights=${SHUTTLE_WEIGHTS}")
fi

# ── Run inference ─────────────────────────────────────────────────────────────
${PYTHON_CMD} main.py \
    --config config.yaml \
    --video "${VIDEO_PATH}" \
    --court-points "${COURT_POINTS}" \
    --output-dir "${OUTPUT_DIR}" \
    --device "${DEVICE}" \
    "${_SET_ARGS[@]}"

echo "[SLURM_TRACK] Finished at $(date)"
