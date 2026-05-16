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
if command -v uv &>/dev/null; then
    PYTHON="uv run python"
else
    PYTHON="python"
fi

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
