# OSCAR SLURM Torch Import Issues

## ✅ RESOLVED (May 16, 2026)

**Status:** All three training jobs running successfully on OSCAR GPU cluster.

**Root Cause:** PyTorch cu130+ incompatible with OSCAR GPU driver 12090; NCCL symbol errors across all versions.

**Solution:** Create conda environment with PyTorch 2.7.1+cu118, load CUDA 11.8 modules on compute nodes.

**Setup:**
```bash
conda create -n badminton_train python=3.11
conda activate badminton_train
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**SLURM Script Changes:**
```bash
module load cuda/11.8.0-kuhf cudnn/8.7.0.84-11.8-kff3
eval "$(conda shell.bash hook)" && conda activate badminton_train
python -u training/train_${mode}.py ...
```

---

## ✅ YOLO Empty Labels Bug (May 17, 2026)

**Problem:** 0-byte label files; box_loss=0, no instances detected.

**Root Cause:** Line 52 `train_yolo.py` deduplication prevented multiple COCO categories (Player, Player1, Player2) from mapping to single YOLO class. ~1,700 annotations silently dropped.

**Solution:** Removed deduplication check; allow all categories to map to class 0.

**Result:** All 1,703 annotations labeled correctly. Training healthy: box_loss=0.714, mAP50=0.995.

---

## ✅ Pipeline Simplification — MOG2 Removal, GameState, Heatmap (May 17, 2026)

Three interrelated changes landed together. See `AGENTS.md` for updated module contracts.

### MOG2 removed from inference pipeline

**Problem:** `TrackNetTracker` and `PlayerDetector` both applied a MOG2 foreground ratio filter after a 150-frame warmup.  On match_clip5 this suppressed ≥ 97% of post-warmup shuttle detections, leaving only 228 detections (frames 0–150) and producing 0 rallies.

**Root Cause:** Background subtraction is too aggressive for a sport where players and shuttles are always moving; the foreground mask rarely exceeds the 5–6% threshold for genuinely valid detections.

**Solution:** Removed `MOG2Manager` from `main.py` (no longer instantiated or passed), removed the filter block and `update_mog2()` from `models/player_yolo.py`, removed the filter block and `mog2_manager` param from `models/shuttle_tracknet.py`.  `utils/mog.py` is retained for training-augmentation scripts.

**Config cleaned up:** `rally_motion_required_streak: 2` (was 1), `rally_min_displacement_px: 5.0` (was 0.5).

---

### GameState rally logic simplified

**Problem:** `GameState` contained unused / counter-productive logic: a stability detector (`stable_frames` / `stable_frame_threshold`), a large-displacement filter (`max_displacement_fraction` / `frame_size`), and an in-loop trajectory-prediction hit detector (`position_history`, `_fit_trajectory`, `_predict_position`, `_detect_hit`, `prediction_error_threshold`, `last_hit_timestamp`, `hit_cooldown_s`).  The hit detector duplicated `HitDetector`; all three features added per-frame overhead and state that leaked across rallies.

**Solution:** All three removed.  Remaining logic: motion streak (2 frames ≥ 5 px) → rally start; inactive timeout (1 s) → rally end; grace period (8 frames); minimum duration gate (0.1 s).  `frame_size` arg kept in `update()` signature for API compat but is a no-op.  `tests/test_game_state.py` updated to match.

---

### Precomputed player-footwork heatmap

**Added:** `utils/precompute_heatmap.py` — builds a static white-to-JET heatmap (P1 tinted green, P2 tinted orange) from all `feet` positions in `tracking_results.json`.  Large Gaussian blur (σ=40), blended 50% onto the court background, saved as `heatmap.png` in the run directory.

`main.py` calls `precompute_heatmap()` at the end of each pipeline run.

`tools/viewer.py` loads `heatmap.png` at startup (computes it on-the-fly if absent), registers a DearPyGui dynamic texture, and displays it in the right sidebar under **FOOTWORK HEATMAP**.  Per frame, the base image is copied and live player dots are drawn at their court-insert positions via the same homography used for the court overlay.

**Note:** Existing tracking runs with raw ByteTrack IDs (pre-PlayerContext fix) will show sparse heatmaps because the filter targets `id ∈ {1, 2}`.  Re-running `main.py` with the updated pipeline will produce correct P1/P2 data.

---

## ✅ Stroke One-Hot Collapse (May 17, 2026)

**Problem:** Model predicted single class (drive=1.0) every epoch; macro_F1=0.084.

**Root Cause:** `annotations.json` had all features=None; model trained on uniform input.

**Solution:** Generated annotations.json with 1,816 events × 198-dim pose features.

**Result:** Training healthy: loss=0.033, macro_F1=0.160, per-class predictions varying.
