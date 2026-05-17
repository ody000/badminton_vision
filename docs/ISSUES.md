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

## ✅ Stroke One-Hot Collapse (May 17, 2026)

**Problem:** Model predicted single class (drive=1.0) every epoch; macro_F1=0.084.

**Root Cause:** `annotations.json` had all features=None; model trained on uniform input.

**Solution:** Generated annotations.json with 1,816 events × 198-dim pose features.

**Result:** Training healthy: loss=0.033, macro_F1=0.160, per-class predictions varying.
