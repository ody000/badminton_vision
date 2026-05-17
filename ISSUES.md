# OSCAR SLURM Torch Import Issues

## Original Error
```
[SLURM] ERROR: 'import torch' failed for PYTHON='uv run python'
  Activate your conda/venv environment before submitting, or
  uncomment the module load / conda activate lines above.
```

When submitting `slurm_train.sh` and `slurm_track.sh` to OSCAR GPU cluster.

---

## Attempts & Results

### Attempt 1: Option 2 - Activate locally before submitting
**Approach:** Activate conda env locally, then submit job with `sbatch`.

**Result:** FAILED  
Error persisted. Local activation doesn't carry to compute nodes.

---

### Attempt 2: Uncomment conda activation in scripts
**Approach:** Uncomment lines in `slurm_train.sh`:
```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /users/zshen38/ulg_new_env
```

**Result:** FAILED  
Path `~/miniconda3/etc/profile.d/conda.sh` not found on OSCAR cluster.

---

### Attempt 3: Use .venv from uv sync
**Approach:** Uncomment venv activation instead:
```bash
source .venv/bin/activate
```

**Result:** FAILED  
Still got torch import error on compute node. Relative path `.venv/bin/activate` didn't work on compute nodes.

---

### Attempt 4: Load CUDA 11.8.0 + cuDNN 8.6.0 modules
**Approach:** Add module loads to script:
```bash
module load cuda/11.8.0 cudnn/8.6.0
```

**Result:** FAILED  
Module not found: `cudnn/8.6.0` doesn't exist on OSCAR.

---

### Attempt 5: Load correct CUDA/cuDNN versions
**Command to find available versions:**
```bash
module avail cuda
module avail cudnn
```

**Available versions found:**
- CUDA: `cuda/11.6.0-fwdj`, `cuda/11.8.0-kuhf`, `cuda/12.9.0-cinr` (default)
- cuDNN: `cudnn/8.7.0.84-11.8-kff3`, `cudnn/9.8.0.87-12-y7fu` (default)

**Approach:** Try CUDA 11.8.0 + cuDNN 8.7.0:
```bash
module load cuda/11.8.0-kuhf cudnn/8.7.0.84-11.8-kff3
```

**Result:** FAILED  
Same `ncclCommResume` symbol error:
```
ImportError: /oscar/home/zshen38/badminton_vision/.venv/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so: undefined symbol: ncclCommResume
```

**Diagnosis:** Torch binary in `.venv` compiled for different CUDA version than available.

---

### Attempt 6: Try CUDA 12.9.0 (default)
**Approach:** Load default CUDA/cuDNN:
```bash
module load cuda/12.9.0-cinr cudnn/9.8.0.87-12-y7fu
```

**Result:** FAILED  
Same `ncclCommResume` error with 12.9.0.

---

### Attempt 7: Reinstall torch for CUDA 12.1
**Approach:** Uninstall old torch, install new one:
```bash
source .venv/bin/activate
module load cuda/12.9.0-cinr cudnn/9.8.0.87-12-y7fu
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Result:** FAILED (partially)
- Torch 2.5.1+cu121 installed
- But local test still gave:
  ```
  ImportError: undefined symbol: ncclCommResume
  ```
- ldd showed all libraries present but incompatible

---

### Attempt 8: Follow slayminton pattern
**Approach:** Copied approach from `../slayminton/slurm_train.sh`:
- Source venv: `source .venv/bin/activate`
- Use: `uv run -v python -u main.py`
- No CUDA module loads

**Result:** FAILED  
Compute node couldn't find `.venv/bin/activate` (relative path issue).

---

### Attempt 9: Use absolute path to venv
**Approach:** Use absolute path in activation:
```bash
source /users/zshen38/badminton_vision/.venv/bin/activate
PYTHON="uv run -v python -u"
```

**Result:** FAILED  
Back to original error:
```
[SLURM] ERROR: 'import torch' failed for PYTHON='uv run -v python -u'
```

---

### Attempt 10: Use plain python from venv
**Approach:** Skip uv run, use plain python from activated venv:
```bash
source /users/zshen38/badminton_vision/.venv/bin/activate
PYTHON="python -u"
```

**Result:** FAILED  
Same torch import error.

---

## Root Cause Analysis

The core issue is **torch installation incompatibility with OSCAR's CUDA/NCCL stack**:

1. **NCCL symbol missing:** The `ncclCommResume` symbol is undefined in all CUDA versions tested (11.8.0, 12.9.0)
2. **Pre-built binary incompatibility:** Torch binary in `.venv` was compiled for an environment not available on OSCAR
3. **Multiple failed reinstall attempts:** Reinstalling torch for cu121 didn't resolve the symbol mismatch

---

## Current State

**Files updated:**
- `slurm_train.sh`: Points to `/users/zshen38/badminton_vision/.venv/bin/activate`, uses `python -u`
- `slurm_track.sh`: Same changes

**Current configuration:**
```bash
source /users/zshen38/badminton_vision/.venv/bin/activate
PYTHON="python -u"
```

**Status:** BROKEN - Torch still not importable

---

## Next Steps (Deferred)

To be addressed systematically later:

1. **Try conda environment** instead of uv-managed venv
   - OSCAR may have better compatibility with conda-managed environments
   - Test: `conda install -c pytorch pytorch torchvision torchaudio pytorch-cuda=12.1`

2. **Check OSCAR's official torch setup**
   - See if OSCAR provides pre-compiled torch modules
   - Run: `module avail pytorch` or `module spider torch`

3. **Rebuild torch from source** (last resort)
   - Compile torch against OSCAR's exact CUDA/NCCL versions
   - Time-consuming but guaranteed compatibility

4. **Verify pyproject.toml / requirements**
   - Check if `uv sync` locked to incompatible torch version
   - May need to update dependencies

5. **Test with minimal reproducible case**
   - Create simple SLURM job that only imports torch
   - Iterate on environment setup without full training overhead

---

## Relevant Files

- `slurm_train.sh` - Main training launcher
- `slurm_track.sh` - Inference launcher
- `.venv/` - Current venv with broken torch
- `pyproject.toml` / `uv.lock` - Dependency versions
