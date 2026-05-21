# Badminton Vision — Implementation Roadmap

**For Haiku handoff.** All paths relative to repo root. Do not alter files or logic not mentioned here.

Pipeline: `slurm_track.sh → main.py → TrackNetV3Tracker (shuttle) + YOLO (players) + HitDetector + GameState`

---

## ✅ Resolved (archive — do not re-implement)

| Task | Fix |
|------|-----|
| Priority 0: TrackNet on CPU | `device=None` default in `shuttle_tracknet.py`; reads from config |
| Phase 1-B: DINO interval caching | `set_detect_interval()` + `_detect_cache` in `DINOTracker`; `player_detect_interval: 3` in config |
| Phase 1-C: FP16 inference | `torch.autocast` in `forward_detect()` |
| Phase 2: TrackNetV3 integration | `shuttle_tracknetv3.py` + `utils/background.py`; `tracknet_version: 3` in config |
| Phase 4: Multi-frame stroke window | `_frame_window` deque + `_pending_hits` in `main.py`; `HitEvent` carries surrounding frames |
| Bug 5-A: TrackNet tracking background | Input tensor was `[frames, bg]`; fixed to `[bg, frames]` in `_run_tracknet()` |
| Bug 5-B: BGR→RGB mismatch | Added `cv2.cvtColor(BGR→RGB)` before `Image.fromarray` in `detect()` |
| Bug 5-C: Heatmap blank | Downstream of 5-A; also fixed `feet_px` alias in `main.py` serialization |
| Task 5-D/E: Diagnostic prints | `[TRACKNETV3 DIAG]` and `[DINO DIAG]` and `[HIT]` prints added; still in code |
| Task 5-G: TrackNet coordinate scaling | `_last_orig_h/w` stored in `detect()`; `scale = orig_h/H_hm` in `_run_tracknet()` |
| Task 5-F: Two-player DINO architecture | Attempted but abandoned; reverted to YOLO (models/yolo.pt) for production |
| Task 5-H: MOG2 fallback | Cancelled — DINO approach abandoned; YOLO used instead |
| Task 5-I: Shuttle circle too large | `box_size` 16→7 heatmap px + 22px video-space cap in `shuttle_tracknetv3.py` |
| Task 5-J Cause 3: Background recomputed every frame | `set_background()` now precomputes `self._background_t` tensor once; `_run_tracknet()` reuses it |
| Task 5-L: Heatmap wrong frame dimensions | `_frame_hw` captured from first decoded frame in `_process_frame`; replaces broken bbox-corner estimate |

---

## ⛔ Task 5-K — DINO two-player architecture (abandoned; reverted to YOLO)

**Status:** CANCELLED after multiple retraining attempts. DINO architecture could not achieve stable two-player tracking despite architectural modifications (patch-average pooling, dual-head outputs, LoRA fine-tuning).

**Decision:** Revert to YOLOv8 (models/yolo.pt), which is fine-tuned for badminton player detection and consistently outperforms DINO in this task.

**Code changes:**
- `main.py` line 67: Import changed from `models.player_dino` → `models.player_yolo`
- `main.py` line 225: API call changed from `detect_yolo_compat()` → `detect()`
- `main.py` line 131: Updated diagnostic string from "DINOTracker" → "YOLO"
- `models/player_yolo.py`: Added `set_detect_interval()` method for API compatibility with existing pipeline

**Rationale:** YOLO is simpler, faster, deterministic, and production-proven on badminton video. The architectural overhead of DINO did not justify the training complexity or uncertainty. The test fine-tuning checkpoint (dino_player.pt) is retained but no longer used.

---

## 🔄 Task 5-J — Viewer glitches (pending decision)

Three remaining causes. Haiku should NOT act on these without explicit instruction.

**Cause 1 — DINO interval caching teleportation:** Player box jumps 3 frames of distance at once when a player moves fast. Fix: interpolate `(cx,cy)` between cached positions, or increase inference frequency.

**Cause 2 — TrackNetV3 8-frame warmup gap:** First 8 frames return `None` for shuttle; viewer either holds stale position or shows a pop-in. Fix: hold last valid detection for ≤2 frames, then hide.

**Cause 4 — DearPyGui texture upload stall:** `dpg.set_value(texture_tag, ...)` blocks main thread during GPU transfer. Fix: decode frames on a background thread, queue pre-decoded frames for the viewer.
