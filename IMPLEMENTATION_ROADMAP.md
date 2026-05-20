# Badminton Vision — Implementation Roadmap

**For Haiku handoff.** All paths relative to repo root. Do not alter files or logic not mentioned here.

Pipeline: `slurm_track.sh → main.py → TrackNetV3Tracker (shuttle) + DINOTracker (players) + HitDetector + GameState`

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
| Task 5-F: Two-player DINO architecture | `TRACKED_CLASSES = ("player_1","player_2")`; head outputs `(B,2,5)`; retrained checkpoint deployed to `models/dino_player.pt` |
| Task 5-H: MOG2 fallback | Cancelled — DINO two-player succeeded |
| Task 5-I: Shuttle circle too large | `box_size` 16→7 heatmap px + 22px video-space cap in `shuttle_tracknetv3.py` |
| Task 5-J Cause 3: Background recomputed every frame | `set_background()` now precomputes `self._background_t` tensor once; `_run_tracknet()` reuses it |
| Task 5-L: Heatmap wrong frame dimensions | `_frame_hw` captured from first decoded frame in `_process_frame`; replaces broken bbox-corner estimate |

---

## 🔄 Task 5-K — DINO patch-average retrain (fix applied, retrain pending)

**Problem confirmed from production data:** DINO position std was 11-23px vs. 100-300px of actual player movement. Root cause: `_extract_cls_token` returned the ViT CLS token, which collapses both players' spatial features into one vector. The head learned mean positions (mAP=0.93 because players are usually near their baseline), not actual tracking.

**Fix already applied to `models/player_dino.py`:** `_extract_cls_token` now returns `features["x_norm_patchtokens"].mean(dim=1)` — the average of all spatial patch tokens. Each patch token encodes a 14×14 image region, so the mean shifts with player positions rather than collapsing them.

**Action required:** Run `slurm_train.sh` — no other code changes needed. The fix is already in the source.

**Expected:** position std increases to 100-300px; boxes visibly follow players during movement. If std stays < 50px after training, increase `BOX_LOSS_WEIGHT` from `0.5` to `1.0–2.0` in `player_dino.py`.

**Note:** Requires full retrain from scratch — old checkpoint head weights are calibrated to CLS token distribution and are incompatible.

---

## 🔄 Task 5-J — Viewer glitches (pending decision)

Three remaining causes. Haiku should NOT act on these without explicit instruction.

**Cause 1 — DINO interval caching teleportation:** Player box jumps 3 frames of distance at once when a player moves fast. Fix: interpolate `(cx,cy)` between cached positions, or increase inference frequency.

**Cause 2 — TrackNetV3 8-frame warmup gap:** First 8 frames return `None` for shuttle; viewer either holds stale position or shows a pop-in. Fix: hold last valid detection for ≤2 frames, then hide.

**Cause 4 — DearPyGui texture upload stall:** `dpg.set_value(texture_tag, ...)` blocks main thread during GPU transfer. Fix: decode frames on a background thread, queue pre-decoded frames for the viewer.
