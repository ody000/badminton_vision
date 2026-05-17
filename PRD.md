# PRD: badminton_vision Clean-Slate Rewrite

**Status:** Ready for implementation  
**Date:** May 2026  
**Author:** Ziqi Shen  

---

## Problem Statement

Coaches and players lack scalable tools for automated badminton replay analysis. Manual
review is slow, subjective, and cannot extract structured metrics like rally duration,
shot type, player coverage, or hit counts at scale.

Two prototype systems exist — `badminton_vision` (primordial) and `slayminton`
(more established) — but neither is production-ready. `badminton_vision` has a cleaner
module structure but broken implementations (wrong TrackNet output channel, no persistent
player IDs, primitive rally logic, scattered hardcoded constants). `slayminton` has
stronger core implementations (robust TrackNet wrapper, sophisticated rally detection,
MOG2 pipeline, court heatmaps) but its architecture is wrong: game logic, visualization,
homography, and kinematics are all collapsed into a single 900-line `visualizations.py`
script. Neither system has a stroke classifier, a working GUI that persists court corners
to disk, multi-GPU training, or a principled hit detection approach.

The result: neither codebase can be extended reliably, trained on OSCAR, or used by a
non-developer to analyze a match video.

---

## Solution

A clean-slate rewrite of `badminton_vision` that:

1. Correctly ports the best implementations from `slayminton` (TrackNet tracker, GameState
   rally logic, MOG2 pipeline, court heatmap visualization) into a properly layered
   architecture.
2. Replaces the failed DINOv3 player tracker with YOLOv8 + ByteTrack, gated by a MOG2
   foreground confidence filter.
3. Introduces a principled hit detector (quadratic RANSAC fit + Kalman interface +
   player proximity gate) replacing the brittle velocity-reversal heuristic.
4. Adds a stroke classifier (MediaPipe Pose keypoints + shuttle trajectory fed into a
   Transformer) trained on the FineBadminton dataset.
5. Provides a headless CLI pipeline (`main.py`) fully compatible with OSCAR/SLURM, a
   local court-corner setup helper (`prepare_run.py`), and a Streamlit results dashboard
   (`dashboard.py`) that reads output JSON artifacts.
6. Provides DDP-ready SLURM training scripts for TrackNet fine-tuning, YOLO fine-tuning,
   and stroke classifier training — all evaluated with IoU, mAP, and distance error.
7. Establishes a single `config.yaml` as the source of truth for all tunable parameters.

`slayminton/` is preserved as a read-only reference and must not be modified.

---

## User Stories

### Pipeline Execution

1. As a researcher, I want to run a single CLI command pointing at a video file and
   receive structured JSON output (per-frame detections, hit events, rally segments,
   analytics), so that I can analyze a match without writing any code.
2. As a researcher, I want the pipeline to run fully headless on OSCAR, so that I can
   process long match videos on GPU without needing a display server.
3. As a researcher, I want all SLURM stdout and stderr logs written to
   `data/output/logs/`, so that I can find job output in a consistent location.
4. As a researcher, I want a timestamped output directory created per run, so that
   multiple runs on the same video don't overwrite each other.
5. As a researcher, I want pipeline parameters (thresholds, timeouts, model paths)
   overridable from the CLI and from SLURM `--export` flags, so that I can run
   parameter sweeps without editing source code.

### Court Corner Setup

6. As a user running locally, I want a GUI tool that shows the first frame of my video
   and lets me click 6 court corners (4 boundary corners + 2 midline points), so that
   I can calibrate homography without manually editing coordinate arrays in code.
7. As a user, I want the court corners I selected to be automatically saved to a JSON
   file keyed by video name, so that I never have to re-click them for the same video.
8. As a researcher, I want to pass the saved court-corners JSON path to the SLURM
   tracking job, so that homography runs correctly on OSCAR without any GUI interaction.
9. As a user, I want to undo the last clicked point during court-corner selection with
   the `z` key, so that I can correct mistakes without restarting.

### Shuttle Tracking

10. As a researcher, I want shuttle tracking to use the correct, up-to-date TrackNet
    weights and implementation from slayminton, so that shuttle detection is as accurate
    as possible.
11. As a researcher, I want the TrackNet frame buffer to be timestamp-aware and flush
    stale frames when a gap larger than 2× the frame interval is detected, so that
    frame-order artifacts do not corrupt the 3-frame temporal input.
12. As a researcher, I want shuttle detections with insufficient white-pixel foreground
    coverage to be automatically rejected after the MOG2 warmup period, so that white
    court lines and shoes do not produce false shuttle detections.
13. As a researcher, I want the MOG2 warmup period to be 150 frames (configurable), so
    that the background model stabilizes before the foreground filter is applied.

### Player Tracking

14. As a researcher, I want players tracked with persistent integer IDs (P1, P2) that
    remain stable across the full video, so that per-player statistics are meaningful.
15. As a researcher, I want player detection to use YOLOv8 on raw RGB frames (not MOG2
    masks), so that YOLO's pretrained weights are applied as intended.
16. As a researcher, I want ByteTrack (Ultralytics built-in) used for player identity
    continuity, so that occlusion and brief disappearances do not cause ID swaps.
17. As a researcher, I want YOLO player detections gated by a MOG2 foreground coverage
    threshold (default 6%), so that static background persons (umpires, crowd) are
    suppressed without rejecting momentarily-paused players.

### Hit Detection

18. As a researcher, I want hit events detected using a quadratic trajectory fit over
    the last 6 shuttle positions (3 pre + 3 post), so that parabolic arcs during rallies
    do not produce false positives.
19. As a researcher, I want RANSAC (10 iterations) used during trajectory fitting, so
    that single-frame TrackNet outliers do not corrupt the quadratic fit.
20. As a researcher, I want a Kalman filter interface stub in the hit detector, so that
    a physics-based trajectory model can be swapped in later without changing any
    downstream code.
21. As a researcher, I want hits attributed to the nearest player within a configurable
    proximity threshold (default 200 cm real-world), so that floor bounces and net
    contacts are not falsely attributed to a player.
22. As a researcher, I want a minimum cooldown between consecutive hits (default 200ms),
    so that a single physical hit does not fire multiple detection events.
23. As a researcher, I want each confirmed hit event stored in `events.json` with
    timestamp, attributed player ID, 6-point pre and post trajectory, and prediction
    error, so that all downstream analysis has a rich hit record to work from.

### Rally Detection

24. As a researcher, I want rally active/inactive state determined by shuttle motion
    inactivity timeout (default 1.0s), so that rallies are segmented without manual
    annotation.
25. As a researcher, I want short inactive gaps below 0.5s merged into surrounding
    rally periods, so that brief tracking dropouts do not fragment a single rally into
    multiple segments.
26. As a researcher, I want rally boundaries stored in `rally_data.json` with
    `rally_id`, `start_time`, `end_time`, and `duration_s`, so that downstream
    analysis has structured rally records.
27. As a researcher, I want the pipeline to correctly close any open rally at end-of-
    stream, so that the final rally is never dropped from the output.

### Stroke Classification

28. As a researcher, I want shot type classified per hit event using MediaPipe Pose
    keypoints at the hit keyframe combined with pre/post shuttle trajectory features,
    so that stroke classification captures both body posture and shuttlecock dynamics.
29. As a researcher, I want all three FineBadminton annotation levels (Foundational
    Actions, Tactical Semantics, Decision Evaluation) stored per hit event, so that
    higher-level tactical analysis is possible in future work.
30. As a researcher, I want the stroke classifier trained on FineBadminton data via a
    SLURM job, with per-class accuracy and macro F1 reported, so that model quality is
    quantified.
31. As a researcher, I want the stroke classifier to be a separate module that consumes
    a hit event dict, so that it can be improved or replaced without touching the
    tracking pipeline.

### Training Pipelines

32. As a researcher, I want to fine-tune TrackNet on the Roboflow badminton dataset
    (shuttle annotations) via a SLURM job, so that shuttle detection is adapted to our
    specific video domain.
33. As a researcher, I want to fine-tune YOLOv8 on the Roboflow badminton dataset
    (player annotations) via a SLURM job, so that player detection accuracy improves
    over the generic COCO pretrained model.
34. As a researcher, I want TrackNet fine-tuning evaluated with both distance error
    (pixels from heatmap peak to GT center) and mAP@0.5, so that I have a meaningful
    metric for a small, fast-moving object where raw IoU is uninformative.
35. As a researcher, I want YOLO fine-tuning evaluated with Ultralytics' built-in
    mAP@0.5, mAP@0.5:0.95, precision, and recall per class, so that I have standard
    detection metrics.
36. As a researcher, I want all training jobs DDP-ready (SyncBatchNorm + torchrun +
    `--nproc_per_node`), so that I can scale to 2 GPUs on OSCAR without code changes.
37. As a researcher, I want training checkpoints and metric logs written to
    `data/output/logs/<run>/`, so that all training artifacts are in a consistent
    findable location.
38. As a researcher, I want a data download script that fetches the Roboflow dataset
    and the FineBadminton dataset into the correct `data/input/` subdirectories, so
    that any collaborator can reproduce the training data setup with one command.
39. As a researcher, I want horizontal-flip augmentation applied to the TrackNet
    training set (doubling dataset size), so that the model learns orientation invariance.

### Dashboard

40. As a coach, I want a Streamlit dashboard that reads pipeline output JSONs and shows
    a rally timeline, per-shot stroke classification table, player coverage heatmaps,
    and shuttle kinematics plots, so that I can review a match without opening JSON
    files manually.
41. As a coach, I want to scrub through the rally timeline and select individual rallies
    to inspect, so that I can focus on specific moments in a match.
42. As a coach, I want stroke type displayed with confidence score per hit, so that I
    can distinguish high-confidence classifications from uncertain ones.
43. As a coach, I want the dashboard to work entirely from the output JSON files
    (no re-running the pipeline), so that review is fast and does not require GPU.

### Configuration

44. As a developer, I want all tunable parameters stored in a single `config.yaml`, so
    that I can reproduce any experiment by archiving one file.
45. As a developer, I want CLI flags and SLURM `--export` overrides to take precedence
    over `config.yaml` values, so that parameter sweeps do not require file edits.
46. As a researcher, I want player and shuttle MOG2 thresholds to be separate
    configurable values, so that they can be tuned independently.

### Testing

47. As a developer, I want module-level smoke tests that run in under 5 seconds with no
    GPU and no video file, so that I can validate each component in isolation during
    development.
48. As a developer, I want a full-pipeline integration test that runs on a short test
    clip and asserts that all output files are produced and at least one rally is
    detected, so that regressions in the end-to-end system are caught before OSCAR jobs.

---

## Implementation Decisions

### Architecture: strict layer separation

Three layers, no cross-layer imports in the wrong direction:
- `models/`: perception only — detect, track, classify. No game logic.
- `core/`: reasoning only — hits, rallies, homography. No rendering, no model calls.
- `utils/`: I/O and rendering only — video, visualization. No game logic.

`main.py` is the only file allowed to import across all three layers.

### Player tracking: YOLOv8 + ByteTrack + MOG2 filter

`model.track(frame, persist=True)` (not `model.predict`) is used so that ByteTrack
assigns and maintains persistent integer track IDs. MOG2 foreground coverage is computed
on each YOLO bounding box and used as a post-detection confidence gate, not as model
input. The MOG2 filter is disabled for the first 150 frames (warmup).

### Shuttle tracking: slayminton TrackNetTracker, adopted verbatim

The existing `slayminton/models/tracknet.py` `TrackNetTracker` is adopted with one
addition: a timestamp-aware frame buffer that flushes stale frames when the inter-frame
gap exceeds 2× the expected frame interval. Output channel count is always inferred
from the checkpoint's final conv layer, never hardcoded.

### Hit detection: three-stage pipeline behind a stable interface

Stage 1 — RANSAC quadratic fit over the last 6 shuttle positions (10 iterations,
minimum 3 inliers). Stage 2 — predict next position from fitted curve; declare hit
candidate if prediction error exceeds threshold. Stage 3 — gate by player proximity
(real-world cm) and cooldown timer. The public interface is
`HitDetector.update(timestamp, pos, player_feet) -> (bool, player_id)`.
A `set_kalman_model()` no-op stub is included so a physics-based model can be
substituted without changing callers.

### MOG2 thresholds

Player bounding box foreground coverage: 6% (default). Shuttle bounding box white-pixel
ratio: 5% (default). These are separate config values because the two use cases have
different noise profiles. Both thresholds are disabled for the first 150 frames.

### Court corner persistence

`prepare_run.py` owns all GUI interaction and always saves to
`data/input/court_points.json` keyed by video stem. `main.py` only reads from JSON —
it never opens a window. This makes the full pipeline headless and OSCAR-compatible.
The saved JSON is copied into each run's output directory so that the exact calibration
used is archived with the results.

### Stroke classifier: Option C+ (pose + trajectory → Transformer)

MediaPipe Pose extracts 33 joint (x, y, visibility) tuples from the hit keyframe.
These are concatenated with 6 pre-hit and 6 post-hit shuttle positions (real-world cm)
to form a fixed-size feature vector. This vector is passed to the existing
`stroke_transformer.py` stub (to be implemented as a Transformer encoder + classification
head). Three output heads correspond to FineBadminton's three annotation levels;
Foundational Actions is the primary training objective. This approach matches the
BST architecture from the literature (mAP ~96% on stroke classification).

### TrackNet fine-tuning metrics

Because shuttles occupy only ~16×16 pixels, raw IoU on bounding boxes is near-zero
even for correct detections. The primary metric is **distance error** (Euclidean
distance in pixels from heatmap argmax to GT center). mAP@0.5 is computed treating
the heatmap peak as a fixed-size box, for comparability with prior work.

### Multi-GPU training

TrackNet training uses PyTorch DDP with `SyncBatchNorm.convert_sync_batchnorm(model)`
before wrapping. SLURM launchers use `torchrun --nproc_per_node=$NGPUS`. YOLO training
uses Ultralytics' native multi-GPU support (`device="0,1"`). Both default to 1 GPU;
scaling to 2 requires only an `--export NGPUS=2` at sbatch time.

### SLURM log paths

All `#SBATCH -o` and `-e` directives point to `data/output/logs/slurm-%j.out` and
`data/output/logs/slurm-%j.err`. A `mkdir -p data/output/logs` precedes any Python
call in every SLURM script.

### Configuration

All tunable parameters live in `config.yaml`. A `load_config(path, overrides)` utility
merges the YAML baseline with a flat dict of CLI/SLURM overrides. No module contains
magic numbers — every threshold, size, timeout, and color must be sourced from config.

### Dashboard: Streamlit, read-only from JSON

`dashboard.py` reads `events.json`, `rally_data.json`, and `analytics.json` from a
selected output directory. It never re-runs the pipeline. Heavy rendering (heatmaps,
kinematics plots) is done with matplotlib; Streamlit handles layout. The dashboard is
not shipped to OSCAR and has no SLURM interaction.

### Output schema

Five JSON files per run (tracking_results, events, rally_data, analytics,
court_points) plus one annotated MP4. See AGENTS.md for complete field-level schema.
This schema is append-compatible: new fields can be added without breaking existing
dashboard consumers.

---

## Testing Decisions

A good test exercises the module's public interface with realistic inputs and asserts
on observable outputs — it does not inspect internal state, mock internals, or
test implementation details. Tests should be deterministic (no random seeds needed
if inputs are synthetic) and run without GPU or real video.

### Tier 1: Module Smoke Tests

All run with `pytest tests/ -k "not integration"`. Target runtime < 5 seconds total.

- **HitDetector**: synthesize a perfect 6-point parabola with one deliberate positional
  discontinuity at frame 4. Assert exactly 1 hit is returned at frame 4. Assert the
  attributed player matches the nearest synthetic player feet coordinate. Assert no
  hit fires on a smooth parabola with no discontinuity.
- **CourtMapper**: feed the four known pixel corners of a regulation court image.
  Assert that the transformed real-world coordinates match the expected cm values
  (610 × 1340) within 1 cm tolerance. Assert `transform_point(None)` returns `None`.
- **TrackNetTracker**: feed 3 blank (288×512×3) uint8 frames. Assert return type is
  dict. Assert no exception is raised regardless of whether confidence threshold is met.
- **PlayerDetector**: feed one blank (720×1280×3) frame. Assert return type is list.
  Assert no exception.
- **GameState**: simulate 10 shuttle detections at 30fps with 2.0px displacement each,
  then 60 frames of None detections. Assert rally starts within the first 3 frames and
  ends after the 1.0s inactivity timeout.

### Tier 2: Integration Test

Run with `pytest tests/test_pipeline.py`. Requires `data/input/test_clip.mp4` (a
10–30 second clip checked into the repository).

- Call `main.run(video_path, court_points_path, config)` programmatically.
- Assert all 5 output JSON files exist in the run output directory.
- Assert `rally_data.json` contains at least 1 rally with `duration_s > 0`.
- Assert `tracking_results.json` contains at least 1 frame where shuttle is non-null.
- Assert `events.json` is a valid JSON array (may be empty for a short clip).
- Assert no Python exception was raised during the run.

---

## Out of Scope

- **Automated court corner detection** (`court_resnet.py`): detecting court lines
  programmatically is a separate research problem. The GUI + JSON workflow is sufficient.
- **Scorekeeping**: determining point wins requires knowing when the shuttle lands
  in/out, which requires 3D trajectory reconstruction or a separate in/out classifier.
  Not in scope.
- **Multi-match analysis**: aggregating statistics across multiple match videos.
- **Real-time streaming**: the pipeline operates on recorded video files only.
- **Racket tracking**: racket bounding boxes are in the Roboflow annotations but are
  not used in this system. Required for Option B stroke classification (video clip
  backbone); deferred to future work.
- **Shot quality prediction**: FineBadminton Decision Evaluation labels are stored in
  `events.json` but no model is trained on them in this iteration.
- **3D trajectory reconstruction**: MonoTrack-style depth estimation from a single
  monocular camera.
- **OSCAR job submission from the dashboard**: the Streamlit dashboard does not SSH
  into OSCAR or submit SLURM jobs. The workflow is local preparation → manual sbatch
  → local review.

---

## Further Notes

### Data availability

The Roboflow training dataset (~10K frames, COCO annotations) is not present in the
local repository — it was on OSCAR and was not synced back. `data/input/train/`,
`data/input/train_mog_frames/`, and `data/input/train_mog_reflect/` are all empty
locally. A `utils/data_download.py` script must be written as part of this rewrite
to fetch both the Roboflow dataset (via the `roboflow` Python package) and the
FineBadminton dataset before any training jobs can run.

The best TrackNet weights are at `slayminton/models/tracknet.pt`. Copy to
`weights/tracknet.pt` as the starting checkpoint for fine-tuning.

### Reference for slayminton code being ported

| New location | Source in slayminton |
|---|---|
| `models/TrackNet.py` | `slayminton/models/tracknet.py` (TrackNet class) |
| `models/shuttle_tracknet.py` | `slayminton/models/tracknet.py` (TrackNetTracker) |
| `core/game_state.py` | `slayminton/core/game_state.py` |
| `core/analysis.py` | `slayminton/core/analysis.py` |
| `utils/mog.py` | `slayminton/scripts/mog.py` |
| `utils/visualization.py` (court drawing) | `slayminton/scripts/visualizations.py` |
| `scripts/augment_data_reflect.py` | `slayminton/scripts/augment_data_reflect.py` |

### FineBadminton dataset

URL: https://ilearn-lab.github.io/MM25-FineBadminton/  
Three annotation levels: Foundational Actions (shot type), Tactical Semantics,
Decision Evaluation. All three are downloaded and stored. Only Foundational Actions
is the primary training target for the stroke classifier in this iteration.

### ByteTrack note

Use `model.track(frame, persist=True, classes=[0])` via Ultralytics. The `persist=True`
flag is required to maintain track IDs across frames. Without it, IDs reset every call.
