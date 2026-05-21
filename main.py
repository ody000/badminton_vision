"""Headless CLI pipeline for badminton match analysis.

Usage:
    python main.py --video data/input/match_clip.mp4
    python main.py --video VIDEO --config config.yaml --court-points data/input/court_points.json
    python main.py --video VIDEO --set fps=60 device=cuda
    python main.py --video VIDEO --annotate   # also write annotated_video.mp4

Output per run: {output_dir}/{video_stem}_{timestamp}/
  tracking_results.json
  events.json
  rally_data.json
  analytics.json
  court_points.json
  annotated_video.mp4   (only written when --annotate is passed)

Visualization note:
  annotated_video.mp4 is opt-in because re-encoding is slow and requires
  downloading the file from OSCAR.  Use tools/viewer.py locally instead:
      python tools/viewer.py --run <run_dir> --video <original_video>
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline entry point (importable for tests)
# ─────────────────────────────────────────────────────────────────────────────

def run(
    video_path: str,
    config_path: str = "config.yaml",
    court_points_path: str | None = None,
    output_dir: str | None = None,
    overrides: dict | None = None,
    annotate: bool = False,
) -> str:
    """Run the full pipeline. Returns path to the run output directory.

    Args:
        video_path:        Path to the source video file.
        config_path:       Path to config.yaml.
        court_points_path: Override for court_points.json location.
        output_dir:        Override for output directory.
        overrides:         Dict of config key overrides (CLI / SLURM --export).
        annotate:          If True, write annotated_video.mp4 to the run dir.
                           Off by default to avoid slow re-encode on OSCAR.
    """
    from utils.config_loader import load_config
    from utils.video_io import VideoIOHandler
    from utils.visualization import render_court_insert, render_frame
    from models.shuttle_tracknetv3 import TrackNetV3Tracker
    from models.player_yolo import PlayerDetector
    from models.stroke_classifier import StrokeClassifier
    from core.homography import CourtMapper
    from core.game_state import GameState, build_rally_status_per_frame
    from core.hit_detector import HitDetector
    from core.analysis import Analysis
    from core.player_context import PlayerContext
    from core.tracking_types import Shuttle, HitEvent

    cfg = load_config(config_path, overrides)

    # Apply top-level overrides from function args
    if output_dir is not None:
        cfg.output_dir = output_dir
    if court_points_path is not None:
        cfg.court_points_file = court_points_path

    fps = float(getattr(cfg, "fps", 30.0))

    # ── Run directory ──────────────────────────────────────────────────────────
    video_stem = Path(video_path).stem
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg.output_dir, f"{video_stem}_{ts_str}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"[MAIN] run_dir={run_dir}")

    # ── Initialize components ─────────────────────────────────────────────────
    # TrackNetV3 is now the only supported shuttle tracker
    from utils.background import estimate_background
    _bg_frames = int(getattr(cfg, "tracknet_bg_frames", 150))
    print(f"[MAIN] Estimating video background from first {_bg_frames} frames...")
    _background = estimate_background(video_path, n_frames=_bg_frames,
                                      resize_hw=(int(getattr(cfg, "tracknet_expected_h", 288)),
                                                 int(getattr(cfg, "tracknet_expected_w", 512))))
    tracknet = TrackNetV3Tracker(cfg=cfg, background=_background)
    device = torch.device(getattr(cfg, "device", "cuda" if torch.cuda.is_available() else "cpu"))
    _player_weights = (
        getattr(cfg, "player_weights", None)
        or getattr(cfg, "player_weights_path", None)
        or getattr(cfg, "player_model_path", None)
    )
    
    # Create a config-like object for PlayerDetector (player_yolo expects cfg object)
    from types import SimpleNamespace
    yolo_cfg = SimpleNamespace(
        player_weights=_player_weights,
        player_conf_threshold=getattr(cfg, "player_conf_threshold", 0.5),
        device=device,
        player_detect_interval=int(getattr(cfg, "player_detect_interval", 1)),
    )
    yolo = PlayerDetector(cfg=yolo_cfg)
    # ── CUDA diagnostic (Priority 0 verification) ───────────────────────────────
    # TrackNetV3Tracker stores network as .tracknet
    _tracknet_net = getattr(tracknet, "tracknet", None)
    _tracknet_device = next(_tracknet_net.parameters()).device if _tracknet_net is not None else "unknown"
    # YOLO device comes from cfg directly (already resolved to torch.device)
    _yolo_device = str(device)
    print(f"[MAIN] CUDA diagnostic: "
          f"torch.cuda.is_available()={torch.cuda.is_available()}, "
          f"yolo.device={_yolo_device}, "
          f"tracknet.device={_tracknet_device}")

    # ── Phase 1-B: Player detection interval caching ─────────────────────────────
    _player_interval = int(getattr(cfg, "player_detect_interval", 1))
    yolo.set_detect_interval(_player_interval)
    print(f"[MAIN] YOLO interval caching: every {_player_interval} frame(s)")

    court_mapper = CourtMapper(cfg)
    court_points_file = getattr(cfg, "court_points_file", "data/input/court_points.json")
    calibrated = court_mapper.load_from_json(court_points_file, video_stem)
    if not calibrated:
        print("[MAIN] Warning: court not calibrated — real-world transforms disabled.")

    # PlayerContext owns all player lifecycle: transform, ID assignment, history
    player_ctx = PlayerContext()

    game_state = GameState(cfg=cfg)
    hit_detector = HitDetector(cfg=cfg)
    stroke_classifier = StrokeClassifier(cfg=cfg)
    analysis = Analysis()

    video_io = VideoIOHandler(
        input_path=video_path,
        output_path=os.path.join(run_dir, "annotated_video.mp4") if annotate else None,
    )
    source_fps = video_io.get_fps()
    tracknet.set_fps(source_fps)

    tracking_results: list[dict] = []
    events: list[dict] = []
    final_timestamp = 0.0
    _frame_hw: list[int | None] = [None, None]  # actual video [h, w]; set from first frame

    # ── Stroke surrounding-frame buffer (Phase 4-B) ───────────────────────────
    # Keeps a rolling window of recent raw frames so StrokeClassifier can access
    # pre-hit frames after hit detection fires (hit detection is inherently delayed).
    _STROKE_PRE_FRAMES  = int(getattr(cfg, "stroke_pre_frames",  3))   # frames before hit
    _STROKE_POST_FRAMES = int(getattr(cfg, "stroke_post_frames", 3))   # frames after hit
    _STROKE_WINDOW      = _STROKE_PRE_FRAMES + 1 + _STROKE_POST_FRAMES # total T

    from collections import deque
    _frame_window: deque[tuple[np.ndarray, int, float]] = deque(
        maxlen=_STROKE_PRE_FRAMES + 1   # only need to buffer pre-hit frames; post collected on the fly
    )

    # Pending hit events waiting for post-hit frames before being emitted
    # Each entry: {"hit_ev": HitEvent, "timestamp": float, "frame_idx": int, "hit_player_id": int, "frames_left": int}
    _pending_hits: list[dict] = []

    # ── Batched vs sequential TrackNet ────────────────────────────────────────
    # detect_batch() amortises GPU kernel launch overhead across N frames, giving
    # ~3–5× better GPU utilisation.  On CPU there is no hardware parallelism, so
    # batching is strictly slower (batch assembly overhead + no latency hiding).
    # We therefore gate it: only activate detect_batch() when running on CUDA.
    _device_str = str(getattr(cfg, "device", "cpu")).lower()
    _use_batched = _device_str.startswith("cuda")
    BATCH = int(getattr(cfg, "tracknet_batch_size", 8))
    frame_buffer: list[tuple] = []  # (frame_bgr, frame_idx, timestamp)

    def _emit_stroke_event(hit_ev: HitEvent, timestamp: float, frame_idx: int, hit_player_id) -> None:
        """Classify and append a completed hit event (pre + post frames available)."""
        # TEMPORARY DIAGNOSTIC (Task 5-E)
        print(f"[HIT] frame={frame_idx} ts={timestamp:.3f}s player={hit_player_id}")

        stroke_result = stroke_classifier.classify(hit_ev)
        event = {
            "timestamp":       timestamp,
            "frame_idx":       frame_idx,
            "player_id":       hit_player_id,
            "stroke_type":     stroke_result.get("stroke_type"),
            "confidence":      stroke_result.get("confidence", 0.0),
            "tactical_semantic": stroke_result.get("tactical_semantic"),
            "decision_eval":   stroke_result.get("decision_eval"),
            "trajectory_pre":  hit_ev.trajectory_pre,
            "trajectory_post": [],
            "prediction_error": None,
        }
        events.append(event)

    def _process_frame(
        frame_bgr: np.ndarray,
        frame_idx: int,
        timestamp: float,
        shuttle_det_dict: dict,
    ) -> None:
        nonlocal final_timestamp
        final_timestamp = timestamp
        h, w = frame_bgr.shape[:2]
        if _frame_hw[0] is None:
            _frame_hw[0], _frame_hw[1] = h, w

        # Keep rolling window of raw frames for multi-frame stroke pose extraction (Phase 4-B)
        _frame_window.append((frame_bgr.copy(), frame_idx, timestamp))

        # 1. Shuttle
        shuttle_tuple = shuttle_det_dict.get("shuttle")  # (ts, x, y, w, h) or None
        shuttle: Shuttle | None = Shuttle.from_tuple(shuttle_tuple) if shuttle_tuple else None

        # 2. Player detection + real-world transform + history accumulation
        raw_players = yolo.detect(frame_bgr)
        players = player_ctx.update(raw_players, court_mapper)

        # 3. Hit detection
        # player_feet_real_list is only needed when the shuttle is visible —
        # HitDetector returns (False, None) immediately when shuttle_pos is None,
        # so there is no point computing it on the majority of frames where the
        # shuttle is not detected.
        shuttle_pos_px = shuttle.center_px if shuttle is not None else None
        player_feet_for_hit = (
            player_ctx.player_feet_real_list(players)
            if shuttle_pos_px is not None
            else []
        )
        is_hit, hit_player_id = hit_detector.update(
            timestamp,
            shuttle_pos_px,
            player_feet_for_hit,
        )

        # 4. Stroke classification on hit (Phase 4-B: deferred emission for multi-frame)
        if is_hit:
            # Collect pre-hit frames from the rolling window
            pre_frames = [f for f, _, _ in _frame_window]           # up to _STROKE_PRE_FRAMES + 1 frames
            pre_ts     = [t for _, _, t in _frame_window]

            hit_ev = HitEvent(
                keyframe=frame_bgr,
                trajectory_pre=[(e[0], e[1], e[2]) for e in hit_detector._buffer],
                trajectory_post=[],
                surrounding_frames=list(pre_frames),                 # pre-hit frames collected now
                surrounding_timestamps=list(pre_ts),
            )

            if _STROKE_POST_FRAMES > 0:
                # Defer emission: collect post-hit frames before classifying
                _pending_hits.append({
                    "hit_ev":        hit_ev,
                    "timestamp":     timestamp,
                    "frame_idx":     frame_idx,
                    "hit_player_id": hit_player_id,
                    "frames_left":   _STROKE_POST_FRAMES,
                })
            else:
                # No post frames needed — classify immediately (original behaviour)
                _emit_stroke_event(hit_ev, timestamp, frame_idx, hit_player_id)

        # Fulfil pending hits that are waiting for post-hit frames (Phase 4-B)
        still_pending = []
        for pending in _pending_hits:
            pending["hit_ev"].surrounding_frames.append(frame_bgr.copy())
            pending["hit_ev"].surrounding_timestamps.append(timestamp)
            pending["frames_left"] -= 1
            if pending["frames_left"] <= 0:
                _emit_stroke_event(
                    pending["hit_ev"],
                    pending["timestamp"],
                    pending["frame_idx"],
                    pending["hit_player_id"],
                )
            else:
                still_pending.append(pending)
        _pending_hits[:] = still_pending

        # 5. Update game state
        game_state.update(timestamp, shuttle_tuple, frame_size=(h, w))

        # 6. Record tracking result
        player_dicts = [p.to_dict() for p in players]
        # Ensure feet_px is populated for heatmap (Bug 5-C Cause B)
        for p_dict in player_dicts:
            if "feet_px" not in p_dict and "feet" in p_dict:
                p_dict["feet_px"] = p_dict["feet"]
        tracking_results.append({
            "frame_idx": frame_idx,
            "timestamp": timestamp,
            "shuttle": shuttle.to_dict() if shuttle is not None else None,
            "players": player_dicts,
            "rally_active": game_state.rally_active,
        })

        # 7. Render frame (only when --annotate is active)
        if annotate:
            feet_history = player_ctx.get_feet_history(max_players=2)
            court_insert = render_court_insert(feet_history, cfg) if feet_history else None
            # Convert Player dataclasses back to dicts that render_frame expects
            det_dicts = [
                {"id": p.id, "box": p.box, "feet": p.feet_px, "feet_real": p.feet_real}
                for p in players
            ]
            annotated = render_frame(
                frame_bgr, det_dicts, game_state.rally_active, court_insert, events, cfg
            )
            video_io.write_frame(annotated)

        if (frame_idx + 1) % 100 == 0:
            print(f"[MAIN] processed {frame_idx + 1} frames @ t={timestamp:.1f}s")

    def _flush_batch() -> None:
        """Run detect_batch() on the accumulated frame_buffer, then process each frame."""
        if not frame_buffer:
            return
        shuttle_dets = tracknet.detect_batch(
            [f for f, _, _ in frame_buffer],
            [t for _, _, t in frame_buffer],
        )
        for (fr, fi, ts), det in zip(frame_buffer, shuttle_dets):
            _process_frame(fr, fi, ts, det)
        frame_buffer.clear()

    for frame_bgr, frame_idx, timestamp in video_io.stream():
        if _use_batched:
            frame_buffer.append((frame_bgr, frame_idx, timestamp))
            if len(frame_buffer) >= BATCH:
                _flush_batch()
        else:
            # CPU path: sequential detect() — simpler and faster than batching on CPU.
            shuttle_det_dict = tracknet.detect(frame_bgr, timestamp)
            _process_frame(frame_bgr, frame_idx, timestamp, shuttle_det_dict)

    if _use_batched:
        _flush_batch()  # process any remaining frames (tail of video)

    # Flush pending hits that reached end of video without enough post frames (Phase 4-B)
    for pending in _pending_hits:
        _emit_stroke_event(
            pending["hit_ev"],
            pending["timestamp"],
            pending["frame_idx"],
            pending["hit_player_id"],
        )
    _pending_hits.clear()

    video_io.release()

    # ── Finalize ──────────────────────────────────────────────────────────────
    game_state.finalize_rally_data(final_timestamp)
    raw_rally_data = game_state.get_rally_data()

    # Second-pass consolidation: merge short inactive gaps (<0.5s) that slipped
    # past the in-loop grace period.  This turns the dead build_rally_status_per_frame
    # function into the active post-processor it was always meant to be.
    min_period_s = float(getattr(cfg, "rally_min_period_s", 0.5))
    _, rally_data = build_rally_status_per_frame(
        raw_rally_data,
        total_frames=len(tracking_results),
        fps=source_fps,
        min_period_s=min_period_s,
    )
    print(
        f"[MAIN] rally consolidation: {len(raw_rally_data)} raw → {len(rally_data)} consolidated"
    )

    analytics = analysis.compute_rally_statistics(rally_data)
    analysis.visualize_results((analytics, run_dir))

    # Per-player hit counts
    per_player_hits: dict = {}
    for ev in events:
        pid = ev.get("player_id")
        if pid is not None:
            key = str(pid)
            per_player_hits[key] = per_player_hits.get(key, 0) + 1

    analytics_out = {
        "rally_count": analytics["rally_count"],
        "mean_rally_duration_s": analytics["mean_rally_duration_s"],
        "min_rally_duration_s": analytics["min_rally_duration_s"],
        "max_rally_duration_s": analytics["max_rally_duration_s"],
        "total_rally_duration_s": analytics["total_rally_duration_s"],
        "per_player_hit_counts": per_player_hits,
        "rally_duration_histogram_path": os.path.join(run_dir, "rally_duration_histogram.png"),
    }

    # ── Write JSON outputs ────────────────────────────────────────────────────
    # TEMPORARY DIAGNOSTIC (Task 5-E)
    frames_processed = len(tracking_results)
    print(f"[MAIN] Hit events detected: {len(events)} across {frames_processed} frames "
          f"({len(events)/max(frames_processed,1)*30*60:.1f} hits/min at 30fps)")

    def _write_json(path, data):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    _write_json(os.path.join(run_dir, "tracking_results.json"), tracking_results)
    _write_json(os.path.join(run_dir, "events.json"), events)
    _write_json(os.path.join(run_dir, "rally_data.json"), rally_data)
    _write_json(os.path.join(run_dir, "analytics.json"), analytics_out)

    # Copy court_points.json to run dir
    if os.path.exists(court_points_file):
        shutil.copy2(court_points_file, os.path.join(run_dir, "court_points.json"))
    else:
        _write_json(os.path.join(run_dir, "court_points.json"), {})

    # Precompute player-footwork heatmap (static PNG loaded by viewer)
    try:
        from utils.precompute_heatmap import precompute_heatmap as _precompute_heatmap
        import json as _json
        _cp_path = os.path.join(run_dir, "court_points.json")
        with open(_cp_path, encoding="utf-8") as _f:
            _cp_data = _json.load(_f)
        _cp_points = next(iter(_cp_data.values())) if _cp_data else []

        if _cp_points and len(tracking_results) > 0:
            # Use actual video frame dimensions captured during processing.
            # Previous approach used player bbox corners (x2, y2) as a proxy,
            # which gave fw≈800 on a 1280px-wide video — breaking homography.
            if _frame_hw[0] is not None:
                _fh, _fw = _frame_hw[0], _frame_hw[1]
            else:
                _fh, _fw = 1080, 1920   # safe fallback

            _hm_path = os.path.join(run_dir, "heatmap.png")
            _precompute_heatmap(
                tracking_results=tracking_results,
                court_points=_cp_points,
                frame_w=_fw,
                frame_h=_fh,
                output_path=_hm_path,
                gaussian_sigma=float(getattr(cfg, "heatmap_gaussian_sigma", 40.0)),
                p1_color_bgr=tuple(getattr(cfg, "player_p1_color_bgr", [57, 255, 20])),
                p2_color_bgr=tuple(getattr(cfg, "player_p2_color_bgr", [0, 165, 255])),
            )
        else:
            print("[MAIN] Skipping heatmap: no court points or no tracking data.")
    except Exception as _e:
        print(f"[MAIN] Warning: heatmap precompute failed: {_e}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n[MAIN] ─── Run complete ───")
    print(f"  Frames processed : {len(tracking_results)}")
    print(f"  Rallies detected : {analytics['rally_count']}")
    print(f"  Hit events       : {len(events)}")
    print(f"  Output dir       : {run_dir}")
    if annotate:
        print(f"  Annotated video  : {os.path.join(run_dir, 'annotated_video.mp4')}")
    else:
        print(f"  (No annotated video — pass --annotate to generate one)")

    return run_dir


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Badminton vision pipeline — headless analysis of match video."
    )
    parser.add_argument("--video", required=True, help="Path to input video file.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    parser.add_argument(
        "--court-points",
        dest="court_points",
        default=None,
        help="Path to court_points.json (overrides config).",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        default=None,
        help="Output directory (overrides config).",
    )
    parser.add_argument("--fps", type=float, default=None, help="FPS override.")
    parser.add_argument("--device", default=None, help="Device override (cpu/cuda).")
    parser.add_argument(
        "--annotate",
        action="store_true",
        default=False,
        help="Write annotated_video.mp4 to the run directory (slow; opt-in).",
    )
    parser.add_argument(
        "--set",
        nargs="*",
        metavar="KEY=VALUE",
        help="Arbitrary config overrides (e.g. --set fps=60 device=cuda).",
    )
    args = parser.parse_args()

    overrides: dict = {}
    if args.fps is not None:
        overrides["fps"] = args.fps
    if args.device is not None:
        overrides["device"] = args.device
    if args.set:
        for kv in args.set:
            if "=" in kv:
                k, v = kv.split("=", 1)
                try:
                    v = int(v)
                except ValueError:
                    try:
                        v = float(v)
                    except ValueError:
                        pass
                overrides[k.strip()] = v

    run(
        video_path=args.video,
        config_path=args.config,
        court_points_path=args.court_points,
        output_dir=args.output_dir,
        overrides=overrides,
        annotate=args.annotate,
    )


if __name__ == "__main__":
    main()
