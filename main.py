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
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


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
    from utils.mog import MOG2Manager
    from utils.video_io import VideoIOHandler
    from utils.visualization import render_court_insert, render_frame
    from models.shuttle_tracknet import TrackNetTracker
    from models.player_yolo import PlayerDetector
    from models.stroke_classifier import StrokeClassifier
    from core.homography import CourtMapper
    from core.game_state import GameState
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
    mog2 = MOG2Manager(
        var_threshold=float(getattr(cfg, "mog2_var_threshold", 200)),
        history=int(getattr(cfg, "mog2_history", 1000)),
    )

    tracknet = TrackNetTracker(cfg=cfg, mog2_manager=mog2)
    yolo = PlayerDetector(cfg=cfg, mog2_manager=mog2)

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

    # ── Main loop ─────────────────────────────────────────────────────────────
    for frame_bgr, frame_idx, timestamp in video_io.stream():
        final_timestamp = timestamp
        h, w = frame_bgr.shape[:2]

        # 1. MOG2 update
        yolo.update_mog2(frame_bgr)

        # 2. Shuttle detection
        shuttle_det_dict = tracknet.detect(frame_bgr, timestamp)
        shuttle_tuple = shuttle_det_dict.get("shuttle")  # (ts, x, y, w, h) or None
        shuttle: Shuttle | None = Shuttle.from_tuple(shuttle_tuple) if shuttle_tuple else None

        # 3. Player detection + real-world transform + history accumulation
        #    PlayerContext replaces the three inline dicts that were here before.
        raw_players = yolo.detect(frame_bgr)
        players = player_ctx.update(raw_players, court_mapper)

        # 4. Hit detection
        shuttle_pos_px = shuttle.center_px if shuttle is not None else None
        is_hit, hit_player_id = hit_detector.update(
            timestamp,
            shuttle_pos_px,
            player_ctx.player_feet_real_list(players),
        )

        # 5. Stroke classification on hit
        if is_hit:
            traj_pre = list(hit_detector._buffer)  # [(ts, x, y), ...]
            hit_ev = HitEvent(
                keyframe=frame_bgr,
                trajectory_pre=[(e[0], e[1], e[2]) for e in traj_pre],
                trajectory_post=[],
            )
            stroke_result = stroke_classifier.classify(hit_ev)
            event = {
                "timestamp": timestamp,
                "frame_idx": frame_idx,
                "player_id": hit_player_id,
                "stroke_type": stroke_result.get("stroke_type"),
                "confidence": stroke_result.get("confidence", 0.0),
                "tactical_semantic": stroke_result.get("tactical_semantic"),
                "decision_eval": stroke_result.get("decision_eval"),
                "trajectory_pre": hit_ev.trajectory_pre,
                "trajectory_post": [],
                "prediction_error": None,
            }
            events.append(event)

        # 6. Update game state
        game_state.update(timestamp, shuttle_tuple, frame_size=(h, w))

        # 7. Record tracking result
        tracking_results.append({
            "frame_idx": frame_idx,
            "timestamp": timestamp,
            "shuttle": shuttle.to_dict() if shuttle is not None else None,
            "players": [p.to_dict() for p in players],
            "rally_active": game_state.rally_active,
        })

        # 8. Render frame (only when --annotate is active)
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

    video_io.release()

    # ── Finalize ──────────────────────────────────────────────────────────────
    game_state.finalize_rally_data(final_timestamp)
    rally_data = game_state.get_rally_data()
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
