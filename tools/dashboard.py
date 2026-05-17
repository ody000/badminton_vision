"""Streamlit dashboard for reviewing badminton_vision pipeline outputs.

Reads JSON artifacts from a run output directory. Never re-runs the pipeline.

Tabs:
  1. Rally Timeline: bar chart of rally segments
  2. Shots: table of events.json
  3. Heatmaps: player coverage summary from analytics.json
  4. Kinematics: shuttle speed over time from tracking_results.json

Usage:
    streamlit run dashboard.py
"""

from __future__ import annotations

import json
import os

import streamlit as st


# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Badminton Vision Dashboard",
    page_icon="🏸",
    layout="wide",
)

st.title("Badminton Vision Dashboard")
st.caption("Reads pipeline output JSONs. Never re-runs the pipeline.")


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar: directory picker
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Run Directory")
    # Compute default path relative to script location
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(_script_dir)
    _default_data_output = os.path.join(_project_root, "data/output")
    run_dir = st.text_input(
        "Path to run output folder",
        value=_default_data_output,
        help="Select the {video_stem}_{timestamp}/ directory produced by main.py",
    )

    # Auto-detect sub-runs
    candidate_dirs = []
    if os.path.isdir(run_dir):
        for entry in sorted(os.listdir(run_dir), reverse=True):
            full = os.path.join(run_dir, entry)
            if os.path.isdir(full) and os.path.exists(os.path.join(full, "analytics.json")):
                candidate_dirs.append(full)

    if candidate_dirs:
        selected_run = st.selectbox("Or select a sub-run:", ["(use path above)"] + candidate_dirs)
        if selected_run != "(use path above)":
            run_dir = selected_run

    st.caption(f"Active dir: `{run_dir}`")


# ─────────────────────────────────────────────────────────────────────────────
# Load JSON files
# ─────────────────────────────────────────────────────────────────────────────

def _load_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


rally_data = _load_json(os.path.join(run_dir, "rally_data.json")) or []
events_data = _load_json(os.path.join(run_dir, "events.json")) or []
analytics_data = _load_json(os.path.join(run_dir, "analytics.json")) or {}
tracking_data = _load_json(os.path.join(run_dir, "tracking_results.json")) or []


# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(["Rally Timeline", "Shots", "Heatmaps", "Kinematics"])


# ── Tab 1: Rally Timeline ─────────────────────────────────────────────────────
with tab1:
    st.subheader("Rally Timeline")

    if not rally_data:
        st.info("No rally data found. Run the pipeline first.")
    else:
        import pandas as pd

        df = pd.DataFrame(rally_data)
        st.metric("Total rallies", len(df))
        col1, col2, col3 = st.columns(3)
        col1.metric("Mean duration (s)", f"{analytics_data.get('mean_rally_duration_s', 0.0):.2f}")
        col2.metric("Min duration (s)", f"{analytics_data.get('min_rally_duration_s', 0.0):.2f}")
        col3.metric("Max duration (s)", f"{analytics_data.get('max_rally_duration_s', 0.0):.2f}")

        st.subheader("Rally Segments")
        try:
            import altair as alt

            chart_data = df[["rally_id", "start_time", "end_time", "duration_s"]].copy()
            bars = (
                alt.Chart(chart_data)
                .mark_bar()
                .encode(
                    x=alt.X("start_time:Q", title="Time (s)"),
                    x2="end_time:Q",
                    y=alt.Y("rally_id:O", title="Rally ID"),
                    color=alt.value("#3A86FF"),
                    tooltip=["rally_id", "start_time", "end_time", "duration_s"],
                )
                .properties(height=max(200, len(df) * 30))
            )
            st.altair_chart(bars, use_container_width=True)
        except ImportError:
            st.bar_chart(df.set_index("rally_id")["duration_s"])

        st.dataframe(df, use_container_width=True)

        # Show histogram image if available
        hist_path = analytics_data.get("rally_duration_histogram_path")
        if hist_path and os.path.exists(str(hist_path)):
            st.image(str(hist_path), caption="Rally Duration Histogram")


# ── Tab 2: Shots ─────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Shot Events")

    if not events_data:
        st.info("No shot events found in events.json.")
    else:
        import pandas as pd

        df_ev = pd.DataFrame(events_data)
        cols_show = ["timestamp", "frame_idx", "player_id", "stroke_type", "confidence",
                     "tactical_semantic", "decision_eval"]
        available = [c for c in cols_show if c in df_ev.columns]
        st.metric("Total hits", len(df_ev))
        st.dataframe(df_ev[available], use_container_width=True)

        if "stroke_type" in df_ev.columns:
            st.subheader("Stroke Type Distribution")
            counts = df_ev["stroke_type"].value_counts()
            st.bar_chart(counts)


# ── Tab 3: Heatmaps ──────────────────────────────────────────────────────────
with tab3:
    st.subheader("Player Coverage (Placeholder)")

    per_player = analytics_data.get("per_player_hit_counts", {})
    if per_player:
        st.subheader("Hit Counts per Player")
        import pandas as pd

        df_hits = pd.DataFrame(
            [{"Player ID": k, "Hits": v} for k, v in per_player.items()]
        )
        st.dataframe(df_hits, use_container_width=True)
        st.bar_chart(df_hits.set_index("Player ID")["Hits"])
    else:
        st.info("No player hit count data available.")

    st.caption(
        "Full heatmap rendering uses the court insert from the annotated video. "
        "Run the pipeline with court calibration to enable this feature."
    )


# ── Tab 4: Kinematics ────────────────────────────────────────────────────────
with tab4:
    st.subheader("Shuttle Speed Over Time")

    if not tracking_data:
        st.info("No tracking data found in tracking_results.json.")
    else:
        import pandas as pd
        import math

        speeds = []
        prev_shuttle = None
        prev_ts = None

        for frame in tracking_data:
            ts = frame.get("timestamp", 0.0)
            s = frame.get("shuttle")
            if s is not None and prev_shuttle is not None and prev_ts is not None:
                dt = ts - prev_ts
                if dt > 0:
                    cx = s["x"] + s["w"] / 2
                    cy = s["y"] + s["h"] / 2
                    px = prev_shuttle["x"] + prev_shuttle["w"] / 2
                    py = prev_shuttle["y"] + prev_shuttle["h"] / 2
                    dist = math.hypot(cx - px, cy - py)
                    speed = dist / dt  # px/s
                    speeds.append({"timestamp": ts, "speed_px_per_s": speed})
            if s is not None:
                prev_shuttle = s
                prev_ts = ts

        if speeds:
            df_speed = pd.DataFrame(speeds)
            st.line_chart(df_speed.set_index("timestamp")["speed_px_per_s"])
            avg_speed = df_speed["speed_px_per_s"].mean()
            st.metric("Average shuttle speed (px/s)", f"{avg_speed:.1f}")
        else:
            st.info("Not enough shuttle detections to compute speed.")
