"""Analyses for shuttle and player tracking data.

Direct port of slayminton/core/analysis.py. No functional changes.
Focused on rally duration statistics and histogram generation.
"""

import os

import matplotlib.pyplot as plt


class Analysis:
    def __init__(self):
        self.last_results = None
        print("[ANALYSIS] initialized")

    def analyze_shuttle_trajectories(self, shuttle_detections):
        # Placeholder for trajectory analysis logic
        pass

    def compute_rally_statistics(self, rally_data):
        """Compute statistics for each rally.

        Key metrics: rally duration.
        Later extensions: number of hits, average shuttle speed,
        player movement distance, average time between hits.
        """
        print(f"[ANALYSIS] compute_rally_statistics input_rallies={len(rally_data or [])}")
        normalized = []
        for idx, item in enumerate(rally_data or []):
            if item is None:
                continue

            start_time = item.get("start_time", None)
            end_time = item.get("end_time", None)
            duration_s = item.get("duration_s", None)

            if duration_s is None and start_time is not None and end_time is not None:
                duration_s = float(end_time) - float(start_time)

            if duration_s is None:
                continue

            duration_s = max(float(duration_s), 0.0)
            normalized.append(
                {
                    "rally_id": int(item.get("rally_id", idx + 1)),
                    "start_time": None if start_time is None else float(start_time),
                    "end_time": None if end_time is None else float(end_time),
                    "duration_s": duration_s,
                }
            )

        durations = [r["duration_s"] for r in normalized]
        rally_count = len(durations)

        results = {
            "rallies": normalized,
            "rally_count": rally_count,
            "total_rally_duration_s": float(sum(durations)) if durations else 0.0,
            "mean_rally_duration_s": float(sum(durations) / rally_count) if rally_count else 0.0,
            "min_rally_duration_s": float(min(durations)) if durations else 0.0,
            "max_rally_duration_s": float(max(durations)) if durations else 0.0,
            "durations_s": durations,
        }

        self.last_results = results
        print(
            "[ANALYSIS] rally_stats "
            f"count={results['rally_count']} "
            f"mean={results['mean_rally_duration_s']:.3f}s "
            f"min={results['min_rally_duration_s']:.3f}s "
            f"max={results['max_rally_duration_s']:.3f}s"
        )
        return results

    def analyze_player_movements(self, player_detections):
        """Process player detections to extract movement patterns.

        Later: smooth trajectories, compute movement vectors, heatmaps.
        """
        pass

    def visualize_results(self, analysis_results):
        """Generate matplotlib visualizations.

        Key: distribution of rally durations (histogram).
        Later: heatmaps, hit positions.
        """
        if isinstance(analysis_results, tuple):
            stats = analysis_results[0] if len(analysis_results) > 0 else {}
            output_dir = analysis_results[1] if len(analysis_results) > 1 else "data/output"
        else:
            stats = analysis_results or {}
            output_dir = stats.get("output_dir", "data/output")

        os.makedirs(output_dir, exist_ok=True)
        durations = stats.get("durations_s", [])

        fig = plt.figure(figsize=(8, 5))
        if durations:
            bins = min(max(len(durations), 5), 20)
            plt.hist(durations, bins=bins, color="#3A86FF", edgecolor="black", alpha=0.85)
        else:
            plt.text(0.5, 0.5, "No rally durations available", ha="center", va="center")
            plt.xlim(0.0, 1.0)
            plt.ylim(0.0, 1.0)

        plt.xlabel("Rally Duration (seconds)")
        plt.ylabel("Count")
        plt.title("Distribution of Rally Durations")
        plt.grid(alpha=0.2)
        plt.tight_layout()

        histogram_path = os.path.join(output_dir, "rally_duration_histogram.png")
        plt.savefig(histogram_path, dpi=160)
        plt.close(fig)
        print(f"[ANALYSIS] saved_histogram path={histogram_path}")

        return {
            "rally_duration_histogram": histogram_path,
        }
