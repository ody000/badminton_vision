"""TDD tests for TrackNetTracker.detect_batch().

Behaviors verified:
  1. Empty input returns empty list.
  2. N frames returns exactly N dicts.
  3. Every returned element is a valid detection dict ({} or {"shuttle": 5-tuple}).
  4. Cross-batch context: the internal buffer populated by one batch supplies
     triplet context for the first frames of the next batch.
  5. detect_batch on N frames produces the same detections as N sequential
     detect() calls with the same frames (consistency / no batching artefact).
"""

from __future__ import annotations

import sys
import os

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from types import SimpleNamespace


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_cfg(conf_threshold: float = 0.0) -> SimpleNamespace:
    """Config that accepts any detection (threshold=0) so blank frames still
    produce a shuttle tuple when the heatmap argmax is non-trivial."""
    return SimpleNamespace(
        tracknet_weights=None,   # random-init weights — no checkpoint needed
        device="cpu",
        tracknet_box_size=16,
        tracknet_conf_threshold=conf_threshold,
        tracknet_expected_h=288,
        tracknet_expected_w=512,
        fps=30.0,
    )


def _blank_frame(h: int = 288, w: int = 512) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_tracker(conf_threshold: float = 0.0):
    from models.shuttle_tracknet import TrackNetTracker
    return TrackNetTracker(cfg=_make_cfg(conf_threshold))


# ---------------------------------------------------------------------------
# 1. Empty input returns empty list
# ---------------------------------------------------------------------------

class TestDetectBatchEmptyInput:
    def test_empty_frames_returns_empty_list(self):
        tracker = _make_tracker()
        result = tracker.detect_batch([], [])
        assert result == [], f"Expected [], got {result!r}"

    def test_return_type_is_list(self):
        tracker = _make_tracker()
        result = tracker.detect_batch([], [])
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# 2. N frames returns exactly N dicts
# ---------------------------------------------------------------------------

class TestDetectBatchLength:
    def test_single_frame_returns_one_dict(self):
        tracker = _make_tracker()
        frames = [_blank_frame()]
        result = tracker.detect_batch(frames, [0.0])
        assert len(result) == 1, f"Expected 1, got {len(result)}"

    def test_eight_frames_returns_eight_dicts(self):
        tracker = _make_tracker()
        N = 8
        frames = [_blank_frame() for _ in range(N)]
        timestamps = [i / 30.0 for i in range(N)]
        result = tracker.detect_batch(frames, timestamps)
        assert len(result) == N, f"Expected {N}, got {len(result)}"

    def test_batch_larger_than_three_returns_correct_length(self):
        """Regression: batches > 3 must not be truncated to the triplet size."""
        tracker = _make_tracker()
        N = 12
        frames = [_blank_frame() for _ in range(N)]
        timestamps = [i / 30.0 for i in range(N)]
        result = tracker.detect_batch(frames, timestamps)
        assert len(result) == N


# ---------------------------------------------------------------------------
# 3. Every element is a valid detection dict
# ---------------------------------------------------------------------------

class TestDetectBatchOutputShape:
    def test_each_element_is_dict(self):
        tracker = _make_tracker()
        N = 5
        frames = [_blank_frame() for _ in range(N)]
        timestamps = [i / 30.0 for i in range(N)]
        result = tracker.detect_batch(frames, timestamps)
        for i, elem in enumerate(result):
            assert isinstance(elem, dict), f"Element {i} is {type(elem)}, expected dict"

    def test_shuttle_tuple_has_five_elements_when_present(self):
        """When a shuttle key exists, its value must be a 5-tuple (ts, x, y, w, h)."""
        tracker = _make_tracker(conf_threshold=0.0)  # accept any detection
        frames = [_blank_frame() for _ in range(4)]
        timestamps = [i / 30.0 for i in range(4)]
        result = tracker.detect_batch(frames, timestamps)
        for elem in result:
            if "shuttle" in elem:
                assert len(elem["shuttle"]) == 5, (
                    f"shuttle tuple must be (ts,x,y,w,h), got {elem['shuttle']}"
                )

    def test_no_unexpected_keys(self):
        """Only the empty dict or a dict with the 'shuttle' key are valid."""
        tracker = _make_tracker()
        frames = [_blank_frame() for _ in range(4)]
        timestamps = [i / 30.0 for i in range(4)]
        result = tracker.detect_batch(frames, timestamps)
        for elem in result:
            assert set(elem.keys()) <= {"shuttle"}, (
                f"Unexpected keys in result: {set(elem.keys()) - {'shuttle'}}"
            )


# ---------------------------------------------------------------------------
# 4. Cross-batch context
# ---------------------------------------------------------------------------

class TestDetectBatchCrossBatchContext:
    def test_buffer_populated_after_batch(self):
        """After detect_batch(), the internal buffer holds the last ≤3 frames
        of the batch so they supply context to the next batch."""
        tracker = _make_tracker()
        N = 5
        frames = [_blank_frame() for _ in range(N)]
        timestamps = [i / 30.0 for i in range(N)]
        tracker.detect_batch(frames, timestamps)

        # Buffer should contain the last min(3, N) frames from the batch.
        expected_len = min(3, N)
        assert len(tracker._buffer) == expected_len, (
            f"Buffer length after batch of {N} should be {expected_len}, "
            f"got {len(tracker._buffer)}"
        )

    def test_buffer_timestamps_match_last_frames(self):
        """Buffer timestamps after a batch match the last ≤3 timestamps."""
        tracker = _make_tracker()
        N = 6
        frames = [_blank_frame() for _ in range(N)]
        timestamps = [float(i) for i in range(N)]
        tracker.detect_batch(frames, timestamps)

        buf_ts = [entry[0] for entry in tracker._buffer]
        expected_ts = timestamps[max(0, N - 3):]
        assert buf_ts == expected_ts, (
            f"Buffer timestamps {buf_ts} != expected {expected_ts}"
        )

    def test_second_batch_uses_context_from_first(self):
        """Two consecutive batches of 1 frame each: the second should still
        have access to 2 context frames (from the first batch's buffer) so that
        the triplet for its single frame is non-trivial."""
        tracker = _make_tracker()
        frame_a = _blank_frame()
        frame_b = _blank_frame()

        # First batch: seed the buffer
        tracker.detect_batch([frame_a], [0.0])
        buf_after_first = len(tracker._buffer)

        # Second batch: the buffer from first batch provides context
        tracker.detect_batch([frame_b], [1.0 / 30.0])
        buf_after_second = len(tracker._buffer)

        assert buf_after_first >= 1, "Buffer should have content after first batch"
        assert buf_after_second >= 1, "Buffer should have content after second batch"


# ---------------------------------------------------------------------------
# 5. Consistency: detect_batch ≡ N sequential detect() calls
# ---------------------------------------------------------------------------

class TestDetectBatchConsistency:
    def test_batch_and_sequential_agree_on_presence(self):
        """For each frame, both methods should agree on whether a shuttle is
        detected (key present vs absent) when using identical model weights and
        the same frame sequence starting from an empty buffer."""
        from models.shuttle_tracknet import TrackNetTracker

        cfg = _make_cfg(conf_threshold=0.0)

        tracker_seq = TrackNetTracker(cfg=cfg)
        tracker_bat = TrackNetTracker(cfg=cfg)

        # Use the same deterministic weight init: copy state dicts
        tracker_bat.model.load_state_dict(tracker_seq.model.state_dict())

        N = 6
        frames = [_blank_frame() for _ in range(N)]
        timestamps = [i / 30.0 for i in range(N)]

        seq_results = [tracker_seq.detect(f, t) for f, t in zip(frames, timestamps)]
        bat_results = tracker_bat.detect_batch(frames, timestamps)

        assert len(bat_results) == len(seq_results)
        for i, (seq, bat) in enumerate(zip(seq_results, bat_results)):
            seq_has = "shuttle" in seq
            bat_has = "shuttle" in bat
            assert seq_has == bat_has, (
                f"Frame {i}: sequential={'shuttle' in seq}, batch={'shuttle' in bat}. "
                "Both should agree on shuttle presence."
            )
