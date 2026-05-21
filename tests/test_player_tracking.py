"""Test centroid-based player tracking to verify ID persistence."""

import numpy as np
from types import SimpleNamespace
from unittest.mock import Mock, patch


# Mock ultralytics.YOLO to avoid import errors in test environment.
import sys
sys.modules['ultralytics'] = Mock()


from models.player_yolo import PlayerDetector


def test_persistent_ids_same_order():
    """Test that IDs remain stable when players move but detection order doesn't change."""
    cfg = SimpleNamespace(
        player_weights="models/yolo.pt",
        player_conf_threshold=0.5,
        device="cpu",
        player_detect_interval=1,
    )
    detector = PlayerDetector(cfg=cfg)

    # Create fake frames with two detections (side by side).
    # Frame 1: Player at (100, 200) and (400, 200)
    fake_frame_1 = np.zeros((480, 640, 3), dtype=np.uint8)

    # Manually set up state for first frame (simulating YOLO output).
    # We'll directly call _match_and_assign_ids with fake detections.
    raw_dets_1 = [
        {"box": [80, 180, 120, 220], "feet": (100, 220), "centroid": (100, 200)},
        {"box": [380, 180, 420, 220], "feet": (400, 220), "centroid": (400, 200)},
    ]
    dets_1 = detector._match_and_assign_ids(raw_dets_1)

    # Verify first frame assigned IDs 0 and 1.
    assert len(dets_1) == 2
    id_0, id_1 = dets_1[0]["id"], dets_1[1]["id"]
    assert id_0 != id_1, "IDs should be different"
    assert id_0 in [0, 1] and id_1 in [0, 1], "First frame should assign IDs 0, 1"

    # Frame 2: Players move slightly but detection order is same.
    # Player 1 moves to (110, 210), Player 2 moves to (410, 210).
    raw_dets_2 = [
        {"box": [90, 190, 130, 230], "feet": (110, 230), "centroid": (110, 210)},
        {"box": [390, 190, 430, 230], "feet": (410, 230), "centroid": (410, 210)},
    ]
    dets_2 = detector._match_and_assign_ids(raw_dets_2)

    # Verify IDs match the first frame (same player, same ID).
    assert dets_2[0]["id"] == id_0, f"Player 1 ID should remain {id_0}, got {dets_2[0]['id']}"
    assert dets_2[1]["id"] == id_1, f"Player 2 ID should remain {id_1}, got {dets_2[1]['id']}"
    print("✓ Test passed: IDs remain stable across frames")


def test_persistent_ids_swapped_order():
    """Test that IDs are maintained even when detection order is swapped (ID swapping bug)."""
    cfg = SimpleNamespace(
        player_weights="models/yolo.pt",
        player_conf_threshold=0.5,
        device="cpu",
        player_detect_interval=1,
    )
    detector = PlayerDetector(cfg=cfg)

    # Frame 1: Two players at (100, 200) and (400, 200), detected in order.
    raw_dets_1 = [
        {"box": [80, 180, 120, 220], "feet": (100, 220), "centroid": (100, 200)},
        {"box": [380, 180, 420, 220], "feet": (400, 220), "centroid": (400, 200)},
    ]
    dets_1 = detector._match_and_assign_ids(raw_dets_1)
    id_left_1 = dets_1[0]["id"]  # Should be 0
    id_right_1 = dets_1[1]["id"]  # Should be 1

    # Frame 2: Same players move, but detection order is REVERSED (this causes swaps).
    # Player at (400, 200) is now detected first, then player at (100, 200).
    raw_dets_2 = [
        {"box": [390, 190, 430, 230], "feet": (410, 230), "centroid": (410, 210)},
        {"box": [90, 190, 130, 230], "feet": (110, 230), "centroid": (110, 210)},
    ]
    dets_2 = detector._match_and_assign_ids(raw_dets_2)
    id_det_0_2 = dets_2[0]["id"]
    id_det_1_2 = dets_2[1]["id"]

    # CRITICAL: Even though detections are in swapped order, IDs should match based on centroid.
    # Detection index 0 in frame 2 is at (410, 210), which is close to (400, 200) from frame 1.
    # So dets_2[0] should have the same ID as dets_1[1] (id_right_1).
    # And dets_2[1] should have the same ID as dets_1[0] (id_left_1).

    assert id_det_0_2 == id_right_1, (
        f"Swapped detection should match by position, not order. "
        f"Expected {id_right_1}, got {id_det_0_2}"
    )
    assert id_det_1_2 == id_left_1, (
        f"Swapped detection should match by position, not order. "
        f"Expected {id_left_1}, got {id_det_1_2}"
    )
    print("✓ Test passed: IDs match by centroid, not by detection order (fixes swapping bug)")


def test_new_player_entry():
    """Test that a new player entry gets a new ID, not reassigned."""
    cfg = SimpleNamespace(
        player_weights="models/yolo.pt",
        player_conf_threshold=0.5,
        device="cpu",
        player_detect_interval=1,
    )
    detector = PlayerDetector(cfg=cfg)

    # Frame 1: One player.
    raw_dets_1 = [
        {"box": [80, 180, 120, 220], "feet": (100, 220), "centroid": (100, 200)},
    ]
    dets_1 = detector._match_and_assign_ids(raw_dets_1)
    id_1 = dets_1[0]["id"]
    assert id_1 == 0

    # Frame 2: Same player + new player.
    raw_dets_2 = [
        {"box": [80, 180, 120, 220], "feet": (100, 220), "centroid": (100, 200)},  # Same player
        {"box": [380, 180, 420, 220], "feet": (400, 220), "centroid": (400, 200)},  # New player
    ]
    dets_2 = detector._match_and_assign_ids(raw_dets_2)

    # First detection should keep ID 0.
    assert dets_2[0]["id"] == id_1
    # Second detection should get a new ID (1).
    assert dets_2[1]["id"] == 1
    print("✓ Test passed: New players get new IDs")


def test_player_exit():
    """Test that when a player leaves, their ID is freed."""
    cfg = SimpleNamespace(
        player_weights="models/yolo.pt",
        player_conf_threshold=0.5,
        device="cpu",
        player_detect_interval=1,
    )
    detector = PlayerDetector(cfg=cfg)

    # Frame 1: Two players.
    raw_dets_1 = [
        {"box": [80, 180, 120, 220], "feet": (100, 220), "centroid": (100, 200)},
        {"box": [380, 180, 420, 220], "feet": (400, 220), "centroid": (400, 200)},
    ]
    dets_1 = detector._match_and_assign_ids(raw_dets_1)
    assert len(dets_1) == 2

    # Frame 2: Only one player (left one stays, right one exits).
    raw_dets_2 = [
        {"box": [80, 180, 120, 220], "feet": (100, 220), "centroid": (100, 200)},
    ]
    dets_2 = detector._match_and_assign_ids(raw_dets_2)
    assert len(dets_2) == 1
    assert dets_2[0]["id"] == dets_1[0]["id"], "Remaining player should keep ID"

    # Verify exited player ID was removed from tracking.
    assert len(detector._tracked_ids) == 1, "Exited player ID should be removed"
    print("✓ Test passed: Exited players are removed from tracking")


if __name__ == "__main__":
    test_persistent_ids_same_order()
    test_persistent_ids_swapped_order()
    test_new_player_entry()
    test_player_exit()
    print("\n✅ All player tracking tests passed!")
