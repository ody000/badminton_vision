"""Kinematics calculations."""
import math
from collections import deque

class KinematicsEngine:
    def __init__(self, history_size=5, proximity_threshold=200):
        # Stores the last N shuttle (x, y) real-world coordinates
        self.trajectory = deque(maxlen=history_size)
        
        # Max distance (in CENTIMETERS) a shuttle can be from a player to count as their hit
        # 200cm = 2 meters. Adjust based on your camera angle accuracy.
        self.proximity_threshold = proximity_threshold
        
    def update_and_check_hit(self, shuttle_coord, player_positions):
        """
        Takes the current frame's data and returns:
        (is_hit: bool, striking_player_index: int or None)
        
        shuttle_coord: (x, y) tuple in real-world cm
        player_positions: List of (x, y) tuples representing player feet in real-world cm
        """
        if shuttle_coord is None:
            return False, None
            
        self.trajectory.append(shuttle_coord)
        
        # We need at least 3 points of data to detect a V-shape change in direction
        if len(self.trajectory) < 3:
            return False, None
            
        is_hit = self._detect_direction_change()
        striking_player_idx = None
        
        if is_hit:
            striking_player_idx = self._find_nearest_player(shuttle_coord, player_positions)
            
            # If the hit was too far from any player, we might ignore it (or count it as a bounce)
            if striking_player_idx is not None:
                # Clear the trajectory after a valid hit so we don't trigger it twice
                self.trajectory.clear()
            else:
                # Optional: If it's a hit but no player is near, it might be a floor bounce!
                # You can handle that logic here later.
                pass 
            
        return is_hit, striking_player_idx
        
    def _detect_direction_change(self):
        """Looks for an abrupt reversal in X or Y velocity."""
        # p0 (oldest), p1 (middle), p2 (newest)
        p0 = self.trajectory[-3]
        p1 = self.trajectory[-2]
        p2 = self.trajectory[-1]
        
        # Calculate velocity vectors
        vec1_x = p1[0] - p0[0]
        vec2_x = p2[0] - p1[0]
        
        vec1_y = p1[1] - p0[1]
        vec2_y = p2[1] - p1[1]
        
        # Did the direction reverse? (Positive to negative, or negative to positive)
        x_reversal = (vec1_x * vec2_x) < 0
        y_reversal = (vec1_y * vec2_y) < 0
        
        # Ensure it's a significant movement (e.g., more than 5cm of travel)
        # This prevents tiny jitter from being counted as a hit
        significant_movement = (abs(vec1_x) + abs(vec1_y)) > 5
        
        return (x_reversal or y_reversal) and significant_movement

    def _find_nearest_player(self, shuttle_coord, player_positions):
        """Attributes the hit to the closest player based on feet position."""
        sx, sy = shuttle_coord
        min_dist = float('inf')
        closest_player_idx = None
        
        for i, (px, py) in enumerate(player_positions):
            # Pythagorean theorem to find distance between player feet and shuttle
            dist = math.hypot(sx - px, sy - py)
            
            if dist < min_dist and dist < self.proximity_threshold:
                min_dist = dist
                closest_player_idx = i
                
        return closest_player_idx