"""Visualization utilities."""
import cv2
import numpy as np
from collections import deque

class Visualizer:
    def __init__(self):
        # Store recent shuttle coordinates to draw a tail/trajectory
        self.shuttle_trajectory = deque(maxlen=20)
        
        # Define colors (B, G, R)
        self.COLOR_PLAYER = (0, 255, 0)   # Green
        self.COLOR_SHUTTLE = (0, 0, 255)  # Red
        self.COLOR_HIT = (0, 255, 255)    # Yellow
        self.COLOR_TEXT = (255, 255, 255) # White

    def draw_frame(self, frame, player_boxes, shuttle_coord, is_hit=False, game_stats=None):
        """
        Draws all tracked objects and stats onto the current frame.
        """
        canvas = frame.copy()
        
        # 1. Draw Player Bounding Boxes
        for box in player_boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), self.COLOR_PLAYER, 2)
            # Draw dots at their feet to show their "grounded" position
            feet_center = (int((x1 + x2) / 2), y2)
            cv2.circle(canvas, feet_center, 5, self.COLOR_PLAYER, -1)

        # 2. Draw Shuttlecock Trajectory
        if shuttle_coord:
            self.shuttle_trajectory.append(shuttle_coord)
            
            # Draw a line connecting recent points
            for i in range(1, len(self.shuttle_trajectory)):
                if self.shuttle_trajectory[i-1] is None or self.shuttle_trajectory[i] is None:
                    continue
                cv2.line(canvas, self.shuttle_trajectory[i-1], self.shuttle_trajectory[i], self.COLOR_SHUTTLE, 3)
            
            # Draw a circle at the current position
            cv2.circle(canvas, shuttle_coord, 8, self.COLOR_SHUTTLE, -1)

        # 3. Visual Feedback for a Hit
        if is_hit:
            cv2.putText(canvas, "HIT DETECTED!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, self.COLOR_HIT, 3)

        # 4. Draw Game Stats overlay (if provided by GameState)
        if game_stats:
            stats_text = f"Rally Status: {'ACTIVE' if game_stats['is_active'] else 'WAITING'} | Shots: {game_stats['shot_count']}"
            cv2.rectangle(canvas, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1) # Black background banner
            cv2.putText(canvas, stats_text, (20, 30), cv2.FONT_HERSHEY_PLAIN, 2, self.COLOR_TEXT, 2)

        return canvas

    def draw_minimap(self, player_2d_positions, court_dims=(305, 670)):
        """
        Creates a separate top-down 2D image of the court with player positions.
        court_dims: Width/Height of the minimap image (half-scale of real world mm).
        """
        # Create a green court background
        minimap = np.full((court_dims[1], court_dims[0], 3), (34, 139, 34), dtype=np.uint8)
        
        # Draw court lines (simplified for demo)
        cv2.rectangle(minimap, (0, 0), (court_dims[0]-1, court_dims[1]-1), (255, 255, 255), 2)
        cv2.line(minimap, (0, court_dims[1]//2), (court_dims[0], court_dims[1]//2), (255, 255, 255), 2) # Net
        
        # Plot players
        for px, py in player_2d_positions:
            # Scale positions down to minimap size
            scaled_x = int(px / 2)
            scaled_y = int(py / 2)
            
            # Ensure they are within bounds before drawing
            if 0 <= scaled_x < court_dims[0] and 0 <= scaled_y < court_dims[1]:
                 cv2.circle(minimap, (scaled_x, scaled_y), 8, (0, 0, 255), -1) # Red dots for players
                 
        return minimap