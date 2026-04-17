import cv2
import numpy as np

class Minimap:
    def __init__(self, court_width_cm=610, court_height_cm=1340, map_scale=0.3):
        # Scale: 0.3 means 100cm (1 meter) = 30 pixels
        self.w_cm = court_width_cm
        self.h_cm = court_height_cm
        self.scale = map_scale
        
        self.map_w = int(self.w_cm * self.scale)
        self.map_h = int(self.h_cm * self.scale)
        
        # Create a blank white court
        self.base_map = np.ones((self.map_h, self.map_w, 3), dtype=np.uint8) * 255
        self._draw_court_lines()

    def _draw_court_lines(self):
        # Draw standard badminton lines (scaled down)
        # Outer boundary
        cv2.rectangle(self.base_map, (0, 0), (self.map_w, self.map_h), (0, 0, 0), 2)
        
        # Net (Center)
        net_y = int(self.h_cm / 2 * self.scale)
        cv2.line(self.base_map, (0, net_y), (self.map_w, net_y), (0, 0, 255), 2)
        
        # Service lines, etc. (Simplified)
        short_serve_offset = int(198 * self.scale) # 1.98m from net
        cv2.line(self.base_map, (0, net_y - short_serve_offset), (self.map_w, net_y - short_serve_offset), (200, 200, 200), 1)
        cv2.line(self.base_map, (0, net_y + short_serve_offset), (self.map_w, net_y + short_serve_offset), (200, 200, 200), 1)

    def draw(self, frame, shuttle_cm, players_cm):
        # Create a fresh copy of the map
        minimap = self.base_map.copy()
        
        # Draw Players (Blue Dots)
        for px, py in players_cm:
            mx = int(px * self.scale)
            my = int(py * self.scale)
            cv2.circle(minimap, (mx, my), 5, (255, 0, 0), -1) # Blue
            
        # Draw Shuttle (Red Dot)
        # --- FIX IS HERE: Explicitly check "is not None" ---
        if shuttle_cm is not None:
            sx, sy = shuttle_cm
            mx = int(sx * self.scale)
            my = int(sy * self.scale)
            
            # Check bounds (Visualize "OUT" shots)
            color = (0, 0, 255) # Red
            if sx < 0 or sx > self.w_cm or sy < 0 or sy > self.h_cm:
                color = (0, 0, 128) # Dark Red (Out of bounds)
                
            cv2.circle(minimap, (mx, my), 4, color, -1)

        # Overlay minimap onto the main frame (Top-Left Corner)
        h, w, _ = minimap.shape
        frame[20:20+h, 20:20+w] = minimap
        return frame