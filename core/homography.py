"""Homography transformation utilities."""
import cv2
import numpy as np

class CourtMapper:
    def __init__(self):
        self.homography_matrix = None
        
        # Standard badminton court dimensions in centimeters (1340cm x 610cm)
        # 0,0 is the top-left corner (Net, Left Side)
        self.real_world_corners = np.array([
            [0, 0],           # Top-Left
            [610, 0],         # Top-Right
            [610, 1340],      # Bottom-Right
            [0, 1340]         # Bottom-Left
        ], dtype=np.float32)

    def calibrate(self, image_corners):
        """
        Generates the transformation matrix.
        image_corners: List of 4 (x,y) pixel points [TL, TR, BR, BL]
        """
        image_corners_np = np.array(image_corners, dtype=np.float32)
        self.homography_matrix, _ = cv2.findHomography(image_corners_np, self.real_world_corners)

    def transform_point(self, point):
        """Transforms a single (x, y) pixel point to real-world cm."""
        if self.homography_matrix is None or point is None:
            return None
            
        # OpenCV needs the point in shape (1, 1, 2)
        p = np.array([[[point[0], point[1]]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(p, self.homography_matrix)
        
        # Return as simple tuple (x, y)
        return transformed[0][0]

    def get_player_feet(self, player_box):
        """Calculates the center of the player's feet from the bounding box."""
        x1, y1, x2, y2 = player_box
        feet_x = (x1 + x2) / 2
        feet_y = y2
        return (feet_x, feet_y)