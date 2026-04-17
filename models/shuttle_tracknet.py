"""Shuttle tracking using TrackNet."""
import cv2
import numpy as np
import torch
import collections
import os

from models.TrackNet import TrackNet 

class ShuttleTracker:
    def __init__(self, weights_path="weights/track.pt", device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = TrackNet().to(self.device)
        self.input_buffer = collections.deque(maxlen=3)
        self.sequence_length = 3
        
        # --- NEW FILTER VARIABLES ---
        self.last_coord = None
        self.frames_since_last_seen = 0
        self.max_distance = 150  # Max pixels the bird can travel in 1 frame (adjust if needed)
        self.reset_frames = 30   # If lost for 30 frames (~0.5 sec), allow picking it up anywhere
        # ----------------------------

        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

    def _preprocess(self):
        stacked_frames = []
        for frame in self.input_buffer:
            resized = cv2.resize(frame, (512, 288))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            normalized = rgb.astype(np.float32) / 255.0
            transposed = np.transpose(normalized, (2, 0, 1))
            stacked_frames.append(transposed)
            
        # Concatenate the 3 frames (3 frames * 3 channels = 9 channels)
        concatenated = np.concatenate(stacked_frames, axis=0)
        tensor = torch.tensor(concatenated, dtype=torch.float32).unsqueeze(0).to(self.device)
        return tensor

    def update(self, frame):
        import math # Make sure math is imported at the top of your file!
        
        original_height, original_width = frame.shape[:2]
        self.input_buffer.append(frame)
        
        if len(self.input_buffer) < self.sequence_length:
            return None 
            
        input_tensor = self._preprocess()
        
        with torch.no_grad():
            heatmap_tensor = self.model(input_tensor)
            
        heatmap = heatmap_tensor[0, 2, :, :].cpu().numpy() 
        raw_coords = self._get_center_from_heatmap(heatmap, original_width, original_height)
        
        # --- THE PHYSICS FILTER ---
        if raw_coords is not None:
            if self.last_coord is None or self.frames_since_last_seen > self.reset_frames:
                # First time seeing it, or we lost it for a while. Accept it blindly.
                self.last_coord = raw_coords
                self.frames_since_last_seen = 0
                return raw_coords
            else:
                # We saw it recently. Check the distance!
                dist = math.dist(self.last_coord, raw_coords)
                if dist <= self.max_distance:
                    # It's a valid physics move! Update memory.
                    self.last_coord = raw_coords
                    self.frames_since_last_seen = 0
                    return raw_coords
                else:
                    # It teleported! This is a white shoe/line. Reject it.
                    self.frames_since_last_seen += 1
                    return None 
        else:
            # TrackNet saw nothing.
            self.frames_since_last_seen += 1
            return None

    def _get_center_from_heatmap(self, heatmap, orig_w, orig_h):
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(heatmap)
        if max_val > 0.5:
            pred_x, pred_y = max_loc
            scaled_x = int(pred_x * (orig_w / 512.0))
            scaled_y = int(pred_y * (orig_h / 288.0))
            return (scaled_x, scaled_y)
            
        return None