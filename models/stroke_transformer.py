"""Stroke classification using Transformer."""
import torch
import numpy as np
from ultralytics import YOLO

class StrokeClassifier:
    def __init__(self, transformer_weights_path):
        self.device = torch.device('cpu')
        
        # 1. The Pose Estimator (Extracts the skeleton)
        # YOLOv8 has a built-in pose model that is fast and lightweight
        print("Loading YOLOv8-Pose for skeleton extraction...")
        self.pose_model = YOLO('yolov8n-pose.pt') 
        
        # 2. The Transformer (Classifies the swing)
        print("Loading Stroke Transformer...")
        # from models.transformer import SpatialTemporalTransformer
        # self.transformer = SpatialTemporalTransformer(num_classes=6) # e.g., Clear, Drop, Smash, Lift, Net, Drive
        # self.transformer.load_state_dict(torch.load(transformer_weights_path, map_location=self.device))
        # self.transformer.eval()
        self.transformer = None
        
        self.classes = ["Clear", "Drop", "Smash", "Lift", "Net Push", "Drive"]

    def predict(self, frame_sequence, player_box):
        """
        frame_sequence: List of ~15 frames around the hit event.
        player_box: The [x1, y1, x2, y2] of the player who hit the shuttle.
        """
        if self.transformer is None:
            return "Smash (Mock)"

        skeleton_sequence = []
        x1, y1, x2, y2 = player_box

        # Extract skeleton for every frame in the sequence
        for frame in frame_sequence:
            # Crop the frame down to just the striking player to save processing power
            # Add a small padding (e.g., 20 pixels) so we don't cut off their racket
            crop = frame[max(0, y1-20):y2+20, max(0, x1-20):x2+20]
            
            # Run Pose Estimation
            results = self.pose_model.predict(crop, device='cpu', verbose=False)
            
            if len(results) > 0 and results[0].keypoints is not None:
                # Extract the 17 standard human keypoints (x, y, confidence)
                keypoints = results[0].keypoints.data[0].cpu().numpy() 
                skeleton_sequence.append(keypoints)
            else:
                # If pose fails on a frame, append zeros to maintain sequence length
                skeleton_sequence.append(np.zeros((17, 3)))

        # Convert the sequence of skeletons into a PyTorch tensor
        # Shape: (Batch=1, Frames=15, Keypoints=17, Features=3)
        input_tensor = torch.tensor(skeleton_sequence, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Transformer outputs probabilities for each shot class
            logits = self.transformer(input_tensor)
            predicted_class_idx = torch.argmax(logits, dim=1).item()

        return self.classes[predicted_class_idx]