"""Court detection using ResNet."""
import torch
import cv2
import numpy as np
import torchvision.transforms as T

class CourtCalibrator:
    def __init__(self, weights_path):
        self.device = torch.device('cpu')
        print(f"Loading Court ResNet model onto {self.device}...")
        
        # 1. Load the architecture (Requires the specific repo's model file)
        # from models.resnet_court import ResNet50Keypoints
        # self.model = ResNet50Keypoints(num_keypoints=4)
        # self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        # self.model.eval()
        
        self.model = None

        # Standard ImageNet normalization used by ResNet
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def calibrate(self, frame):
        if self.model is None:
            print("WARNING: No ResNet weights found. Using mock court corners.")
            # Default corners for a standard 1080p camera angle
            return [[400, 250], [1520, 250], [1800, 950], [120, 950]]

        # Resize for ResNet (usually 224x224 or 512x512 depending on the training)
        orig_h, orig_w = frame.shape[:2]
        resized = cv2.resize(frame, (224, 224))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor and add batch dimension
        input_tensor = self.transform(rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Output shape is typically [1, 8] for 4 (x,y) pairs
            keypoints_normalized = self.model(input_tensor).squeeze().cpu().numpy()

        # Reshape to 4x2 array and scale back to original video resolution
        corners = []
        for i in range(0, 8, 2):
            x = int(keypoints_normalized[i] * orig_w)
            y = int(keypoints_normalized[i+1] * orig_h)
            corners.append([x, y])

        return corners