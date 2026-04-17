"""Player detection using YOLO."""
from ultralytics import YOLO

class PlayerDetector:
    def __init__(self, weights_path='yolov8n.pt', conf_threshold=0.5):
        print(f"Loading YOLOv8 model from {weights_path} onto CPU...")
        # Ultralytics will automatically download 'yolov8n.pt' from the internet 
        # the very first time you run this, if it's not already in your folder.
        self.model = YOLO(weights_path)
        self.conf_threshold = conf_threshold
        self.person_class_id = 0 # COCO dataset ID for 'person'

    def detect(self, frame):
        """
        Runs inference and returns a list of bounding boxes for players.
        Returns: List of [x1, y1, x2, y2] coordinates.
        """
        # Run YOLO on the frame. 
        # device='cpu' forces it to use your Intel Mac CPU.
        # verbose=False stops YOLO from spamming your terminal every single frame.
        results = self.model.predict(
            frame, 
            classes=[self.person_class_id], 
            conf=self.conf_threshold, 
            device='cpu', 
            verbose=False
        )
        
        player_boxes = []
        for result in results:
            # result.boxes.xyxy contains the [x1, y1, x2, y2] tensors
            for box in result.boxes.xyxy:
                # Convert the PyTorch tensor to a standard Python list of integers
                x1, y1, x2, y2 = map(int, box.tolist())
                
                # Optional: Add logic here to filter out people sitting in the background
                # (e.g., umpires or crowd) by checking if the foot coordinate (y2) 
                # is within the bounds of your court homography matrix.
                
                player_boxes.append([x1, y1, x2, y2])
                
        return player_boxes