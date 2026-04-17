import cv2
import numpy as np
import os
from ultralytics import YOLO

def create_dummy_video(filename="dummy_test.avi"):
    """Generates a quick 3-second video to test the pipeline."""
    print(f"No input video found. Generating a dummy test video ({filename})...")
    
    # Swapped to MJPG and .avi, which is practically guaranteed to work on macOS
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(filename, fourcc, 30.0, (640, 480))
    
    if not out.isOpened():
        print("ERROR: OpenCV VideoWriter failed to open. Codec issue!")
        return False

    # Create 90 frames (3 seconds at 30fps)
    for i in range(90):
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 200 # Light gray background
        
        # Draw a moving "player" (rectangle)
        px1, py1 = 100 + i*2, 200
        px2, py2 = 150 + i*2, 350
        cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), -1)
        
        # Draw a moving "shuttlecock" (circle)
        cx, cy = 50 + i*5, int(240 + 100 * np.sin(i * 0.2))
        cv2.circle(frame, (cx, cy), 5, (0, 0, 0), -1)
        
        out.write(frame)
        
    out.release()
    print(f"Successfully created {filename}")
    return True

def main():
    video_path = "input.mp4"
    
    # Generate a dummy video if input.mp4 doesn't exist
    if not os.path.exists(video_path):
        video_path = "dummy_test.avi"
        if not create_dummy_video(video_path):
            return # Exit if video creation failed

    # Safety Check: Did OpenCV actually write the file?
    if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
        print(f"CRITICAL ERROR: {video_path} is missing or 0 bytes! VideoWriter failed silently.")
        return

    print("Loading YOLOv8 onto CPU...")
    yolo_model = YOLO("yolov8n.pt") 

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: OpenCV could not read {video_path}.")
        return

    print("Starting video processing loop. Press 'q' to exit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video reached.")
            break

        # 1. Test YOLOv8 (Player Detection)
        results = yolo_model.predict(frame, classes=[0], device='cpu', verbose=False)
        
        # Draw bounding boxes
        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, box.tolist())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Player", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 2. Test Display Window
        cv2.imshow("Environment Test Run", frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Test complete. Environment is working!")

if __name__ == "__main__":
    main()