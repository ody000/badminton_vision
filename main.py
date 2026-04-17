import cv2
import csv
import os
import numpy as np
from utils.video_io import VideoIOHandler
from models.player_yolo import PlayerDetector
from models.shuttle_tracknet import ShuttleTracker
from core.kinematics import KinematicsEngine
from core.homography import CourtMapper
from utils.minimap import Minimap  # <--- NEW IMPORT

def main():
    input_video = "data/input/match_clip.mp4"
    output_video = "data/output/tracked_match.mp4"
    csv_file = "data/output/match_data.csv"

    # Initialize Modules
    video = VideoIOHandler(input_video, output_video)
    yolo = PlayerDetector(weights_path="yolov8n.pt")
    tracknet = ShuttleTracker(weights_path="weights/tracknet_weights.pt")
    court_mapper = CourtMapper()
    kinematics = KinematicsEngine(history_size=5, proximity_threshold=200)
    minimap_viz = Minimap()

    # --- CALIBRATION (Replace with YOUR numbers!) ---
    src_corners = [[239, 371],  # Top-Left (Far left corner of court)
        [642, 371],  # Top-Right (Far right corner of court)
        [837, 770],  # Bottom-Right (Near right corner)
        [54, 770]]
    court_mapper.calibrate(src_corners)
    
    # Open CSV for writing
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Frame", "Shuttle_X_cm", "Shuttle_Y_cm", "Event", "Striking_Player"])

        print("Starting Analysis...")
        
        for frame, frame_num in video.stream():
            # 1. Perception
            player_boxes = yolo.detect(frame)
            pixel_shuttle = tracknet.update(frame)

            # 2. Homography
            real_shuttle = court_mapper.transform_point(pixel_shuttle)
            real_feet = []
            for box in player_boxes:
                feet_pix = court_mapper.get_player_feet(box)
                real_feet.append(court_mapper.transform_point(feet_pix))

            # 3. Physics / Kinematics
            is_hit, striker_idx = kinematics.update_and_check_hit(real_shuttle, real_feet)
        
            # 4. Recording Data
            event_str = "Hit" if is_hit else ""
            striker_str = f"Player {striker_idx}" if striker_idx is not None else ""
            
            # --- FIX IS HERE: Explicitly check "is not None" ---
            if real_shuttle is not None:
                writer.writerow([frame_num, int(real_shuttle[0]), int(real_shuttle[1]), event_str, striker_str])
            else:
                writer.writerow([frame_num, "", "", "", ""])

            # 5. Visualization (The Minimap!)
            frame = minimap_viz.draw(frame, real_shuttle, real_feet)
            
            # Draw Hit Text
            if is_hit:
                cv2.putText(frame, "SMASH!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            video.write_frame(frame)
            if frame_num % 30 == 0: print(f"Processed {frame_num} frames...")

    print(f"Analysis Complete. Data saved to {csv_file}")

if __name__ == "__main__":
    main()

# import cv2
# import os
# import numpy as np
# from utils.video_io import VideoIOHandler
# from models.player_yolo import PlayerDetector
# from models.shuttle_tracknet import ShuttleTracker
# from core.kinematics import KinematicsEngine
# from core.homography import CourtMapper # <--- IMPORTED
# from utils.visualization import Visualizer

# def main():
#     input_video = "data/input/match_clip.mp4"
#     output_video = "data/output/tracked_match.mp4"

#     if not os.path.exists(input_video):
#         print("Please place a video named 'match_clip.mp4' in the data/input/ folder!")
#         return

#     print("Initializing modules...")
#     video = VideoIOHandler(input_video, output_video, buffer_size=30)
#     yolo = PlayerDetector(weights_path="yolov8n.pt")
#     tracknet = ShuttleTracker(weights_path="weights/tracknet_weights.pt")
#     court_mapper = CourtMapper()
    
#     # --- IMPORTANT: CALIBRATION STEP ---
#     # You must find the pixel coordinates of the 4 court corners in your specific video.
#     # Order: [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
#     # Use a paint tool or a temp script to hover over the video frame and find these X,Y values.
#     # (These are dummy values; REPLACE THEM with yours!)
#     src_corners = [
#         [239, 371],  # Top-Left (Far left corner of court)
#         [642, 371],  # Top-Right (Far right corner of court)
#         [837, 770],  # Bottom-Right (Near right corner)
#         [54, 770]   # Bottom-Left (Near left corner)
#     ]
#     court_mapper.calibrate(src_corners)
    
#     # Initialize Physics (Note: Threshold is now in CM, not pixels! 200cm = 2 meters)
#     kinematics = KinematicsEngine(history_size=5, proximity_threshold=200)
    
#     viz = Visualizer()
#     hit_count = 0
    
#     print("Processing video...")
    
#     for frame, frame_num in video.stream():
#         # 1. PERCEPTION
#         player_boxes = yolo.detect(frame)
#         pixel_shuttle = tracknet.update(frame)

#         # 2. HOMOGRAPHY (Pixels -> Real World CM)
#         real_shuttle = court_mapper.transform_point(pixel_shuttle)
        
#         real_player_feet = []
#         for box in player_boxes:
#             pixel_feet = court_mapper.get_player_feet(box)
#             real_feet = court_mapper.transform_point(pixel_feet)
#             # Store as tuple: (box, real_feet_coord)
#             real_player_feet.append((real_feet))

#         # 3. KINEMATICS (Using Real World Data)
#         # We need to extract just the boxes for proximity checks, but conceptually 
#         # checking distance in real-world units is better. 
#         # For now, we pass real_shuttle to track trajectory, but we need to check 
#         # proximity against real_players.
        
#         # Update Kinematics with REAL coordinates
#         # Note: We need to adapt Kinematics to handle the new format if we want perfect accuracy,
#         # but passing the real_shuttle point is the critical part for velocity checking.
        
#         # To make it compatible with your current kinematics.py _find_nearest_player:
#         # We'll pass the pixel boxes for now (to keep it simple), 
#         # but ideally, you'd rewrite _find_nearest_player to use real_feet distance.
        
#         is_hit, striking_player_idx = kinematics.update_and_check_hit(real_shuttle, real_player_feet)
        
#         if is_hit and striking_player_idx is not None:
#              print(f"🏸 HIT by Player {striking_player_idx}! Frame {frame_num}")
        
#         # 4. VISUALIZATION
#         # Draw detecting boxes on the camera frame
#         frame = viz.draw_frame(frame, player_boxes, pixel_shuttle)
        
#         # OPTIONAL: Draw a mini-map in the corner?
#         # We can implement this in visualizer later.
        
#         video.write_frame(frame)
#         if frame_num % 10 == 0: print(f"Processed {frame_num}...")

#     video.release()
#     print("Done!")

# if __name__ == "__main__":
#     main()