"""Video input/output utilities."""
import cv2
import os
from collections import deque

class VideoIOHandler:
    def __init__(self, input_path, output_path, buffer_size=30):
        # --- 1. Setup Reader ---
        abs_input = os.path.abspath(input_path)
        if not os.path.exists(abs_input):
            raise FileNotFoundError(f"Cannot find input video at {abs_input}")
            
        self.cap = cv2.VideoCapture(abs_input)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Loaded video: {self.width}x{self.height} at {self.fps} FPS ({self.total_frames} frames)")

        # --- 2. Setup Buffer ---
        self.frame_buffer = deque(maxlen=buffer_size)
        self.frame_count = 0

        # --- 3. Setup Writer (The Brute-Force Fallback) ---
        abs_output = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(abs_output), exist_ok=True)
        
        # Get the filename without the extension so we can test different ones
        base_path, _ = os.path.splitext(abs_output)
        
        # Priority list of Mac codecs: 
        # 1. H.264 (avc1) in MP4 
        # 2. Standard MP4 (mp4v)
        # 3. Apple QuickTime (mp4v in MOV container)
        # 4. Motion JPEG (MJPG in AVI container)
        codecs_to_try = [
            ('avc1', '.mp4'),
            ('mp4v', '.mp4'),
            ('mp4v', '.mov'),
            ('MJPG', '.avi')
        ]
        
        self.writer = None
        for codec_name, ext in codecs_to_try:
            test_path = base_path + ext
            fourcc = cv2.VideoWriter_fourcc(*codec_name)
            
            # Attempt to open the writer with the current codec
            temp_writer = cv2.VideoWriter(test_path, fourcc, self.fps, (self.width, self.height))
            
            if temp_writer.isOpened():
                self.writer = temp_writer
                print(f"[VideoIO] Success! Using codec '{codec_name}' -> saving to {test_path}")
                break
            else:
                print(f"[VideoIO] Mac rejected codec '{codec_name}'. Trying next...")
                
        if self.writer is None or not self.writer.isOpened():
            raise Exception("CRITICAL: Your Mac rejected all standard video codecs. Check OpenCV installation.")

    def stream(self):
        """Yields the current frame and frame number while maintaining the buffer."""
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
                
            self.frame_count += 1
            self.frame_buffer.append(frame)
            
            yield frame, self.frame_count

    def get_recent_frames(self, count):
        """Returns the last 'count' frames for sequence models."""
        if len(self.frame_buffer) < count:
            return None 
        return list(self.frame_buffer)[-count:]

    def write_frame(self, processed_frame):
        """Writes the annotated frame to the output video file."""
        self.writer.write(processed_frame)

    def release(self):
        """Cleans up the memory and finalizes the output file."""
        self.cap.release()
        if self.writer:
            self.writer.release()
        print("Video reading and writing closed successfully.")