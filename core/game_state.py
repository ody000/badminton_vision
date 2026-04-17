"""Game state management."""
class GameStateManager:
    def __init__(self, fps):
        self.fps = fps
        
        # State variables
        self.is_rally_active = False
        self.shot_count = 0
        self.rally_start_frame = 0
        self.rally_end_frame = 0
        
        # To detect rally start: wait for 2 consecutive hits
        self.consecutive_hits = 0
        self.last_hit_frame = 0

    def update(self, current_frame_num, is_hit_this_frame, shuttle_coord):
        """
        Updates game state based on frame events.
        Returns a dictionary of stats for visualization.
        """
        
        # --- Logic to START a Rally ---
        # A simple heuristic: A rally starts if there are two hits within a short time window (service + return)
        if not self.is_rally_active and is_hit_this_frame:
            if (current_frame_num - self.last_hit_frame) < (self.fps * 2): # e.g., within 2 seconds
                self.consecutive_hits += 1
            else:
                self.consecutive_hits = 1 # Reset count if too much time passed
                
            self.last_hit_frame = current_frame_num
            
            if self.consecutive_hits >= 2:
                self.rally_start()
                # Retroactively count the service hit
                self.shot_count = 2 


        # --- Logic DURING a Rally ---
        if self.is_rally_active:
            if is_hit_this_frame:
                self.shot_count += 1
                self.last_hit_frame = current_frame_num
            
            # --- Logic to END a Rally ---
            # Heuristic 1: No hits for a long time (shuttle likely on the floor)
            time_since_last_hit = (current_frame_num - self.last_hit_frame) / self.fps
            if time_since_last_hit > 3.0: # No hit for 3 seconds
                self.rally_end(current_frame_num)
            
            # Heuristic 2: Shuttle is lost (TrackNet returns None) for many consecutive frames
            # (This logic would require tracking missed frames, simplified here)
            if shuttle_coord is None and self.shot_count > 0:
                 # In a real system, you'd have a counter for missed frames
                 pass

        return self.get_stats()

    def rally_start(self):
        print(f"--- RALLY STARTED ---")
        self.is_rally_active = True
        self.shot_count = 0
        self.rally_start_frame = self.last_hit_frame

    def rally_end(self, end_frame):
        duration = (end_frame - self.rally_start_frame) / self.fps
        print(f"--- RALLY ENDED --- Duration: {duration:.2f}s | Shots: {self.shot_count}")
        self.is_rally_active = False
        self.consecutive_hits = 0 # Reset for next rally

    def get_stats(self):
        return {
            'is_active': self.is_rally_active,
            'shot_count': self.shot_count
        }