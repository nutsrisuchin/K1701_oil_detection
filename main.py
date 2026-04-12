import cv2
import time
import os
from collections import deque
from ultralytics import YOLO

# ==========================================
# 0. CONFIGURATION DICTIONARIES
# ==========================================

# USER APPLIED TRIPWIRE CONFIG:
TRIPWIRE_CONFIG = {
    0: 0.5,  # Far left glass
    1: 0.55,
    2: 0.65,
    3: 0.55,  # Center glass (might need a higher line due to angle)
    4: 0.6,
    5: 0.56,
    6: 0.55 
}

# STRICT DROP ZONE (Square): (x_min, x_max, y_min, y_max)
# Remember: y_min (Top edge) must be smaller than y_max (Bottom edge)
DROP_ZONE_CONFIG = {
    0: (0.4, 0.8, 0.5, 0.9),  
    1: (0.4, 0.8, 0.5, 0.9),
    2: (0.4, 0.8, 0.65, 0.9),
    3: (0.4, 0.8, 0.5, 0.9),
    4: (0.3, 0.7, 0.6, 0.9), # ID 4: Custom Air Gap fix to avoid the standing oil illusion
    5: (0.3, 0.7, 0.56, 0.9),
    6: (0.4, 0.7, 0.55, 0.9) 
}

# ==========================================
# 1. INDEPENDENT TRACKER CLASS
# ==========================================
class SightGlass:
    def __init__(self, glass_id, tripwire_ratio=0.6, x_min_ratio=0.1, x_max_ratio=0.9, y_min_ratio=0.1, y_max_ratio=0.9):
        self.glass_id = glass_id
        self.tripwire_ratio = tripwire_ratio
        
        self.x_min_ratio = x_min_ratio
        self.x_max_ratio = x_max_ratio
        self.y_min_ratio = y_min_ratio
        self.y_max_ratio = y_max_ratio
        
        # Independent MOG2 subtractor for this specific glass
        self.back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
        self.total_drops = 0
        self.drop_timestamps = deque()
        
        # State Machine Latch to prevent double-counting slow drops
        self.is_drop_crossing = False 

    def update_and_count(self, main_frame, x1, y1, x2, y2):
        # Crop the frame to the specific bounding box
        roi = main_frame[y1:y2, x1:x2]
        if roi.size == 0: 
            return

        roi_height, roi_width = roi.shape[:2]
        
        # Calculate pixel coordinates for all boundaries
        tripwire_y = int(roi_height * self.tripwire_ratio) 
        x_min_px = int(roi_width * self.x_min_ratio)
        x_max_px = int(roi_width * self.x_max_ratio)
        y_min_px = int(roi_height * self.y_min_ratio)
        y_max_px = int(roi_height * self.y_max_ratio)

        # 1. Draw the ROI box (Green) and Tripwire (Blue)
        cv2.rectangle(main_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.line(main_frame, (x1, y1 + tripwire_y), (x2, y1 + tripwire_y), (255, 0, 0), 2)

        # 2. Draw the Strict Drop Zone Square (Yellow, thick line)
        cv2.rectangle(main_frame, (x1 + x_min_px, y1 + y_min_px), (x1 + x_max_px, y1 + y_max_px), (0, 255, 255), 2)

        # ---------------------------------------------------------
        # 3. VERTICAL RULER (Y-Axis Calibration)
        # ---------------------------------------------------------
        for i in range(1, 10):
            ratio = i / 10.0
            tick_y = int(y1 + (roi_height * ratio))
            cv2.line(main_frame, (x2 - 5, tick_y), (x2, tick_y), (255, 255, 255), 1)
            cv2.putText(main_frame, f"{ratio:.1f}", (x2 - 24, tick_y + 3), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        # ---------------------------------------------------------
        # 4. HORIZONTAL RULER (X-Axis Calibration)
        # ---------------------------------------------------------
        for i in range(1, 10):
            ratio = i / 10.0
            tick_x = int(x1 + (roi_width * ratio))
            cv2.line(main_frame, (tick_x, y2 - 5), (tick_x, y2), (255, 255, 255), 1)
            cv2.putText(main_frame, f"{ratio:.1f}", (tick_x - 8, y2 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        # Apply background subtraction
        fg_mask = self.back_sub.apply(roi)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Find moving blobs
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        current_time = time.time()
        
        # Track if any valid drops are currently inside the Strict Drop Zone square
        motion_in_zone_this_frame = False

        for contour in contours:
            if cv2.contourArea(contour) > 15:
                cx, cy, cw, ch = cv2.boundingRect(contour)
                centroid_y = cy + (ch // 2)
                centroid_x = cx + (cw // 2)

                # Draw the red dot for all detected motion
                cv2.circle(main_frame, (x1 + centroid_x, y1 + centroid_y), 4, (0, 0, 255), -1)

                # TRIGGER: Centroid must be INSIDE the yellow square
                if (centroid_y > tripwire_y and
                    y_min_px < centroid_y < y_max_px and
                    x_min_px < centroid_x < x_max_px):
                    
                    motion_in_zone_this_frame = True

        # --- STATE MACHINE COUNTING LOGIC ---
        if motion_in_zone_this_frame and not self.is_drop_crossing:
            self.total_drops += 1
            self.drop_timestamps.append(current_time)
            self.is_drop_crossing = True  # Lock the counter
            
        # If the square is completely empty, unlock the counter for the next drop
        elif not motion_in_zone_this_frame:
            self.is_drop_crossing = False

        # Calculate live DPM
        while self.drop_timestamps and self.drop_timestamps[0] < current_time - 60:
            self.drop_timestamps.popleft()
        
        current_dpm = len(self.drop_timestamps)

        # Overlay metrics
        cv2.putText(main_frame, f"ID:{self.glass_id} DPM:{current_dpm} Tot:{self.total_drops}", 
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

# ==========================================
# 2. MAIN EXECUTION LOOP
# ==========================================
def main():
    print("Loading YOLO model...")
    model = YOLO('best_glass.pt') 

    trackers = {}
    video_source = 'MVI_9973.MP4' 
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"Error: Could not open video source {video_source}")
        return

    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    video_name = os.path.splitext(os.path.basename(video_source))[0]
    output_video_path = os.path.join(output_dir, f"{video_name}_test.mp4")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print("Starting visual pipeline. Press 'q' to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("End of video stream.")
            break

        # Stage 1: Detect glasses
        results = model.predict(source=frame, conf=0.5, verbose=False)
        
        current_boxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                current_boxes.append((x1, y1, x2, y2))

        # Sort left to right
        current_boxes.sort(key=lambda b: b[0])

        # Stage 2: Hand off to the mathematical trackers
        for i, (x1, y1, x2, y2) in enumerate(current_boxes):
            if i not in trackers:
                t_ratio = TRIPWIRE_CONFIG.get(i, 0.6) 
                
                # Unpack the 4-ratio configuration
                x_min, x_max, y_min, y_max = DROP_ZONE_CONFIG.get(i, (0.1, 0.9, 0.1, 0.9))
                trackers[i] = SightGlass(glass_id=i, 
                                        tripwire_ratio=t_ratio, 
                                        x_min_ratio=x_min, 
                                        x_max_ratio=x_max,
                                        y_min_ratio=y_min,
                                        y_max_ratio=y_max)
            
            trackers[i].update_and_count(frame, x1, y1, x2, y2)

        out.write(frame)
        cv2.imshow("Automated Lubrication Monitor", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video saved successfully to: {output_video_path}")

if __name__ == "__main__":
    main()