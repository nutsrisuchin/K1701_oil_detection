import cv2
import time
import os
from collections import deque
from ultralytics import YOLO

# ==========================================
# 0. CONFIGURATION DICTIONARIES
# ==========================================

# TRIPWIRE HEIGHT: The vertical blue line.
TRIPWIRE_CONFIG = {
    0: 0.5,  # Far left glass
    1: 0.55,
    2: 0.65,
    3: 0.55, # Center glass (might need a higher line due to angle)
    4: 0.6,
    5: 0.56,
    6: 0.55 
}

# STRICT DROP ZONE (Square): (x_min, x_max, y_min, y_max)
DROP_ZONE_CONFIG = {
    0: (0.4, 0.8, 0.5, 0.9),  
    1: (0.4, 0.8, 0.5, 0.9),
    2: (0.4, 0.8, 0.65, 0.9),
    3: (0.4, 0.8, 0.5, 0.9),
    4: (0.3, 0.7, 0.6, 0.9),  # ID 4: Custom Air Gap fix to avoid the standing oil illusion
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
        
        self.back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
        self.total_drops = 0
        self.drop_timestamps = deque()
        
        # Simplified Cooldown Logic
        self.last_drop_time = 0
        self.cooldown_time = 0.5  # Seconds to wait before counting a new drop

    def update_and_count(self, main_frame, x1, y1, x2, y2):
        roi = main_frame[y1:y2, x1:x2]
        if roi.size == 0: 
            return

        roi_height, roi_width = roi.shape[:2]
        
        tripwire_y = int(roi_height * self.tripwire_ratio) 
        x_min_px = int(roi_width * self.x_min_ratio)
        x_max_px = int(roi_width * self.x_max_ratio)
        y_min_px = int(roi_height * self.y_min_ratio)
        y_max_px = int(roi_height * self.y_max_ratio)

        # Draw ROI (Green), Tripwire (Blue), and Drop Zone (Yellow)
        cv2.rectangle(main_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.line(main_frame, (x1, y1 + tripwire_y), (x2, y1 + tripwire_y), (255, 0, 0), 2)
        cv2.rectangle(main_frame, (x1 + x_min_px, y1 + y_min_px), (x1 + x_max_px, y1 + y_max_px), (0, 255, 255), 2)

        # Vertical Ruler
        for i in range(1, 10):
            ratio = i / 10.0
            tick_y = int(y1 + (roi_height * ratio))
            cv2.line(main_frame, (x2 - 5, tick_y), (x2, tick_y), (255, 255, 255), 1)
            cv2.putText(main_frame, f"{ratio:.1f}", (x2 - 24, tick_y + 3), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        # Horizontal Ruler
        for i in range(1, 10):
            ratio = i / 10.0
            tick_x = int(x1 + (roi_width * ratio))
            cv2.line(main_frame, (tick_x, y2 - 5), (tick_x, y2), (255, 255, 255), 1)
            cv2.putText(main_frame, f"{ratio:.1f}", (tick_x - 8, y2 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        # Background Subtraction
        fg_mask = self.back_sub.apply(roi)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        current_time = time.time()

        for contour in contours:
            if cv2.contourArea(contour) > 15:
                cx, cy, cw, ch = cv2.boundingRect(contour)
                centroid_y = cy + (ch // 2)
                centroid_x = cx + (cw // 2)

                cv2.circle(main_frame, (x1 + centroid_x, y1 + centroid_y), 4, (0, 0, 255), -1)

                # TRIGGER: Inside Yellow Box AND Below Blue Line AND Cooldown Passed
                if (centroid_y > tripwire_y and
                    y_min_px < centroid_y < y_max_px and
                    x_min_px < centroid_x < x_max_px):
                    
                    if (current_time - self.last_drop_time) > self.cooldown_time:
                        self.total_drops += 1
                        self.drop_timestamps.append(current_time)
                        self.last_drop_time = current_time

        # Calculate live DPM
        while self.drop_timestamps and self.drop_timestamps[0] < current_time - 60:
            self.drop_timestamps.popleft()
        
        current_dpm = len(self.drop_timestamps)

        cv2.putText(main_frame, f"ID:{self.glass_id} DPM:{current_dpm} Tot:{self.total_drops}", 
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

# ==========================================
# 2. MAIN EXECUTION LOOP
# ==========================================
def main():
    # Model name updated to best_glass.pt
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

        results = model.predict(source=frame, conf=0.5, verbose=False)
        
        current_boxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                current_boxes.append((x1, y1, x2, y2))

        current_boxes.sort(key=lambda b: b[0])

        for i, (x1, y1, x2, y2) in enumerate(current_boxes):
            if i not in trackers:
                t_ratio = TRIPWIRE_CONFIG.get(i, 0.6) 
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