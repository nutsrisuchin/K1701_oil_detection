import cv2
import time
import os
from collections import deque
from ultralytics import YOLO

# ==========================================
# 0. CONFIGURATION
# ==========================================
# STRICT DROP ZONE (Square): (x_min, x_max, y_min, y_max)
# Values represent the ratio (0.0 to 1.0) of the glass bounding box.
DROP_ZONE_CONFIG = {
    0: (0.4, 0.8, 0.5, 0.9),  
    1: (0.4, 0.8, 0.5, 0.9),
    2: (0.4, 0.8, 0.65, 0.9),
    3: (0.4, 0.8, 0.5, 0.9),
    4: (0.3, 0.7, 0.6, 0.9),  
    5: (0.3, 0.7, 0.56, 0.9),
    6: (0.4, 0.7, 0.55, 0.9) 
}

# ==========================================
# 1. THE PRESENCE TRACKER CLASS 
# ==========================================
class SightGlass:
    def __init__(self, glass_id, drop_zone_ratios):
        self.glass_id = glass_id
        self.total_drops = 0
        self.drop_timestamps = deque()
        
        # Unpack the specific drop zone for this glass
        self.x_min_ratio, self.x_max_ratio, self.y_min_ratio, self.y_max_ratio = drop_zone_ratios
        
        # State Machine Memory
        self.is_bubbling = False
        self.empty_frames = 0

    def update_and_count(self, main_frame, glass_box, global_bubble_boxes):
        gx1, gy1, gx2, gy2 = glass_box
        roi_width = gx2 - gx1
        roi_height = gy2 - gy1
        
        # 1. Calculate absolute pixel coordinates for the Strict Drop Zone
        dz_x1 = int(gx1 + (roi_width * self.x_min_ratio))
        dz_x2 = int(gx1 + (roi_width * self.x_max_ratio))
        dz_y1 = int(gy1 + (roi_height * self.y_min_ratio))
        dz_y2 = int(gy1 + (roi_height * self.y_max_ratio))

        # Draw the main glass boundary (Green)
        cv2.rectangle(main_frame, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
        # Draw the Strict Drop Zone (Purple)
        cv2.rectangle(main_frame, (dz_x1, dz_y1), (dz_x2, dz_y2), (255, 0, 255), 2)

        # ---------------------------------------------------------
        # DRAW CALIBRATION RULERS (0.1 to 0.9)
        # ---------------------------------------------------------
        for i in range(1, 10):
            ratio = i / 10.0
            tick_x = int(gx1 + (roi_width * ratio))
            cv2.line(main_frame, (tick_x, gy1), (tick_x, gy1 + 5), (255, 255, 255), 1)
            cv2.putText(main_frame, f"{ratio:.1f}", (tick_x - 8, gy1 + 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        for i in range(1, 10):
            ratio = i / 10.0
            tick_y = int(gy1 + (roi_height * ratio))
            cv2.line(main_frame, (gx1, tick_y), (gx1 + 5, tick_y), (255, 255, 255), 1)
            cv2.putText(main_frame, f"{ratio:.1f}", (gx1 + 8, tick_y + 3), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        # ---------------------------------------------------------
        
        current_time = time.time()

        # 2. Filter YOLO detections to ONLY include bubbles inside the Purple Drop Zone
        valid_bubbles = []
        for box in global_bubble_boxes:
            bx1, by1, bx2, by2 = map(int, box)
            
            # Calculate the exact center of the bubble
            cx = bx1 + ((bx2 - bx1) // 2)
            cy = by1 + ((by2 - by1) // 2)
            
            # Check if it falls inside the Purple Box
            if (dz_x1 <= cx <= dz_x2) and (dz_y1 <= cy <= dz_y2):
                valid_bubbles.append((bx1, by1, bx2, by2))
                cv2.rectangle(main_frame, (bx1, by1), (bx2, by2), (0, 255, 255), 1) # Yellow = Valid
            else:
                cv2.rectangle(main_frame, (bx1, by1), (bx2, by2), (0, 0, 100), 1) # Dark Red = Ignored

        # 3. Simple Presence Logic
        if len(valid_bubbles) > 0:
            self.empty_frames = 0 # Reset cooldown
            
            # If a bubble is here, and the latch was unlocked, count it!
            if not self.is_bubbling:
                self.total_drops += 1
                self.drop_timestamps.append(current_time)
                self.is_bubbling = True # Lock the counter
                
        else:
            # No valid bubbles detected. Start ticking up the cooldown timer.
            self.empty_frames += 1
            
            # If the zone has been empty for enough consecutive frames, unlock.
            # (Note: 5 frames at 30fps is roughly 0.16 seconds. Adjust this if needed).
            if self.is_bubbling and self.empty_frames > 5:
                self.is_bubbling = False 

        # --- METRICS OVERLAY ---
        while self.drop_timestamps and self.drop_timestamps[0] < current_time - 60:
            self.drop_timestamps.popleft()
        
        current_dpm = len(self.drop_timestamps)
        
        text_color = (0, 0, 255) if self.is_bubbling else (255, 255, 0)
        cv2.putText(main_frame, f"ID:{self.glass_id} DPM:{current_dpm} Tot:{self.total_drops}", 
                    (gx1, gy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)


# ==========================================
# 2. MAIN EXECUTION LOOP (FULL FRAME INFERENCE)
# ==========================================
def main():
    print("Loading YOLO models...")
    model_glass = YOLO('best_glass.pt')  
    model_bubble = YOLO('best_bubble.pt') 

    trackers = {}
    video_source = 'MVI_9973.MP4' 
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"Error: Could not open video source {video_source}")
        return

    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(video_source))[0]
    output_video_path = os.path.join(output_dir, f"{video_name}_final.mp4")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print("Starting visual pipeline. Press 'q' to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # ---------------------------------------------------------
        # STAGE 1: DETECT EVERYTHING ON THE FULL FRAME
        # ---------------------------------------------------------
        # Detect Glasses
        glass_results = model_glass.predict(source=frame, conf=0.5, verbose=False)
        glass_boxes = []
        for result in glass_results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                glass_boxes.append((x1, y1, x2, y2))
        
        glass_boxes.sort(key=lambda b: b[0]) # Sort left to right

        # Detect Bubbles (Lowered to 0.15 just in case it is struggling)
        bubble_results = model_bubble.predict(source=frame, conf=0.15, verbose=False)
        all_bubble_boxes = []
        for result in bubble_results:
            for box in result.boxes:
                bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                all_bubble_boxes.append((bx1, by1, bx2, by2))

        # ---------------------------------------------------------
        # STAGE 2: MATHEMATICAL FILTERING (THE HANDOFF)
        # ---------------------------------------------------------
        for i, glass_box in enumerate(glass_boxes):
            gx1, gy1, gx2, gy2 = glass_box
            
            if i not in trackers:
                zone_config = DROP_ZONE_CONFIG.get(i, (0.1, 0.9, 0.1, 0.9))
                trackers[i] = SightGlass(glass_id=i, drop_zone_ratios=zone_config)
            
            # Filter: Which bubbles are physically inside the Green Box of THIS specific glass?
            bubbles_in_this_glass = []
            for (bx1, by1, bx2, by2) in all_bubble_boxes:
                # Find the center of the bubble
                cx = bx1 + ((bx2 - bx1) // 2)
                cy = by1 + ((by2 - by1) // 2)
                
                # If the center of the bubble is inside the green glass box, pass it to the tracker
                if (gx1 <= cx <= gx2) and (gy1 <= cy <= gy2):
                    bubbles_in_this_glass.append((bx1, by1, bx2, by2))

            # Pass the filtered list to the tracker's logic
            trackers[i].update_and_count(frame, glass_box, bubbles_in_this_glass)

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