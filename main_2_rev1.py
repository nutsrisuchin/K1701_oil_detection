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

# TRIPWIRE: y-ratio of the glass bounding box.
# A drop is counted when a tracked bubble crosses this line top-to-bottom.
TRIPWIRE_CONFIG = {
    0: 0.5,   # Far left glass
    1: 0.55,
    2: 0.65,
    3: 0.55,  # Center glass (slightly below zone top due to angle)
    4: 0.6,
    5: 0.6,
    6: 0.55
}

# ==========================================
# 1. THE SIGHT GLASS TRACKER CLASS
# ==========================================
class SightGlass:
    def __init__(self, glass_id, drop_zone_ratios, tripwire_ratio):
        self.glass_id = glass_id
        self.total_drops = 0
        self.drop_timestamps = deque()

        self.x_min_ratio, self.x_max_ratio, self.y_min_ratio, self.y_max_ratio = drop_zone_ratios
        self.tripwire_ratio = tripwire_ratio

        # Tripwire tracking state
        self.prev_bubble_positions = {}  # {track_id: (cx, cy)} from previous frame
        self.counted_ids = set()         # track IDs already counted (won't be re-counted)

    def update_and_count(self, main_frame, glass_box, all_bubbles_in_glass):
        """
        all_bubbles_in_glass: list of (bx1, by1, bx2, by2, track_id)
            All tracked bubbles physically inside the green glass bounding box.
            Includes bubbles above the tripwire so crossing can be detected.
        """
        gx1, gy1, gx2, gy2 = glass_box
        roi_width = gx2 - gx1
        roi_height = gy2 - gy1

        # Absolute pixel coordinates for the Drop Zone
        dz_x1 = int(gx1 + (roi_width * self.x_min_ratio))
        dz_x2 = int(gx1 + (roi_width * self.x_max_ratio))
        dz_y1 = int(gy1 + (roi_height * self.y_min_ratio))
        dz_y2 = int(gy1 + (roi_height * self.y_max_ratio))

        # Absolute pixel y-coordinate for the Tripwire
        tripwire_y = int(gy1 + (roi_height * self.tripwire_ratio))

        # --- DRAW OVERLAYS ---
        # Green: glass boundary
        cv2.rectangle(main_frame, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
        # Purple: drop zone
        cv2.rectangle(main_frame, (dz_x1, dz_y1), (dz_x2, dz_y2), (255, 0, 255), 2)
        # Orange: tripwire
        cv2.line(main_frame, (dz_x1, tripwire_y), (dz_x2, tripwire_y), (0, 165, 255), 2)
        cv2.putText(main_frame, "WIRE", (dz_x2 + 4, tripwire_y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 165, 255), 1)

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

        # Build center-point position dict for ALL bubbles in this glass
        current_positions = {}
        for (bx1, by1, bx2, by2, track_id) in all_bubbles_in_glass:
            cx = bx1 + ((bx2 - bx1) // 2)
            cy = by1 + ((by2 - by1) // 2)
            current_positions[track_id] = (cx, cy)

        # Draw bubbles: yellow if inside drop zone, dark red if outside
        for (bx1, by1, bx2, by2, track_id) in all_bubbles_in_glass:
            cx, cy = current_positions[track_id]
            if (dz_x1 <= cx <= dz_x2) and (dz_y1 <= cy <= dz_y2):
                cv2.rectangle(main_frame, (bx1, by1), (bx2, by2), (0, 255, 255), 1)  # Yellow = in zone
                cv2.putText(main_frame, f"{track_id}", (bx1, by1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
            else:
                cv2.rectangle(main_frame, (bx1, by1), (bx2, by2), (0, 0, 100), 1)   # Dark Red = ignored

        # --- TRIPWIRE CROSSING DETECTION ---
        for track_id, (cx, cy) in current_positions.items():
            if track_id in self.counted_ids:
                continue  # Already counted this bubble

            if track_id in self.prev_bubble_positions:
                # Seen before: require an explicit top-to-bottom crossing
                _, prev_cy = self.prev_bubble_positions[track_id]
                crossed = prev_cy < tripwire_y and cy >= tripwire_y and (dz_x1 <= cx <= dz_x2)
            else:
                # First detection of this ID: the drop fell fast enough that it was
                # already below the tripwire when the tracker first picked it up.
                # Count it if it landed inside the drop zone.
                crossed = cy >= tripwire_y and (dz_x1 <= cx <= dz_x2) and (dz_y1 <= cy <= dz_y2)

            if crossed:
                self.total_drops += 1
                self.drop_timestamps.append(current_time)
                self.counted_ids.add(track_id)
                cv2.circle(main_frame, (cx, tripwire_y), 10, (0, 165, 255), 3)

        # Advance position history to the current frame
        self.prev_bubble_positions = current_positions

        # --- METRICS OVERLAY ---
        while self.drop_timestamps and self.drop_timestamps[0] < current_time - 60:
            self.drop_timestamps.popleft()

        current_dpm = len(self.drop_timestamps)

        any_in_zone = any(
            (dz_x1 <= cx <= dz_x2) and (dz_y1 <= cy <= dz_y2)
            for (cx, cy) in current_positions.values()
        )
        text_color = (0, 0, 255) if any_in_zone else (255, 255, 0)
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
        # Glasses: plain predict (static objects, no tracking needed)
        glass_results = model_glass.predict(source=frame, conf=0.5, verbose=False)
        glass_boxes = []
        for result in glass_results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                glass_boxes.append((x1, y1, x2, y2))

        glass_boxes.sort(key=lambda b: b[0])  # Sort left to right

        # Bubbles: track() assigns persistent IDs across frames (ByteTrack)
        bubble_results = model_bubble.track(source=frame, conf=0.15, verbose=False, persist=True)
        all_bubble_boxes = []
        for result in bubble_results:
            for box in result.boxes:
                if box.id is None:
                    continue  # Tracker couldn't assign an ID this frame — skip
                bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                track_id = int(box.id[0])
                all_bubble_boxes.append((bx1, by1, bx2, by2, track_id))

        # ---------------------------------------------------------
        # STAGE 2: ASSIGN BUBBLES TO EACH GLASS AND UPDATE TRACKERS
        # ---------------------------------------------------------
        for i, glass_box in enumerate(glass_boxes):
            gx1, gy1, gx2, gy2 = glass_box

            if i not in trackers:
                zone_config = DROP_ZONE_CONFIG.get(i, (0.1, 0.9, 0.1, 0.9))
                wire_ratio = TRIPWIRE_CONFIG.get(i, 0.55)
                trackers[i] = SightGlass(glass_id=i, drop_zone_ratios=zone_config,
                                         tripwire_ratio=wire_ratio)

            # Pass ALL bubbles in this glass (including above tripwire) so
            # crossing detection has the prior-frame "above" position available.
            bubbles_in_this_glass = []
            for (bx1, by1, bx2, by2, track_id) in all_bubble_boxes:
                cx = bx1 + ((bx2 - bx1) // 2)
                cy = by1 + ((by2 - by1) // 2)
                if (gx1 <= cx <= gx2) and (gy1 <= cy <= gy2):
                    bubbles_in_this_glass.append((bx1, by1, bx2, by2, track_id))

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
