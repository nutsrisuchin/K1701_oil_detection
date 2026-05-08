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

        self.prev_zone_count = 0
        self.prev_ids_in_zone = set()  # IDs in zone last frame

    def update_and_count(self, main_frame, glass_box, all_bubbles_in_glass):
        """
        all_bubbles_in_glass: list of (bx1, by1, bx2, by2, track_id)
        """
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

        # 2. Filter bubbles inside the Purple Drop Zone
        current_zone_count = 0
        current_ids_in_zone = set()
        for (bx1, by1, bx2, by2, track_id) in all_bubbles_in_glass:
            cx = bx1 + ((bx2 - bx1) // 2)
            cy = by1 + ((by2 - by1) // 2)

            if (dz_x1 <= cx <= dz_x2) and (dz_y1 <= cy <= dz_y2):
                current_zone_count += 1
                current_ids_in_zone.add(track_id)
                cv2.rectangle(main_frame, (bx1, by1), (bx2, by2), (0, 255, 255), 1)  # Yellow = in zone
                cv2.putText(main_frame, f"{track_id}", (bx1, by1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
            else:
                cv2.rectangle(main_frame, (bx1, by1), (bx2, by2), (0, 0, 100), 1)  # Dark Red = ignored

        # 3. Combined count-delta + ID-change detection
        if current_zone_count > 0:
            # Signal 1: box count increased → new drops arrived
            new_drops_by_count = max(0, current_zone_count - self.prev_zone_count)
            # Signal 2: IDs changed while count stayed same → drop swapped in same frame
            new_ids = current_ids_in_zone - self.prev_ids_in_zone
            new_drops_by_id = len(new_ids) if current_zone_count == self.prev_zone_count else 0

            new_drops = max(new_drops_by_count, new_drops_by_id)
            if new_drops > 0:
                self.total_drops += new_drops
                for _ in range(new_drops):
                    self.drop_timestamps.append(current_time)

        self.prev_zone_count = current_zone_count
        self.prev_ids_in_zone = current_ids_in_zone

        # --- METRICS OVERLAY ---
        while self.drop_timestamps and self.drop_timestamps[0] < current_time - 60:
            self.drop_timestamps.popleft()

        current_dpm = len(self.drop_timestamps)

        text_color = (0, 0, 255) if current_zone_count > 0 else (255, 255, 0)
        cv2.putText(main_frame, f"ID:{self.glass_id} DPM:{current_dpm} Tot:{self.total_drops}",
                    (gx1, gy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)


# ==========================================
# 2. MAIN EXECUTION LOOP (FULL FRAME INFERENCE)
# ==========================================
def main():
    # Paths are configurable via environment variables so this runs in a container.
    # Mount your models, video, and output directory via Docker volumes.
    model_dir = os.environ.get('MODEL_DIR', '/app/models')
    video_source = os.environ.get('VIDEO_SOURCE', '/data/input/video.mp4')
    final_output_dir = os.environ.get('OUTPUT_DIR', '/data/output')
    # Set HEADLESS=true when running without a display (default in containers).
    headless = os.environ.get('HEADLESS', 'true').lower() == 'true'

    print("Loading YOLO models...")
    model_glass = YOLO(f'{model_dir}/best_glass.pt')
    model_bubble = YOLO(f'{model_dir}/best_bubble.pt')

    trackers = {}
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"Error: Could not open video source {video_source}")
        return

    os.makedirs(final_output_dir, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(video_source))[0]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{video_name}_final_{timestamp}.mp4"

    # Write to /tmp first (avoids issues with spaces/special chars in path)
    tmp_output_path = os.path.join('/tmp', filename)
    final_output_path = os.path.join(final_output_dir, filename)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    for codec in ['avc1', 'mp4v', 'XVID']:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(tmp_output_path, fourcc, fps, (width, height))
        if out.isOpened():
            print(f"Using codec: {codec}")
            break
        print(f"Codec {codec} failed, trying next...")
    else:
        print("Error: No working video codec found.")
        return

    print(f"Starting pipeline. Headless={headless}. Press 'q' to quit (display mode only).")

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

        # Bubbles: track() assigns persistent IDs across frames (ByteTrack)
        bubble_results = model_bubble.track(source=frame, conf=0.15, verbose=False, persist=True)
        all_bubble_boxes = []
        for result in bubble_results:
            for box in result.boxes:
                if box.id is None:
                    continue
                bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                track_id = int(box.id[0])
                all_bubble_boxes.append((bx1, by1, bx2, by2, track_id))

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
            for (bx1, by1, bx2, by2, track_id) in all_bubble_boxes:
                cx = bx1 + ((bx2 - bx1) // 2)
                cy = by1 + ((by2 - by1) // 2)
                if (gx1 <= cx <= gx2) and (gy1 <= cy <= gy2):
                    bubbles_in_this_glass.append((bx1, by1, bx2, by2, track_id))

            # Pass the filtered list to the tracker's logic
            trackers[i].update_and_count(frame, glass_box, bubbles_in_this_glass)

        out.write(frame)

        if not headless:
            cv2.imshow("Automated Lubrication Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    if not headless:
        cv2.destroyAllWindows()

    import shutil
    shutil.move(tmp_output_path, final_output_path)
    print(f"Video saved successfully to: {final_output_path}")

if __name__ == "__main__":
    main()
