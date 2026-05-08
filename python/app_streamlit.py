import streamlit as st
import cv2
import time
import os
import tempfile
from pathlib import Path
from collections import deque
from ultralytics import YOLO

# ==========================================
# 0. CONFIGURATION
# ==========================================
# Model weights sit in the project root (one level up from this script)
MODEL_DIR = Path(__file__).parent.parent

DROP_ZONE_CONFIG = {
    0: (0.4, 0.8, 0.5, 0.9),
    1: (0.4, 0.8, 0.5, 0.9),
    2: (0.4, 0.8, 0.65, 0.9),
    3: (0.4, 0.8, 0.5, 0.9),
    4: (0.3, 0.7, 0.6, 0.9),
    5: (0.3, 0.7, 0.56, 0.9),
    6: (0.4, 0.7, 0.55, 0.9),
}


# ==========================================
# 1. SIGHT GLASS TRACKER (unchanged from main_2_rev1.py)
# ==========================================
class SightGlass:
    def __init__(self, glass_id, drop_zone_ratios):
        self.glass_id = glass_id
        self.total_drops = 0
        self.drop_timestamps = deque()
        self.x_min_ratio, self.x_max_ratio, self.y_min_ratio, self.y_max_ratio = drop_zone_ratios
        self.prev_zone_count = 0
        self.prev_ids_in_zone = set()

    def update_and_count(self, main_frame, glass_box, all_bubbles_in_glass):
        gx1, gy1, gx2, gy2 = glass_box
        roi_width = gx2 - gx1
        roi_height = gy2 - gy1

        dz_x1 = int(gx1 + (roi_width * self.x_min_ratio))
        dz_x2 = int(gx1 + (roi_width * self.x_max_ratio))
        dz_y1 = int(gy1 + (roi_height * self.y_min_ratio))
        dz_y2 = int(gy1 + (roi_height * self.y_max_ratio))

        cv2.rectangle(main_frame, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
        cv2.rectangle(main_frame, (dz_x1, dz_y1), (dz_x2, dz_y2), (255, 0, 255), 2)

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

        current_time = time.time()
        current_zone_count = 0
        current_ids_in_zone = set()

        for (bx1, by1, bx2, by2, track_id) in all_bubbles_in_glass:
            cx = bx1 + ((bx2 - bx1) // 2)
            cy = by1 + ((by2 - by1) // 2)
            if (dz_x1 <= cx <= dz_x2) and (dz_y1 <= cy <= dz_y2):
                current_zone_count += 1
                current_ids_in_zone.add(track_id)
                cv2.rectangle(main_frame, (bx1, by1), (bx2, by2), (0, 255, 255), 1)
                cv2.putText(main_frame, f"{track_id}", (bx1, by1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
            else:
                cv2.rectangle(main_frame, (bx1, by1), (bx2, by2), (0, 0, 100), 1)

        if current_zone_count > 0:
            new_drops_by_count = max(0, current_zone_count - self.prev_zone_count)
            new_ids = current_ids_in_zone - self.prev_ids_in_zone
            new_drops_by_id = len(new_ids) if current_zone_count == self.prev_zone_count else 0
            new_drops = max(new_drops_by_count, new_drops_by_id)
            if new_drops > 0:
                self.total_drops += new_drops
                for _ in range(new_drops):
                    self.drop_timestamps.append(current_time)

        self.prev_zone_count = current_zone_count
        self.prev_ids_in_zone = current_ids_in_zone

        while self.drop_timestamps and self.drop_timestamps[0] < current_time - 60:
            self.drop_timestamps.popleft()

        current_dpm = len(self.drop_timestamps)
        text_color = (0, 0, 255) if current_zone_count > 0 else (255, 255, 0)
        cv2.putText(main_frame, f"ID:{self.glass_id} DPM:{current_dpm} Tot:{self.total_drops}",
                    (gx1, gy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

        return current_dpm


# ==========================================
# 2. MODEL LOADER (cached — loads once per session)
# ==========================================
@st.cache_resource
def load_models(conf_glass: float, conf_bubble: float):
    glass_path = MODEL_DIR / "best_glass.pt"
    bubble_path = MODEL_DIR / "best_bubble.pt"
    if not glass_path.exists() or not bubble_path.exists():
        st.error(f"Model weights not found in {MODEL_DIR}. Expected best_glass.pt and best_bubble.pt.")
        st.stop()
    model_glass = YOLO(str(glass_path))
    model_bubble = YOLO(str(bubble_path))
    return model_glass, model_bubble, conf_glass, conf_bubble


# ==========================================
# 3. STREAMLIT UI
# ==========================================
def main():
    st.set_page_config(
        page_title="K1701 Oil Drop Detection",
        page_icon="🔬",
        layout="wide",
    )

    st.title("K1701 — Automated Oil Drop Monitor")
    st.caption("Upload a video to run real-time oil bubble detection across sight glasses.")

    # Persist the output video bytes across Streamlit reruns (e.g. when download button is clicked)
    if "result_bytes" not in st.session_state:
        st.session_state.result_bytes = None
    if "result_filename" not in st.session_state:
        st.session_state.result_filename = None

    # --- Sidebar configuration ---
    with st.sidebar:
        st.header("Detection Settings")
        conf_glass = st.slider("Glass confidence threshold", 0.1, 1.0, 0.5, 0.05)
        conf_bubble = st.slider("Bubble confidence threshold", 0.05, 1.0, 0.15, 0.05)
        preview_every = st.number_input(
            "Preview every N frames",
            min_value=1, max_value=30, value=5,
            help=(
                "Inference runs on every frame (keeps tracking accurate). "
                "This only controls how often the live preview updates. "
                "Higher = faster processing, choppier preview."
            ),
        )
        preview_scale = st.slider(
            "Preview scale (%)",
            min_value=25, max_value=100, value=60, step=5,
            help="Downscales the displayed frame only — output video is always full resolution.",
        )
        st.divider()
        st.info(
            "**Models loaded:**\n"
            f"- `best_glass.pt`\n"
            f"- `best_bubble.pt`\n\n"
            "Weights are embedded in the app — no upload needed."
        )

    # Load models (cached after first run)
    with st.spinner("Loading YOLO models..."):
        model_glass, model_bubble, *_ = load_models(conf_glass, conf_bubble)
    st.success("Models ready.", icon="✅")

    # --- Video upload ---
    uploaded = st.file_uploader(
        "Upload a video file",
        type=["mp4", "mov", "avi", "mkv", "MP4", "MOV"],
        help="Video will be processed frame-by-frame. Large files take longer.",
    )

    if uploaded is None:
        st.info("Upload a video above to begin.")
        return

    col_info, col_btn = st.columns([3, 1])
    col_info.write(f"**File:** {uploaded.name}  |  **Size:** {uploaded.size / 1_000_000:.1f} MB")
    run = col_btn.button("▶ Run Detection", type="primary", use_container_width=True)

    if not run:
        # Show download button if a previous run already produced output
        if st.session_state.result_bytes is not None:
            st.download_button(
                label="⬇ Download annotated video",
                data=st.session_state.result_bytes,
                file_name=st.session_state.result_filename,
                mime="video/mp4",
                type="primary",
            )
        return

    # New run started — clear any previous result
    st.session_state.result_bytes = None
    st.session_state.result_filename = None

    # --- Save upload to temp file ---
    tmp_input = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_input.write(uploaded.read())
    tmp_input.flush()
    tmp_input_path = tmp_input.name
    tmp_input.close()

    cap = cv2.VideoCapture(tmp_input_path)
    if not cap.isOpened():
        st.error("Could not open the uploaded video.")
        os.unlink(tmp_input_path)
        return

    fps = max(int(cap.get(cv2.CAP_PROP_FPS)), 1)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # --- Setup annotated video writer (try codecs in order, same as original pipeline) ---
    tmp_output_path = tempfile.mktemp(suffix=".mp4")
    out = None
    for codec in ["avc1", "mp4v", "XVID"]:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        candidate = cv2.VideoWriter(tmp_output_path, fourcc, fps, (width, height))
        if candidate.isOpened():
            out = candidate
            break
        candidate.release()

    if out is None:
        st.error("No working video codec found on this system. Output video cannot be saved.")
        cap.release()
        os.unlink(tmp_input_path)
        return

    # --- UI placeholders ---
    st.divider()
    status_text = st.empty()
    progress_bar = st.progress(0.0)
    frame_placeholder = st.empty()
    metrics_placeholder = st.empty()

    trackers: dict[int, SightGlass] = {}
    frame_idx = 0
    start_time = time.time()

    # ==========================================
    # 4. MAIN PROCESSING LOOP
    # ==========================================
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Stage 1: detect glasses
        glass_results = model_glass.predict(source=frame, conf=conf_glass, verbose=False)
        glass_boxes = []
        for result in glass_results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                glass_boxes.append((x1, y1, x2, y2))
        glass_boxes.sort(key=lambda b: b[0])

        # Stage 1: track bubbles
        bubble_results = model_bubble.track(source=frame, conf=conf_bubble, verbose=False, persist=True)
        all_bubble_boxes = []
        for result in bubble_results:
            for box in result.boxes:
                if box.id is None:
                    continue
                bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                track_id = int(box.id[0])
                all_bubble_boxes.append((bx1, by1, bx2, by2, track_id))

        # Stage 2: per-glass filtering and counting
        for i, glass_box in enumerate(glass_boxes):
            gx1, gy1, gx2, gy2 = glass_box
            if i not in trackers:
                zone_config = DROP_ZONE_CONFIG.get(i, (0.1, 0.9, 0.1, 0.9))
                trackers[i] = SightGlass(glass_id=i, drop_zone_ratios=zone_config)

            bubbles_in_glass = [
                (bx1, by1, bx2, by2, tid)
                for (bx1, by1, bx2, by2, tid) in all_bubble_boxes
                if (gx1 <= bx1 + (bx2 - bx1) // 2 <= gx2) and (gy1 <= by1 + (by2 - by1) // 2 <= gy2)
            ]
            trackers[i].update_and_count(frame, glass_box, bubbles_in_glass)

        out.write(frame)
        frame_idx += 1

        # --- Update UI every N frames ---
        if frame_idx % preview_every == 0 or frame_idx == 1:
            elapsed = time.time() - start_time
            fps_actual = frame_idx / elapsed if elapsed > 0 else 0
            eta_s = (total_frames - frame_idx) / fps_actual if fps_actual > 0 else 0

            status_text.markdown(
                f"**Frame** {frame_idx} / {total_frames} &nbsp;|&nbsp; "
                f"**Speed** {fps_actual:.1f} fps &nbsp;|&nbsp; "
                f"**ETA** {eta_s:.0f} s"
            )
            progress_bar.progress(min(frame_idx / total_frames, 1.0))

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if preview_scale < 100:
                scale = preview_scale / 100.0
                preview_h = int(rgb.shape[0] * scale)
                preview_w = int(rgb.shape[1] * scale)
                rgb = cv2.resize(rgb, (preview_w, preview_h), interpolation=cv2.INTER_LINEAR)
            frame_placeholder.image(rgb, channels="RGB", use_container_width=True)

            if trackers:
                with metrics_placeholder.container():
                    cols = st.columns(len(trackers))
                    for i, (glass_id, tracker) in enumerate(sorted(trackers.items())):
                        dpm = len(tracker.drop_timestamps)
                        cols[i].metric(
                            label=f"Glass {glass_id}",
                            value=f"{dpm} DPM",
                            delta=f"Total: {tracker.total_drops} drops",
                        )

    cap.release()
    out.release()
    os.unlink(tmp_input_path)

    # ==========================================
    # 5. FINAL SUMMARY + DOWNLOAD
    # ==========================================
    progress_bar.progress(1.0)
    status_text.success(f"Done — processed {frame_idx} frames in {time.time() - start_time:.1f} s")

    with metrics_placeholder.container():
        st.subheader("Final Results")
        if trackers:
            cols = st.columns(len(trackers))
            for i, (glass_id, tracker) in enumerate(sorted(trackers.items())):
                cols[i].metric(
                    label=f"Glass {glass_id}",
                    value=f"{tracker.total_drops} drops",
                )

    video_name = os.path.splitext(uploaded.name)[0]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_filename = f"{video_name}_detected_{timestamp}.mp4"

    if not os.path.exists(tmp_output_path):
        st.error("Output video was not created — no frames were written. Check codec support.")
        return

    with open(tmp_output_path, "rb") as f:
        st.session_state.result_bytes = f.read()
    st.session_state.result_filename = output_filename
    os.unlink(tmp_output_path)

    st.download_button(
        label="⬇ Download annotated video",
        data=st.session_state.result_bytes,
        file_name=st.session_state.result_filename,
        mime="video/mp4",
        type="primary",
    )


if __name__ == "__main__":
    main()
