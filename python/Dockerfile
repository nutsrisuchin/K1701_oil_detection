# Base image includes PyTorch + ultralytics + OpenCV (headless) pre-installed.
# GPU variant: ultralytics/ultralytics:latest-cuda  (requires nvidia-docker runtime)
FROM ultralytics/ultralytics:latest

WORKDIR /app

# Copy model weights into the image so no external mount is needed for them.
# If you retrain frequently, mount models as a volume instead.
COPY best_glass.pt  /app/models/best_glass.pt
COPY best_bubble.pt /app/models/best_bubble.pt

COPY main_2_rev1.py /app/main_2_rev1.py

# Default environment variables.
# Override with -e flags or docker-compose.yml.
ENV MODEL_DIR=/app/models \
    VIDEO_SOURCE=/data/input/video.mp4 \
    OUTPUT_DIR=/data/output \
    HEADLESS=true

# /data/input  → mount your video here
# /data/output → mount your output directory here
VOLUME ["/data/input", "/data/output"]

CMD ["python", "/app/main_2_rev1.py"]
