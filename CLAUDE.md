# K1701 Oil Detection — CLAUDE.md

## Project Overview

Automated lubrication monitor that detects oil/bubble drops passing through sight glasses on an industrial pipeline. Uses two YOLO models (glass detection + bubble tracking) and ByteTrack to count drops per minute (DPM) per glass in real time.

**Goal:** Wrap the existing Python inference pipeline into a web app where users upload a video and watch annotated results live.

---

## Repository Layout

```
K1701_oil_detection/
├── best_glass.pt          # YOLO model — detects sight glass boundaries
├── best_bubble.pt         # YOLO model — detects/tracks oil bubbles
├── MVI_9973.MP4           # Sample input video
├── python/
│   ├── main_2_rev1.py     # Current inference pipeline (production)
│   ├── main_2.py          # Earlier iteration
│   ├── main_rev1.py       # Earlier iteration
│   ├── main.py            # Earliest iteration
│   ├── Dockerfile         # CPU image (ultralytics/ultralytics:latest)
│   └── docker-compose.yml # Local dev compose
├── typescript/            # TypeScript / React frontend (to be built)
└── output/                # Annotated output videos
```

---

## Core Inference Logic (`python/main_2_rev1.py`)

- **Stage 1 — Full-frame inference:** `model_glass.predict()` finds sight glass bounding boxes; `model_bubble.track()` runs ByteTrack across the whole frame.
- **Stage 2 — Mathematical filtering:** For each glass box, filter which bubble centroids fall inside it, then delegate to a `SightGlass` tracker.
- **`SightGlass` class:** Maintains a configurable drop zone (ratio-based inside each glass), detects new drops via count-delta + ID-change signals, computes rolling DPM over the last 60 s.
- **Env vars:** `MODEL_DIR`, `VIDEO_SOURCE`, `OUTPUT_DIR`, `HEADLESS` — all overridable at runtime.
- **Output:** Annotated MP4 written to `OUTPUT_DIR`.

---

## Web App Architecture

Two parallel implementations sharing the same inference backend:

### Option A — Streamlit (Python, fast to prototype)

```
python/
├── app_streamlit.py       # Streamlit entry point
├── inference/
│   ├── pipeline.py        # Refactored pipeline (generator that yields annotated frames)
│   └── sight_glass.py     # SightGlass class (extracted from main_2_rev1.py)
└── requirements.txt
```

- User uploads video → saved to a temp file.
- `pipeline.py` yields `(frame_bgr, metrics_dict)` tuples.
- Streamlit displays frames via `st.image()` inside a loop; metrics shown in `st.metric()` widgets.
- Use `st.empty()` placeholders to update in place (avoid full re-render).

Run locally:
```bash
streamlit run python/app_streamlit.py
```

### Option B — TypeScript / React + FastAPI backend

```
typescript/          # React + Vite frontend
├── src/
│   ├── App.tsx
│   ├── components/
│   │   ├── VideoUploader.tsx
│   │   ├── LivePlayer.tsx      # Renders MJPEG or WebSocket frames
│   │   └── MetricsDashboard.tsx
│   └── hooks/
│       └── useDetectionStream.ts
├── package.json
└── vite.config.ts

python/
├── api/
│   ├── main_api.py            # FastAPI app
│   ├── routers/
│   │   ├── upload.py          # POST /upload  → returns job_id
│   │   └── stream.py          # GET  /stream/{job_id}  → MJPEG or WS
│   └── pipeline.py            # Same generator as Streamlit option
└── requirements-api.txt
```

- Frontend POSTs video → backend assigns `job_id`, starts background inference.
- Live frames streamed via **WebSocket** (`/ws/{job_id}`) as JPEG bytes.
- Metrics pushed in the same WebSocket message as JSON alongside the frame.

Run locally:
```bash
# Backend
uvicorn python.api.main_api:app --reload --port 8000
# Frontend
cd typescript && npm run dev
```

---

## Key Design Rules

- **Do not modify `main_2_rev1.py` directly.** Extract `SightGlass` and the main loop into `pipeline.py` as a generator; keep the original file intact for reference.
- `DROP_ZONE_CONFIG` must remain configurable per glass ID — expose it as a UI slider or JSON input if users need to calibrate.
- Always run YOLO inference with `verbose=False` to avoid flooding logs.
- Model weights (`best_glass.pt`, `best_bubble.pt`) are committed to the repo. Do not re-download them; mount or copy them into containers.
- Target frame processing >= 15 fps on CPU; GPU preferred for production.

---

## Docker

Current `python/Dockerfile` base: `ultralytics/ultralytics:latest` (CPU).
For GPU: swap base to `ultralytics/ultralytics:latest-cuda` and enable the `deploy.resources` block in `docker-compose.yml`.

Build & run:
```bash
cd python
docker compose up --build
```

Volumes:
- `/data/input` — mount video source
- `/data/output` — annotated video appears here

---

## Deployment Methods

### GCP (recommended for GPU workloads)

| Method | When to use |
|--------|-------------|
| **Cloud Run** | Streamlit or FastAPI, CPU, serverless, pay-per-request |
| **Cloud Run + GPU** | Same but with L4/T4 GPU (Cloud Run GPU preview) |
| **GKE Autopilot** | Production scale, multiple concurrent jobs |
| **Vertex AI Prediction** | Managed model serving separate from the web layer |

**Cloud Run quick deploy (Streamlit):**
```bash
# Build & push image
gcloud builds submit --tag gcr.io/PROJECT_ID/k1701-oil-detection python/

# Deploy (increase memory for large videos)
gcloud run deploy k1701-oil-detection \
  --image gcr.io/PROJECT_ID/k1701-oil-detection \
  --platform managed \
  --region asia-southeast1 \
  --memory 4Gi \
  --timeout 3600 \
  --allow-unauthenticated
```

**GCS for video uploads** (avoid Cloud Run's ephemeral filesystem):
- Upload to a GCS bucket via signed URL.
- Inference worker reads from GCS, writes annotated output back.

### AWS

| Method | When to use |
|--------|-------------|
| **App Runner** | Streamlit or FastAPI, simple, auto-scaling, no infra |
| **ECS Fargate** | Container-first, CPU or GPU (Fargate GPU preview) |
| **EC2 + GPU (g4dn)** | Predictable high-throughput, lowest inference latency |
| **SageMaker** | Managed model endpoints if decoupling inference from web |

**App Runner quick deploy:**
```bash
# Push to ECR
aws ecr create-repository --repository-name k1701-oil-detection
docker tag k1701-oil-detection:latest <account>.dkr.ecr.<region>.amazonaws.com/k1701-oil-detection:latest
docker push <account>.dkr.ecr.<region>.amazonaws.com/k1701-oil-detection:latest

# Create App Runner service via console or CLI:
aws apprunner create-service \
  --service-name k1701-oil-detection \
  --source-configuration '{"ImageRepository":{"ImageIdentifier":"<ecr-uri>","ImageConfiguration":{"Port":"8501"},"ImageRepositoryType":"ECR"}}'
```

**S3 for video uploads:** presigned URL -> S3 bucket -> inference worker reads and writes back.

---

## Environment Variables (all implementations)

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_DIR` | `/app/models` | Directory containing `.pt` weight files |
| `VIDEO_SOURCE` | `/data/input/video.mp4` | Input video path |
| `OUTPUT_DIR` | `/data/output` | Where annotated MP4 is written |
| `HEADLESS` | `true` | Set `false` only for local desktop display |
| `CONFIDENCE_GLASS` | `0.5` | YOLO confidence threshold for glass detection |
| `CONFIDENCE_BUBBLE` | `0.15` | YOLO confidence threshold for bubble tracking |

---

## Common Commands

```bash
# Run inference locally (CPU, headless)
cd python && python main_2_rev1.py

# Run with Docker Compose
cd python && docker compose up --build

# Run Streamlit app (once built)
streamlit run python/app_streamlit.py --server.port 8501

# Run FastAPI backend (once built)
uvicorn python.api.main_api:app --reload --port 8000

# Run TypeScript frontend (once built)
cd typescript && npm run dev

# Type-check TypeScript
cd typescript && npm run typecheck

# Lint Python
cd python && ruff check .
```

---

## Out of Scope

- Training or fine-tuning YOLO models — weights are final.
- Real-time camera feed (RTSP) — input is always an uploaded file for now.
- Authentication/auth layer — internal tool, unauthenticated endpoints acceptable for MVP.
