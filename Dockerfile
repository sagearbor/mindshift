# MindShift backend — FastAPI + live WebSocket audio pipeline.
# Container target for Cloud Run (or any container host). Deepgram is the STT
# path here; faster-whisper is intentionally NOT installed (heavy, and slow
# without a GPU) — the image stays light.
FROM python:3.11-slim

# System certs are enough for outbound TLS (Deepgram/Anthropic); the app also
# pins certifi in-code. No build tools needed for the pinned wheels.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install deps first for layer caching.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# App code (server/ only — apps/, tests/, docs/ are excluded via .dockerignore).
COPY server/ ./server/

WORKDIR /app/server

# Cloud Run provides $PORT (default 8080). Bind 0.0.0.0 so it's reachable.
# WebSockets on Cloud Run need a high request timeout — set --timeout=3600 and
# --min-instances=1 on `gcloud run deploy` (see docs/DEPLOY.md) so a live audio
# session isn't cut off or dropped on a cold start.
ENV PORT=8080
EXPOSE 8080
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
