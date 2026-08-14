# MindShift backend — FastAPI + live WebSocket audio pipeline.
# Container target for Cloud Run (or any container host). Deepgram is the
# default STT path; faster-whisper is installed by default too (see
# INSTALL_WHISPER below) so uploads can run on our own compute
# (MINDSHIFT_UPLOAD_STT=whisper) and survive a Deepgram outage via the
# automatic fallback.
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

# Voice enrollment (ECAPA-TDNN speaker embeddings) is OPTIONAL and OFF by default:
# torch + speechbrain add ~1.5-2GB to the image, so the default build stays light
# (Deepgram-only). Enable it at build time WITHOUT editing this file:
#   docker build --build-arg INSTALL_VOICE=1 ...
# The CPU-only torch wheel is used deliberately (Cloud Run is CPU-only; the CUDA
# wheel would be far larger). First model load fetches a ~20MB checkpoint from the
# HF Hub and caches it; measured cold model-load ~2-4s on a Cloud Run vCPU, so run
# with --min-instances=1 to keep the loaded model warm. See requirements-voice.txt.
ARG INSTALL_VOICE=1
COPY requirements-voice.txt ./
RUN if [ "$INSTALL_VOICE" = "1" ]; then \
      pip install --no-cache-dir torch torchaudio \
        --index-url https://download.pytorch.org/whl/cpu && \
      pip install --no-cache-dir -r requirements-voice.txt ; \
    fi

# Local Whisper STT (faster-whisper) is optional but ON by default: it powers
# MINDSHIFT_UPLOAD_STT=whisper for prerecorded uploads AND the automatic
# fallback when Deepgram is unavailable, plus STT_PROVIDER=whisper for live
# sessions. Disable at build time WITHOUT editing this file:
#   docker build --build-arg INSTALL_WHISPER=0 ...
# The model is PRE-DOWNLOADED at build time (into the HF cache the runtime
# loader reads) so a cold start never blocks on a Hugging Face fetch.
# WHISPER_MODEL=small/int8: best accuracy of the sizes that fit comfortably in
# the service's 2Gi Cloud Run memory alongside torch/ECAPA (~400-500MB RSS
# loaded; medium would roughly triple that and starve the media pipeline).
# The ENV mirrors the ARG so the runtime loads EXACTLY the size that was
# pre-downloaded; a WHISPER_MODEL set at deploy time still overrides (but then
# cold starts pay the download for non-prebaked sizes).
ARG INSTALL_WHISPER=1
ARG WHISPER_MODEL=small
COPY requirements-whisper.txt ./
RUN if [ "$INSTALL_WHISPER" = "1" ]; then \
      pip install --no-cache-dir -r requirements-whisper.txt && \
      python -c "from faster_whisper import WhisperModel; \
WhisperModel('${WHISPER_MODEL}', device='cpu', compute_type='int8')" ; \
    fi
ENV WHISPER_MODEL=${WHISPER_MODEL}

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
