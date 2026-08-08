FROM python:3.12-slim

# PYTHONUNBUFFERED: without it, print() output stays invisible in
# `docker compose logs` until 8KB piles up — the postmortem pipeline
# would look frozen while it's actually working fine.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Install packages before copying code, so editing code doesn't
# trigger a full reinstall on every build.
COPY --chown=user requirements.txt ./requirements.txt
RUN --mount=type=cache,target=/home/user/.cache/pip,uid=1000,gid=1000 \
    python -m pip install -r requirements.txt

# Download the embeddings model at build time. Otherwise the app fetches
# ~90MB on every container start, and won't boot if huggingface.co is down.
ENV HF_HOME=/home/user/.cache/huggingface
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Same shape as the repo: backend/ and frontend/ side by side under /app.
# This is what makes _BACKEND_DIR.parent / "frontend" resolve correctly
# both on your laptop and inside the container.
COPY --chown=user backend/  /app/backend/
COPY --chown=user frontend/ /app/frontend/

RUN mkdir -p /tmp/opsiq_stores

# Run from inside backend/ so `uvicorn main:app` finds main.py
WORKDIR /app/backend

ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/login" || exit 1

# "exec" makes uvicorn the main process so Ctrl+C stops it immediately
# instead of waiting 10 seconds for a force-kill.
# --workers 1 is required, not incidental. See the note in session.py.
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1"]