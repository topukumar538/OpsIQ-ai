FROM python:3.12-slim

# PYTHONUNBUFFERED is load-bearing: without it stdout is block-buffered off a
# TTY and the postmortem pipeline's progress output stays invisible in
# `docker compose logs` until 8KB accumulates.
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

# Requirements first so source edits don't invalidate the install layer.
# No --no-cache-dir: it would defeat the BuildKit cache mount below.
# uid/gid on the mount because we run as user 1000, not root.
COPY --chown=user requirements.txt ./requirements.txt
RUN --mount=type=cache,target=/home/user/.cache/pip,uid=1000,gid=1000 \
    python -m pip install -r requirements.txt

# Bake the embeddings model so cold starts don't re-download ~90MB and boot
# doesn't depend on huggingface.co being reachable. Before the source COPY so
# editing code doesn't invalidate this layer.
ENV HF_HOME=/home/user/.cache/huggingface
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Mirror the repo layout: backend and frontend as siblings under /app, so
# _BACKEND_DIR.parent / "frontend" resolves identically in dev and in the
# container.
COPY --chown=user backend/  /app/backend/
COPY --chown=user frontend/ /app/frontend/

RUN mkdir -p /tmp/opsiq_stores

WORKDIR /app/backend

ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/login" || exit 1

# exec so uvicorn is PID 1 and receives SIGTERM directly — otherwise the shell
# swallows it and the container waits out the full kill timeout.
#
# --workers 1 is required, not incidental: the session cache and locks are
# in-process. See the note at the top of session.py.
#
# --proxy-headers makes uvicorn read X-Forwarded-For, so request.client.host is
# the real client rather than the reverse proxy. Without it, every request
# behind a proxy shares a single rate-limit bucket and the first user to hit
# the limit locks out everyone else.
# --forwarded-allow-ips='*' is only safe because nothing reaches this container
# except through the platform's proxy.
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1 --proxy-headers --forwarded-allow-ips='*'"]