# Location: backend/Dockerfile
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for some Python packages
# libpq-dev — asyncpg needs PostgreSQL client libraries
# build-essential — some packages need gcc to compile
RUN apt-get update && apt-get install -y \
    libpq-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (cached layer — only rebuilds if requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directory for FAISS stores
RUN mkdir -p /tmp/opsiq_stores

# Expose port
EXPOSE 8000

# Start the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]