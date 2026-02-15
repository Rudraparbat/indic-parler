FROM python:3.11-slim

# ─────────────────────────────────────────────
# System dependencies
# ─────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    ffmpeg \
    libsndfile1 \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# ─────────────────────────────────────────────
# Working directory
# ─────────────────────────────────────────────
WORKDIR /app

# ─────────────────────────────────────────────
# Install Python dependencies
# ─────────────────────────────────────────────
COPY requirements.txt .

# Skip LFS for git installs to avoid LFS download errors
ENV GIT_LFS_SKIP_SMUDGE=1

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir transformers==4.46.1 && \
    pip install --no-cache-dir -r requirements.txt

# ─────────────────────────────────────────────
# Copy server code
# ─────────────────────────────────────────────
COPY main.py .

# ─────────────────────────────────────────────
# Expose port
# ─────────────────────────────────────────────
EXPOSE 8080

# ─────────────────────────────────────────────
# Start server
# ─────────────────────────────────────────────
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]