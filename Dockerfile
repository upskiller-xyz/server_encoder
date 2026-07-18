
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV (headless)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade build tooling & purge old bundled/cached wheels the base image ships
# (fixes pip / setuptools / wheel CVEs; ensurepip stashes vulnerable .whl files
# that scanners still flag even after an upgrade)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && { \
        find / -type d -name "_bundled" -path "*ensurepip*" -exec rm -rf {} + 2>/dev/null || true; \
        find / -type f -name "*.whl" -delete 2>/dev/null || true; \
        rm -rf /root/.cache/pip; \
    }

# Copy requirements first for better layer caching
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Set environment variables
ENV PORT=8082 \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=-1 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    OPENCV_IO_ENABLE_OPENEXR=0 \
    OMP_NUM_THREADS=1

# Make files read-only for security
RUN chmod -R 444 ./src/ && \
    chmod 444 ./requirements.txt

# Run with gunicorn
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 900 --access-logfile - --error-logfile - src.main:app