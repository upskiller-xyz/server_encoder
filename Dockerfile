
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

# Pinned build-tooling versions keep the image reproducible; override with
# --build-arg if a newer patched release is needed. These pins clear the
# pip / setuptools / wheel CVE scan findings (setuptools 83 also vendors the
# patched jaraco.context 6.1 + wheel 0.46.3 under setuptools/_vendor/).
ARG PIP_VERSION=26.1.2
ARG SETUPTOOLS_VERSION=83.0.0
ARG WHEEL_VERSION=0.47.0
# Upgrade build tooling, then purge the vulnerable bundled .whl the base image
# ships: the scoped find deletes the ensurepip _bundled wheels (old pip/
# setuptools/wheel that scanners flag), plus the pip cache. Scoped to
# /usr/local/lib on purpose (no full-filesystem scan). The block ends in
# `true`, so this best-effort cleanup never fails the build; the pip upgrade
# stays &&-gated so a failed upgrade still does.
RUN pip install --no-cache-dir --upgrade "pip==${PIP_VERSION}" "setuptools==${SETUPTOOLS_VERSION}" "wheel==${WHEEL_VERSION}" \
    && { \
        find /usr/local/lib -type d -name "_bundled" -path "*ensurepip*" -exec rm -rf {} + 2>/dev/null; \
        rm -rf /root/.cache/pip; \
        true; \
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