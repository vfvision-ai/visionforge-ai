# Multi-stage build for smaller final image
# Stage 1: Builder - Install dependencies
FROM python:3.9-slim-bookworm AS builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update -o Acquire::ForceIPv4=true && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git && \
    rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt requirements-dev.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime - Create minimal production image
FROM python:3.9-slim-bookworm AS runtime

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app" \
    ENVIRONMENT=production

# Install only runtime dependencies
RUN apt-get update -o Acquire::ForceIPv4=true && \
    apt-get install -y --no-install-recommends \
        curl \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 mluser && \
    chown -R mluser:mluser /app && \
    mkdir -p /app/experiments /app/uploads/extracted /app/data /app/logs /app/models && \
    chmod -R 777 /app/experiments /app/uploads /app/data /app/logs /app/models

# Switch to non-root user
USER mluser

# Expose the port that Streamlit runs on
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Command to run the application
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=true"]
