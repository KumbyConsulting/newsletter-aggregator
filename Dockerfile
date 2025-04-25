FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create and set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set up Gunicorn config
ENV GUNICORN_CMD_ARGS="--workers=8 --threads=8 --worker-class=gthread --worker-tmp-dir /dev/shm --bind=0.0.0.0:$PORT --timeout=300 --max-requests=1000 --max-requests-jitter=50 --keep-alive=5 --access-logfile=- --error-logfile=- --log-level=info"

# Run the application
CMD exec gunicorn $GUNICORN_CMD_ARGS app:app 