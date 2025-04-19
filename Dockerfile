FROM python:3.11-slim

WORKDIR /app

# Copy requirements first to leverage Docker caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt waitress

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV USE_CLOUD_LOGGING=true
ENV PORT=8080
ENV FLASK_ENV=production
ENV PYTHONDONTWRITEBYTECODE=1
ENV STORAGE_BACKEND=firestore
ENV SKIP_CHROMADB_INIT=true

# Create a non-root user
RUN useradd -m appuser

# Create simple startup script for quick start
RUN echo '#!/bin/bash\necho "Starting app on port $PORT"\ncd /app && python start.py' > /app/startup.sh && \
    chmod +x /app/startup.sh

# Copy the application code last to minimize rebuilds
COPY . .

# Create directories for persistent data and set permissions
RUN mkdir -p /app/data /app/chroma_db && \
    chown -R appuser:appuser /app && \
    chmod -R 755 /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8080

# Use the startup script
CMD ["/app/startup.sh"] 