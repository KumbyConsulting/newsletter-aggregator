FROM python:3.11-slim

WORKDIR /app

# Copy requirements first to leverage Docker caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt waitress

# Copy the rest of the application
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV USE_CLOUD_LOGGING=true
# ENV PORT=8080  # Removed as Cloud Run sets this automatically
ENV HEALTH_PORT=8081
ENV FLASK_ENV=production
ENV PYTHONDONTWRITEBYTECODE=1
ENV STORAGE_BACKEND=chromadb

# Create a non-root user first
RUN useradd -m appuser

# Create directories for persistent data and set permissions
RUN mkdir -p /app/data /app/chroma_db && \
    chown -R appuser:appuser /app/data /app/chroma_db && \
    chmod -R 755 /app/data /app/chroma_db && \
    chown -R appuser:appuser /app

# Create a startup script
RUN echo '#!/bin/bash\necho "Starting application..."\n\n# Run initialization scripts\necho "Initializing application..."\npython -c "import chromadb; print(\"ChromaDB initialized\"); client = chromadb.PersistentClient(path=\"/app/chroma_db\")"\n\n# Start the application with timeout\necho "Starting main application..."\ntimeout 60s python start.py' > /app/startup.sh && \
    chmod +x /app/startup.sh

# Add pre-initialization for ChromaDB
RUN python -c "import chromadb; chromadb.PersistentClient(path=\"/app/chroma_db\")" || true

# Switch to non-root user
USER appuser

# Expose ports for main app and health check
EXPOSE 8080 8081

# Add health check
HEALTHCHECK --interval=5s --timeout=3s --start-period=10s --retries=3 CMD curl -f http://localhost:8081/_ah/health || exit 1

# Use our custom startup script
CMD ["/app/startup.sh"] 