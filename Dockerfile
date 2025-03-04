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
ENV PORT=8080

# Create a non-root user
RUN useradd -m appuser
USER appuser

# Expose port 8080 for the Flask app
EXPOSE 8080

# Use our custom startup script
CMD ["python", "start.py"] 