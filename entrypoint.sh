#!/bin/bash
# Fetch secret from Secret Manager and export as env var
export GEMINI_API_KEY=$(gcloud secrets versions access latest --secret="GEMINI_API_KEY" --project="$GOOGLE_CLOUD_PROJECT")
echo "GEMINI_API_KEY is: $GEMINI_API_KEY"  # DEBUG: Remove after confirming
# Start the app (adjust as needed)
exec gunicorn -b :$PORT app:app
# For Flask directly, use: exec flask run --host=0.0.0.0 --port=$PORT
# For uvicorn (FastAPI): exec uvicorn app:app --host 0.0.0.0 --port $PORT 