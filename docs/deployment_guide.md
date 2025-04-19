# Newsletter Aggregator Deployment Guide

This document provides detailed instructions on how to deploy both the backend and frontend components of the Newsletter Aggregator application using Google Cloud Platform.

## Prerequisites

- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) installed
- Docker installed
- Git repository cloned
- GCP project created with billing enabled
- Required GCP APIs enabled:
  - Cloud Run API
  - Container Registry API
  - Secret Manager API
  - Firestore API
  - Cloud Storage API
  - Vertex AI API (if using Google's AI services)

## Environment Setup

1. First, clone the repository and navigate to the project folder:

```bash
git clone <repository-url>
cd newsletter-aggregator
```

2. Create and configure backend environment variables:

```bash
cp .env.example .env
```

Edit the `.env` file with your specific configuration values:

```
# Flask Configuration
FLASK_SECRET_KEY=your-secure-random-key
FLASK_ENV=production

# AI Service Configuration
GEMINI_API_KEY=your-gemini-api-key

# Google Cloud Configuration
GCP_PROJECT_ID=your-gcp-project-id
GCP_REGION=us-central1
GCS_BUCKET_NAME=newsletter-aggregator
USE_GCS_BACKUP=true
USE_VERTEX_AI=true
USE_CLOUD_LOGGING=true
STORAGE_BACKEND=firestore
USE_SECRET_MANAGER=true
```

3. Configure the frontend environment:

```bash
cd frontend
cp .env.example .env
```

Edit the frontend `.env` file:

```
# API Configuration
NEXT_PUBLIC_API_URL=https://your-backend-service-url
NEXT_PUBLIC_ENABLE_INSIGHTS=true
```

## Backend Deployment

### Option 1: Using Cloud Build (Recommended for Production)

1. Make sure you're authenticated with Google Cloud:

```bash
gcloud auth login
gcloud config set project your-gcp-project-id
```

2. (First time only) Create the secret for the Gemini API key:

```bash
echo -n "your-gemini-api-key" | gcloud secrets create GEMINI_API_KEY --data-file=-
```

3. Submit the build to Cloud Build:

```bash
cd /path/to/newsletter-aggregator  # Navigate to root directory
gcloud builds submit --config cloudbuild.yaml
```

This will:
- Build the container image
- Push it to Container Registry
- Deploy to Cloud Run with the specified configuration
- Set up environment variables and secrets

### Option 2: Manual Deployment

1. Build the Docker image:

```bash
docker build -t gcr.io/your-project-id/newsletter-aggregator:v1 .
```

2. Push the image to Container Registry:

```bash
docker push gcr.io/your-project-id/newsletter-aggregator:v1
```

3. Deploy to Cloud Run:

```bash
gcloud run deploy newsletter-aggregator \
  --image=gcr.io/your-project-id/newsletter-aggregator:v1 \
  --region=us-central1 \
  --platform=managed \
  --allow-unauthenticated \
  --memory=1Gi \
  --cpu=1 \
  --min-instances=0 \
  --max-instances=10 \
  --timeout=600s \
  --cpu-boost \
  --set-secrets=GEMINI_API_KEY=GEMINI_API_KEY:latest \
  --set-env-vars=GCP_PROJECT_ID=your-project-id,GCS_BUCKET_NAME=newsletter-aggregator,USE_GCS_BACKUP=true,USE_VERTEX_AI=true,USE_CLOUD_LOGGING=true,STORAGE_BACKEND=firestore,FLASK_ENV=production,USE_SECRET_MANAGER=true,SKIP_CHROMADB_INIT=true \
  --port=8080
```

## Frontend Deployment

### Option 1: Using Cloud Build (Recommended for Production)

1. Update the API URL in `frontend/cloudbuild.yaml` to point to your deployed backend service:

```yaml
'--set-env-vars=NEXT_PUBLIC_API_URL=https://your-backend-service-url'
```

2. Submit the build to Cloud Build:

```bash
cd frontend
gcloud builds submit --config cloudbuild.yaml
```

### Option 2: Using the Deployment Script

1. Update the `PROJECT_ID` and `NEXT_PUBLIC_API_URL` in `frontend/deploy.sh`:

```bash
PROJECT_ID="your-project-id"
...
--set-env-vars=NEXT_PUBLIC_API_URL=https://your-backend-service-url
```

2. Make the script executable and run it:

```bash
cd frontend
chmod +x deploy.sh
./deploy.sh
```

### Option 3: Manual Deployment

1. Build the Docker image:

```bash
cd frontend
docker build -t gcr.io/your-project-id/newsletter-frontend:v1 .
```

2. Push the image to Container Registry:

```bash
docker push gcr.io/your-project-id/newsletter-frontend:v1
```

3. Deploy to Cloud Run:

```bash
gcloud run deploy newsletter-frontend \
  --image=gcr.io/your-project-id/newsletter-frontend:v1 \
  --region=us-central1 \
  --platform=managed \
  --allow-unauthenticated \
  --memory=512Mi \
  --cpu=1 \
  --min-instances=0 \
  --max-instances=3 \
  --timeout=300s \
  --set-env-vars=NEXT_PUBLIC_API_URL=https://your-backend-service-url
```

## Verifying Deployment

### Backend Verification
After deployment, verify the backend is running properly:

```bash
curl https://your-backend-service-url/api/health
```

You should receive a JSON response indicating the service is healthy.

### Frontend Verification
Navigate to your deployed frontend URL in a browser:

```
https://newsletter-frontend-[unique-id].us-central1.run.app
```

## Additional Configuration

### Setting up a Custom Domain (Optional)

1. Configure a custom domain in Cloud Run:

```bash
gcloud run domain-mappings create --service newsletter-frontend --domain your-domain.com --region us-central1
```

2. Follow the DNS verification steps provided by Google Cloud.

### Continuous Deployment (Optional)

To set up continuous deployment with Cloud Build triggers:

1. Connect your GitHub/GitLab repository to Cloud Build
2. Create a trigger that executes the cloudbuild.yaml files on push to main branch

## Troubleshooting

### Common Issues:

1. **Deployment Failures**: Check Cloud Build logs for specific error messages.
2. **API Connection Issues**: Verify the NEXT_PUBLIC_API_URL environment variable matches your backend service URL exactly.
3. **Permission Errors**: Ensure the Cloud Run service account has necessary permissions to access other GCP services.

## Maintenance

### Updating the Application:

1. Make changes to your code and push to repository
2. Re-run the deployment process (Cloud Build or manual)
3. The new version will be deployed with zero downtime

### Monitoring:

Monitor your application using Cloud Run's built-in monitoring tools or set up Cloud Monitoring for more detailed insights. 