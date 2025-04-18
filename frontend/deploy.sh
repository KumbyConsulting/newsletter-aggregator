#!/bin/bash
# Script to deploy the frontend to Google Cloud

# Set variables
PROJECT_ID="newsletter-450510"
REGION="us-central1"
SERVICE_NAME="newsletter-frontend"
IMAGE_NAME="gcr.io/$PROJECT_ID/newsletter-frontend:v1"

# Print current status
echo "Starting deployment of $SERVICE_NAME to $PROJECT_ID in $REGION"

# Set the GCP project
gcloud config set project $PROJECT_ID

# Build the Docker image
echo "Building Docker image..."
docker build -t $IMAGE_NAME .

# Push the image to Google Container Registry
echo "Pushing image to Container Registry..."
docker push $IMAGE_NAME

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --image=$IMAGE_NAME \
  --region=$REGION \
  --platform=managed \
  --allow-unauthenticated \
  --memory=512Mi \
  --cpu=1 \
  --min-instances=0 \
  --max-instances=3 \
  --timeout=300s \
  --set-env-vars=NEXT_PUBLIC_API_URL=https://newsletter-aggregator-857170198287.us-central1.run.app/

echo "Deployment completed. Service should be available soon at:"
echo "https://$SERVICE_NAME-857170198287.$REGION.run.app" 