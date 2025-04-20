#!/bin/bash
set -e

# Get the project ID and number
PROJECT_ID=$(gcloud config get-value project)
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")

# Replace placeholders in the OpenAPI spec
sed -i "s/SERVICE_PROJECT_NUMBER/$PROJECT_NUMBER/g" openapi.yaml

# Create a new API Gateway API
echo "Creating API Gateway API..."
gcloud api-gateway apis create newsletter-aggregator-api \
  --project=$PROJECT_ID

# Create a new API Config
echo "Creating API Config..."
gcloud api-gateway api-configs create newsletter-aggregator-config \
  --api=newsletter-aggregator-api \
  --openapi-spec=openapi.yaml \
  --project=$PROJECT_ID \
  --backend-auth-service-account=newsletter-aggregator-sa@$PROJECT_ID.iam.gserviceaccount.com

# Create a new API Gateway
echo "Creating API Gateway..."
gcloud api-gateway gateways create newsletter-aggregator-gateway \
  --api=newsletter-aggregator-api \
  --api-config=newsletter-aggregator-config \
  --location=us-central1 \
  --project=$PROJECT_ID

# Get the gateway URL
GATEWAY_URL=$(gcloud api-gateway gateways describe newsletter-aggregator-gateway \
  --location=us-central1 \
  --format="value(defaultHostname)" \
  --project=$PROJECT_ID)

echo "API Gateway deployed successfully!"
echo "Gateway URL: https://$GATEWAY_URL" 