#!/bin/bash
set -e

echo "======= FIXING API GATEWAY DEPLOYMENT ======="
PROJECT_ID=$(gcloud config get-value project)
REGION="us-central1"
API_NAME="newsletter-aggregator-api"
GATEWAY_NAME="newsletter-aggregator-gateway"
BACKEND_URL=$(gcloud run services describe newsletter-aggregator --region=$REGION --format="get(status.url)")
SERVICE_ACCOUNT="newsletter-aggregator-sa@${PROJECT_ID}.iam.gserviceaccount.com"

echo "Current Cloud Run service URL: $BACKEND_URL"
echo "API Gateway name: $GATEWAY_NAME"

# Check logs for API Gateway errors
echo -e "\n1. Checking Cloud Run service recent logs:"
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=newsletter-aggregator AND severity>=ERROR" --limit 5 --format json | jq

# Check API Gateway logs
echo -e "\n2. Checking API Gateway logs:"
gcloud logging read "resource.type=api_gateway AND resource.labels.gateway_id=$GATEWAY_NAME AND severity>=ERROR" --limit 5 --format json | jq

# Skip validation as it's not available in this gcloud version
echo -e "\n3. Skipping OpenAPI spec validation (not available in current gcloud version)"

# Check that the OpenAPI spec has correct backend service URL
echo -e "\n4. Checking if OpenAPI spec has correct backend URL:"
SPEC_BACKEND_URL=$(grep -o "https://[^\"]*\.run\.app" openapi.yaml | head -1)
echo "Backend URL in OpenAPI spec: $SPEC_BACKEND_URL"
echo "Actual Cloud Run URL: $BACKEND_URL"

if [ "$SPEC_BACKEND_URL" != "$BACKEND_URL" ]; then
  echo -e "\n=== FIXING: Updating backend URL in OpenAPI spec ==="
  # Backup the original file
  cp openapi.yaml openapi.yaml.bak
  # Replace the backend URL
  sed -i "s|$SPEC_BACKEND_URL|$BACKEND_URL|g" openapi.yaml
  echo "Updated backend URL in OpenAPI spec."
fi

# Create a new API config and update the gateway
echo -e "\n5. Creating a new API config version:"
NEW_CONFIG_ID="newsletter-aggregator-config-fixed-$(date +%Y%m%d%H%M%S)"
echo "New config ID: $NEW_CONFIG_ID"

gcloud api-gateway api-configs create "${NEW_CONFIG_ID}" \
  --api=${API_NAME} \
  --openapi-spec=openapi.yaml \
  --project=${PROJECT_ID} \
  --backend-auth-service-account=${SERVICE_ACCOUNT}

echo -e "\n6. Updating API Gateway to use new config:"
gcloud api-gateway gateways update ${GATEWAY_NAME} \
  --api=${API_NAME} \
  --api-config=${NEW_CONFIG_ID} \
  --location=${REGION} \
  --project=${PROJECT_ID}

echo -e "\n7. Verifying gateway deployment:"
GATEWAY_URL=$(gcloud api-gateway gateways describe ${GATEWAY_NAME} \
  --location=${REGION} \
  --format="value(defaultHostname)" \
  --project=${PROJECT_ID})

echo "Gateway URL: https://$GATEWAY_URL"

# Wait for deployment to complete
echo -e "\n8. Waiting for deployment to propagate (60 seconds)..."
sleep 60

# Test endpoints again
echo -e "\n9. Testing API Gateway endpoints:"
ENDPOINTS=(
  "/"
  "/api"
  "/api/topics"
  "/api/topics/stats"
)

for endpoint in "${ENDPOINTS[@]}"; do
  echo -n "Testing https://$GATEWAY_URL$endpoint: "
  RESPONSE=$(curl -s -w "\nStatus code: %{http_code}\n" "https://$GATEWAY_URL$endpoint")
  echo "$RESPONSE"
  echo "---------------------------------"
done

echo -e "\n===== DIAGNOSIS RESULTS =====:"
echo "Based on the Cloud Run logs, your backend service is experiencing crashes."
echo "The Gunicorn workers are exiting with SystemExit: 1, which indicates an application error."
echo "This is why both the API Gateway and direct access to the Cloud Run service are returning 404 errors."

echo -e "\nRECOMMENDED ACTIONS:"
echo "1. Fix the backend service issues - check application code and logs for errors"
echo "2. Modify the Gunicorn settings in your Dockerfile and cloudrun-service.yaml:"
echo "   - Consider reverting to the standard worker class instead of UvicornWorker if your app isn't async-compatible"
echo "   - Adjust timeouts and other parameters for better stability"
echo "3. After fixing the backend service, redeploy and then check the API Gateway again" 