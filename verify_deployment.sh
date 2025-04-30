#!/bin/bash
set -e

echo "======= VERIFYING BACKEND DEPLOYMENT ======="
REGION="us-central1"
PROJECT_ID=$(gcloud config get-value project)
SERVICE_NAME="newsletter-aggregator"

# Helper function to print colored status
print_status() {
  if [ "$2" == "200" ]; then
    echo -e "$1: \033[32mOK (200)\033[0m"
  else
    echo -e "$1: \033[31mFailed ($2)\033[0m"
  fi
}

# 1. Check service status
echo -e "\n1. Checking Cloud Run service status:"
gcloud run services describe $SERVICE_NAME --region=$REGION --format="yaml(status)" | grep -E "url|Ready"

# 2. Get current Cloud Run service URL
BACKEND_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format="get(status.url)")
echo -e "\nBackend URL: $BACKEND_URL"

# 3. Test basic endpoints
echo -e "\n2. Testing basic endpoints:"

ENDPOINTS=(
  "/"
  "/api"
  "/api/topics"
  "/api/topics/stats"
  "/api/health"
  "/api/status"
)

for endpoint in "${ENDPOINTS[@]}"; do
  STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$BACKEND_URL$endpoint")
  print_status "$BACKEND_URL$endpoint" "$STATUS"
done

# 4. Get logs from the Cloud Run service for the last 10 minutes
echo -e "\n3. Recent logs from Cloud Run service (last 10 minutes):"
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=$SERVICE_NAME AND timestamp>=-PT10M" --limit 10 --format="table(timestamp, severity, textPayload)" --order="desc"

# 5. Check API Gateway
echo -e "\n4. Checking API Gateway configuration:"
GATEWAY_NAME="newsletter-aggregator-gateway"
GATEWAY_URL=$(gcloud api-gateway gateways describe $GATEWAY_NAME --location=$REGION --format="value(defaultHostname)")
API_CONFIG=$(gcloud api-gateway gateways describe $GATEWAY_NAME --location=$REGION --format="value(apiConfig)")

echo "Gateway URL: https://$GATEWAY_URL"
echo "Current API Config: $API_CONFIG"

# 6. Test Gateway Endpoints
echo -e "\n5. Testing API Gateway endpoints:"
for endpoint in "${ENDPOINTS[@]}"; do
  STATUS=$(curl -s -o /dev/null -w "%{http_code}" "https://$GATEWAY_URL$endpoint")
  print_status "https://$GATEWAY_URL$endpoint" "$STATUS"
done

echo -e "\n======== DEPLOYMENT VERIFICATION COMPLETE ========"
echo "If endpoints are still failing, check the following:"
echo "1. Look for more detailed error messages in the logs"
echo "2. Check that your Flask app is correctly configured to work with Waitress"
echo "3. Verify that the app.py:app callable is correctly exposed"
echo "4. Consider checking if any Flask extensions need special configuration with Waitress" 