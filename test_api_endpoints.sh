#!/bin/bash
set -e

# You may need to set BACKEND_URL manually if not using gcloud
BACKEND_URL=$(gcloud run services describe newsletter-aggregator --region=us-central1 --format="get(status.url)")
GATEWAY_URL="https://newsletter-aggregator-gateway-axs105xr.uc.gateway.dev"

echo "Testing various API endpoints to find a working one..."

# Define an array of valid GET endpoints to test
ENDPOINTS=(
  "/"
  "/_ah/health"
  "/_ah/warmup"
  "/api/topics"
  "/api/topics/stats"
  "/api/articles"
  "/api/status"
  "/api/update/status"
  "/api/backups"
  "/api/sources"
  "/similar-articles/"
  "/rag"
  "/test_summary"
)

echo -e "\n=== Testing Backend Service Endpoints ==="
for endpoint in "${ENDPOINTS[@]}"; do
  echo -n "Testing $BACKEND_URL$endpoint: "
  STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$BACKEND_URL$endpoint")
  if [ "$STATUS" = "200" ]; then
    echo -e "\033[32mOK (200)\033[0m"
  else
    echo -e "\033[31mFailed ($STATUS)\033[0m"
  fi
done

echo -e "\n=== Testing API Gateway Endpoints ==="
for endpoint in "${ENDPOINTS[@]}"; do
  echo -n "Testing $GATEWAY_URL$endpoint: "
  STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$GATEWAY_URL$endpoint")
  if [ "$STATUS" = "200" ]; then
    echo -e "\033[32mOK (200)\033[0m"
  else
    echo -e "\033[31mFailed ($STATUS)\033[0m"
  fi
done 