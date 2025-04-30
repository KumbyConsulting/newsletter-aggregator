#!/bin/bash
set -e

echo "======= DEBUGGING API GATEWAY CONNECTION ======="
PROJECT_ID=$(gcloud config get-value project)
REGION="us-central1"
API_NAME="newsletter-aggregator-api"
GATEWAY_NAME="newsletter-aggregator-gateway"

# 1. Check gateway status
echo -e "\n\033[1m1. API Gateway Status:\033[0m"
GATEWAY_DETAILS=$(gcloud api-gateway gateways describe $GATEWAY_NAME --location=$REGION --format=json)
echo "$GATEWAY_DETAILS" | jq .

# Get the hostname and current API config
GATEWAY_HOSTNAME=$(echo "$GATEWAY_DETAILS" | jq -r .defaultHostname)
CURRENT_CONFIG=$(echo "$GATEWAY_DETAILS" | jq -r .apiConfig | sed 's/.*configs\///')
echo -e "\nGateway hostname: https://$GATEWAY_HOSTNAME"
echo -e "Current config: $CURRENT_CONFIG"

# 2. Check current API config
echo -e "\n\033[1m2. Current API Config Details:\033[0m"
CONFIG_DETAILS=$(gcloud api-gateway api-configs describe $CURRENT_CONFIG --api=$API_NAME --format=json)
echo "$CONFIG_DETAILS" | jq .

# 3. Test API Endpoint
echo -e "\n\033[1m3. Testing API Endpoint:\033[0m"
echo -e "Testing: https://$GATEWAY_HOSTNAME/api/topics/stats"
curl -s -w "\nStatus code: %{http_code}\n" https://$GATEWAY_HOSTNAME/api/topics/stats

# 4. Check directly against backend service
echo -e "\n\033[1m4. Testing Backend Service Directly:\033[0m"
BACKEND_URL=$(gcloud run services describe newsletter-aggregator --region=$REGION --format="get(status.url)")
echo -e "Backend URL: $BACKEND_URL"
echo -e "Testing: $BACKEND_URL/api/topics/stats"
curl -s -w "\nStatus code: %{http_code}\n" $BACKEND_URL/api/topics/stats

# 5. Show deployment hook status
echo -e "\n\033[1m5. Latest API Config Deployments:\033[0m"
gcloud api-gateway api-configs list --api=$API_NAME --sort-by=~createTime --limit=3

echo -e "\n\033[1m6. Check API Gateway Service Status:\033[0m"
gcloud services list --enabled --filter=apigateway.googleapis.com 