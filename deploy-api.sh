#!/bin/bash

# Configuration
PROJECT_ID="newsletter-450510"
API_ID="newsletter-aggregator-api"
GATEWAY_ID="newsletter-aggregator-gateway"
LOCATION="us-central1"
CONFIG_PREFIX="newsletter-aggregator-config-v"
KEEP_VERSIONS=3  # Number of recent versions to keep

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# API Configuration
BASE_URL="https://newsletter-aggregator-gateway-axs105xr.uc.gateway.dev"
API_KEY="AIzaSyB057kXlCQSfbKmw8Pinuu4JKxjaRave4k"
ORIGINS=(
  "https://newsletter-aggregator-knap.vercel.app"
  "https://newsletter-aggregator.vercel.app"
  "http://localhost:3000"
)

# Critical endpoints to test
declare -A ENDPOINTS=(
  ["GET:/api/update/status"]="Update Status Endpoint"
  ["GET:/api/articles"]="Articles Endpoint"
  ["GET:/api/topics"]="Topics Endpoint"
  ["POST:/api/rag"]="RAG Query Endpoint"
)

# Function to test an endpoint
test_endpoint() {
  local method=$1
  local endpoint=$2
  local origin=$3
  local description=$4
  local expected_status=$5

  echo -e "${BLUE}Testing $description ($method $endpoint)...${NC}"
  
  if [ "$method" = "OPTIONS" ]; then
    response=$(curl -s -o /dev/null -w "%{http_code}" \
      -X OPTIONS \
      "${BASE_URL}${endpoint}?key=${API_KEY}" \
      -H "Origin: $origin" \
      -H "Access-Control-Request-Method: ${method}")
  else
    response=$(curl -s -o /dev/null -w "%{http_code}" \
      -X $method \
      "${BASE_URL}${endpoint}?key=${API_KEY}" \
      -H "Origin: $origin")
  fi

  if [ "$response" = "$expected_status" ]; then
    echo -e "${GREEN}✓ $description: $method returned $response${NC}"
    return 0
  else
    echo -e "${RED}✗ $description: $method returned $response (expected $expected_status)${NC}"
    return 1
  fi
}

# Function to run all tests
run_tests() {
  local failed_tests=0
  
  # Test each endpoint with each origin
  for origin in "${ORIGINS[@]}"; do
    echo -e "\n${YELLOW}Testing with origin: $origin${NC}"
    
    for endpoint_key in "${!ENDPOINTS[@]}"; do
      IFS=':' read -r method path <<< "$endpoint_key"
      description="${ENDPOINTS[$endpoint_key]}"
      
      # Test the actual endpoint
      test_endpoint "$method" "$path" "$origin" "$description" "200" || ((failed_tests++))
      
      # Test OPTIONS preflight
      test_endpoint "OPTIONS" "$path" "$origin" "$description (CORS Preflight)" "204" || ((failed_tests++))
    done
  done
  
  return $failed_tests
}

# Function to rollback deployment
rollback_deployment() {
  local previous_version=$1
  echo -e "${YELLOW}Rolling back to version $previous_version...${NC}"
  
  if gcloud api-gateway gateways update $GATEWAY_ID \
    --api=$API_ID \
    --api-config=$previous_version \
    --location=$LOCATION \
    --project=$PROJECT_ID; then
    echo -e "${GREEN}Successfully rolled back to $previous_version${NC}"
    return 0
  else
    echo -e "${RED}Failed to rollback to $previous_version${NC}"
    return 1
  fi
}

# Main deployment process
echo -e "${YELLOW}Starting API Gateway deployment...${NC}"

# Get the latest version number
echo "Finding latest config version..."
LATEST_VERSION=$(gcloud api-gateway api-configs list \
  --api=$API_ID \
  --project=$PROJECT_ID \
  --format="value(CONFIG_ID)" | \
  grep -o '[0-9]*$' | sort -n | tail -1)

# Store current config for potential rollback
CURRENT_CONFIG="${CONFIG_PREFIX}${LATEST_VERSION}"

# Increment version
NEW_VERSION=$((LATEST_VERSION + 1))
NEW_CONFIG_ID="${CONFIG_PREFIX}${NEW_VERSION}"

echo -e "${YELLOW}Creating new config version: ${NEW_CONFIG_ID}${NC}"

# Create new API config
if ! gcloud api-gateway api-configs create $NEW_CONFIG_ID \
  --api=$API_ID \
  --openapi-spec=openapi.yaml \
  --project=$PROJECT_ID; then
  echo -e "${RED}Failed to create API config${NC}"
  exit 1
fi

echo -e "${GREEN}Successfully created new API config${NC}"

# Update gateway with new config
echo -e "${YELLOW}Updating gateway with new config...${NC}"
if ! gcloud api-gateway gateways update $GATEWAY_ID \
  --api=$API_ID \
  --api-config=$NEW_CONFIG_ID \
  --location=$LOCATION \
  --project=$PROJECT_ID; then
  echo -e "${RED}Failed to update gateway${NC}"
  exit 1
fi

echo -e "${GREEN}Successfully updated gateway${NC}"

# Wait for changes to propagate
echo -e "${YELLOW}Waiting for changes to propagate (30 seconds)...${NC}"
sleep 30

# Run tests
echo -e "${YELLOW}Running tests...${NC}"
if ! run_tests; then
  echo -e "${RED}Tests failed! Rolling back deployment...${NC}"
  if rollback_deployment $CURRENT_CONFIG; then
    echo -e "${YELLOW}Rollback successful. Please fix the issues and try again.${NC}"
    exit 1
  else
    echo -e "${RED}Rollback failed! Manual intervention required!${NC}"
    exit 2
  fi
fi

# Clean up old configs
echo -e "${YELLOW}Cleaning up old configurations...${NC}"
CONFIGS_TO_DELETE=$(gcloud api-gateway api-configs list \
  --api=$API_ID \
  --project=$PROJECT_ID \
  --format="value(CONFIG_ID)" | \
  sort -V | head -n -$KEEP_VERSIONS)

if [ ! -z "$CONFIGS_TO_DELETE" ]; then
  echo "Deleting old configurations:"
  for CONFIG in $CONFIGS_TO_DELETE; do
    echo "Deleting $CONFIG..."
    gcloud api-gateway api-configs delete $CONFIG \
      --api=$API_ID \
      --project=$PROJECT_ID --quiet
  done
else
  echo "No old configurations to delete"
fi

echo -e "${GREEN}Deployment complete!${NC}"
echo -e "${BLUE}API Gateway URL: ${BASE_URL}${NC}"
echo -e "${BLUE}Configuration version: ${NEW_CONFIG_ID}${NC}" 