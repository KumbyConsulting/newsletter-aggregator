#!/bin/bash

# Load environment variables
set -a
source .env
set +a

# Set up Google Cloud SDK environment variables
export CLOUDSDK_ROOT_DIR="$PWD/google-cloud-sdk"
export PATH="$CLOUDSDK_ROOT_DIR/bin:$PATH"
export CLOUDSDK_PYTHON=python3
export CLOUDSDK_METRICS_ENVIRONMENT=newsletter-aggregator-dev
export CLOUDSDK_CORE_DISABLE_PROMPTS=1
export CLOUDSDK_AUTH_BROWSER=0

# Initialize gcloud if needed (without browser)
if [ ! -f "$CLOUDSDK_ROOT_DIR/bin/gcloud" ]; then
    echo "Google Cloud SDK not found in $CLOUDSDK_ROOT_DIR"
    echo "Please install it first using the installation instructions"
    exit 1
fi

# Verify installation
gcloud --version > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "Google Cloud SDK is properly configured"
else
    echo "Error: Google Cloud SDK is not properly configured"
    exit 1
fi 