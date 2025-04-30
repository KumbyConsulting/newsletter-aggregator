#!/bin/bash
# =========================================================
# Newsletter Aggregator Deployment Script
# ---------------------------------------------------------
# This script automates the deployment of the Newsletter Aggregator
# application to Google Cloud Run using Cloud Build.
# =========================================================

# Exit on any error
set -e

# Display help message
function show_help {
  echo "Newsletter Aggregator Deployment Script"
  echo "---------------------------------------"
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  -h, --help        Show this help message"
  echo "  -p, --project     Set Google Cloud project ID (required)"
  echo "  -r, --region      Set Google Cloud region (default: us-central1)"
  echo "  -e, --env         Path to environment file (default: env.yaml)"
  echo
  echo "Example:"
  echo "  $0 --project newsletter-450510 --region us-central1"
  echo
}

# Default values
PROJECT_ID=""
REGION="us-central1"
ENV_FILE="env.yaml"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -h|--help)
      show_help
      exit 0
      ;;
    -p|--project)
      PROJECT_ID="$2"
      shift
      shift
      ;;
    -r|--region)
      REGION="$2"
      shift
      shift
      ;;
    -e|--env)
      ENV_FILE="$2"
      shift
      shift
      ;;
    *)
      echo "Unknown option: $1"
      show_help
      exit 1
      ;;
  esac
done

# Check if PROJECT_ID is set
if [ -z "$PROJECT_ID" ]; then
  echo "Error: PROJECT_ID is required"
  echo "Try getting your project ID with: gcloud config get-value project"
  show_help
  exit 1
fi

# Check if env file exists
if [ ! -f "$ENV_FILE" ]; then
  echo "Warning: Environment file $ENV_FILE not found. Creating a default one."
  cat > $ENV_FILE << EOF
# Environment variables for the newsletter aggregator
# This file is used by the deployment script to set environment variables

# Main application settings
# PORT is automatically set by Cloud Run, do not include it here
PROMETHEUS_PORT: "8000"

# Add your application-specific environment variables below
# DATABASE_URL: ""
# API_KEY: "" 
# DEBUG: "False"
EOF
fi

# Check if required configuration files exist
if [ ! -f "cloudbuild.yaml" ]; then
  echo "Error: cloudbuild.yaml not found."
  exit 1
fi

if [ ! -f "service.yaml" ]; then
  echo "Error: service.yaml not found."
  exit 1
fi

echo "======================================================"
echo "         Newsletter Aggregator Deployment             "
echo "======================================================"
echo "Project ID: $PROJECT_ID"
echo "Region: $REGION"
echo "Environment file: $ENV_FILE"
echo "------------------------------------------------------"

# Configure gcloud with PROJECT_ID
echo "[1/4] Setting project to $PROJECT_ID..."
gcloud config set project "$PROJECT_ID" || { echo "Failed to set project"; exit 1; }

# Enable required APIs if not already enabled
echo "[2/4] Ensuring required APIs are enabled..."
gcloud services enable cloudbuild.googleapis.com run.googleapis.com artifactregistry.googleapis.com containerregistry.googleapis.com || { echo "Failed to enable APIs"; exit 1; }

# Submit the build to Cloud Build
echo "[3/4] Submitting build to Cloud Build..."
gcloud builds submit --config=cloudbuild.yaml --substitutions=_REGION="$REGION" || { echo "Build submission failed"; exit 1; }

# Show deployment info
echo "[4/4] Deployment complete!"
echo "------------------------------------------------------"
echo "To view your application, run:"
echo "gcloud run services describe newsletter-aggregator --region=$REGION --format='value(status.url)'"
echo "======================================================" 