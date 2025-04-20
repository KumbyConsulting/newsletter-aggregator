#!/bin/bash
set -e

# Get the project ID
PROJECT_ID=$(gcloud config get-value project)

# Enable required APIs
echo "Enabling required APIs..."
gcloud services enable apigateway.googleapis.com --project=$PROJECT_ID
gcloud services enable servicemanagement.googleapis.com --project=$PROJECT_ID
gcloud services enable servicecontrol.googleapis.com --project=$PROJECT_ID

echo "APIs enabled successfully!" 