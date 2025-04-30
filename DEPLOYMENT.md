# Newsletter Aggregator Deployment Guide

This document provides instructions for deploying the Newsletter Aggregator application to Google Cloud Run.

## Prerequisites

- Google Cloud SDK installed and configured
- Docker installed locally (for testing)
- A Google Cloud Project with billing enabled
- Required permissions to deploy to Cloud Run and use Cloud Build

## Deployment Files

The deployment process uses the following files:

- `Dockerfile` - Defines the container image using Python 3.11 and Waitress
- `cloudbuild.yaml` - Configuration for Cloud Build
- `service.yaml` - Configuration for Cloud Run service
- `env.yaml` - Environment variables for the application
- `deploy.sh` - Deployment script that orchestrates the process

## Deployment Steps

### 1. Set up your environment

Make sure you have the Google Cloud SDK installed and configured.

```bash
# Check gcloud installation
gcloud --version

# Login to your Google Cloud account if needed
gcloud auth login
```

### 2. Configure environment variables

Edit the `env.yaml` file to set your application's environment variables.

> **Important**: Do not set reserved environment variables in `env.yaml`. Cloud Run automatically sets these variables:
> - `PORT` - The port your container should listen on
> - `K_SERVICE` - The name of the Cloud Run service being deployed
> - `K_REVISION` - The revision of the Cloud Run service being deployed
> - `K_CONFIGURATION` - The configuration of the Cloud Run service

### 3. Run the deployment script

```bash
# Deploy with the script
./deploy.sh --project YOUR_PROJECT_ID --region us-central1
```

Replace `YOUR_PROJECT_ID` with your actual Google Cloud project ID.

### 4. Access your application

After deployment, the script will output a command to get the URL of your deployed application.

## Manual Deployment

If you prefer to deploy manually without the script:

```bash
# Set your project
export PROJECT_ID=$(gcloud config get-value project)
export REGION=us-central1

# Submit build
gcloud builds submit --config=cloudbuild.yaml --substitutions=_REGION="$REGION"
```

## Troubleshooting

- **Build failure**: Check the build logs in Cloud Build console
- **Deployment failure**: Verify your service.yaml configuration
- **Runtime errors**: Check Cloud Run logs

## Customization

- **Scaling**: Modify `autoscaling.knative.dev/maxScale` and `autoscaling.knative.dev/minScale` in `service.yaml`
- **Resources**: Adjust CPU and memory in `service.yaml`
- **Environment**: Add or modify environment variables in `env.yaml` 