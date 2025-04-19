# Google Cloud CLI Cheat Sheet for Newsletter Aggregator

This quick reference guide provides common Google Cloud CLI commands for deploying and managing the Newsletter Aggregator application.

## Authentication and Project Setup

```bash
# Log in to Google Cloud
gcloud auth login

# Set the active project
gcloud config set project PROJECT_ID

# List configured properties
gcloud config list

# View all available configurations
gcloud config configurations list
```

## Cloud Run Commands

```bash
# List deployed services
gcloud run services list

# Get details about a service
gcloud run services describe newsletter-aggregator --region=us-central1

# View logs for a service
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=newsletter-aggregator" --limit=10

# Update an existing service
gcloud run services update newsletter-aggregator --memory=2Gi --region=us-central1

# Delete a service
gcloud run services delete newsletter-aggregator --region=us-central1
```

## Container Registry Commands

```bash
# List container images
gcloud container images list --repository=gcr.io/PROJECT_ID

# List tags for a specific image
gcloud container images list-tags gcr.io/PROJECT_ID/newsletter-aggregator

# Delete an image
gcloud container images delete gcr.io/PROJECT_ID/newsletter-aggregator:TAG --force-delete-tags
```

## Secret Manager Commands

```bash
# Create a new secret
echo -n "secret-value" | gcloud secrets create SECRET_NAME --data-file=-

# List available secrets
gcloud secrets list

# Access a secret version
gcloud secrets versions access latest --secret=GEMINI_API_KEY

# Update an existing secret
echo -n "new-secret-value" | gcloud secrets versions add GEMINI_API_KEY --data-file=-
```

## Deployment Commands

```bash
# Backend deployment with Cloud Build
cd /path/to/newsletter-aggregator
gcloud builds submit --config cloudbuild.yaml

# Frontend deployment with Cloud Build
cd /path/to/newsletter-aggregator/frontend
gcloud builds submit --config cloudbuild.yaml

# Manual backend deployment
gcloud run deploy newsletter-aggregator \
  --image=gcr.io/PROJECT_ID/newsletter-aggregator:v1 \
  --region=us-central1 \
  --platform=managed \
  --allow-unauthenticated

# Manual frontend deployment
gcloud run deploy newsletter-frontend \
  --image=gcr.io/PROJECT_ID/newsletter-frontend:v1 \
  --region=us-central1 \
  --platform=managed \
  --allow-unauthenticated
```

## Monitoring and Troubleshooting

```bash
# View recent errors
gcloud logging read "severity>=ERROR" --project=PROJECT_ID --limit=10

# Check service health
curl https://newsletter-aggregator-URL/api/health

# Get resource utilization
gcloud monitoring metrics list | grep cloud_run
```

## Custom Domain Configuration

```bash
# Map a custom domain to a service
gcloud run domain-mappings create --service=newsletter-frontend --domain=yourdomain.com --region=us-central1

# List domain mappings
gcloud run domain-mappings list --region=us-central1
```

## Firestore Commands

```bash
# Initialize Firestore
gcloud firestore databases create --region=us-central1

# Delete all documents in a collection (caution!)
gcloud firestore delete-documents --collection-id=articles --all-documents
```

## Cloud Storage Commands

```bash
# Create a bucket
gcloud storage buckets create gs://newsletter-aggregator

# List buckets
gcloud storage buckets list

# Upload a file
gcloud storage cp local-file.json gs://newsletter-aggregator/

# Download a file
gcloud storage cp gs://newsletter-aggregator/backup.json ./
```

## IAM and Permissions

```bash
# List service accounts
gcloud iam service-accounts list

# Create a service account
gcloud iam service-accounts create newsletter-sa --description="Service account for newsletter app"

# Add a role to a service account
gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="serviceAccount:newsletter-sa@PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/run.invoker"
```

Remember to replace `PROJECT_ID` with your actual Google Cloud project ID wherever applicable. 