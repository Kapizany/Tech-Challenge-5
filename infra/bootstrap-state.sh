#!/usr/bin/env bash
set -euo pipefail

: "${GCP_PROJECT_ID:?Set GCP_PROJECT_ID}"
: "${GCP_REGION:=southamerica-east1}"
: "${TF_STATE_BUCKET:?Set a globally unique TF_STATE_BUCKET name}"

gcloud config set project "${GCP_PROJECT_ID}"
gcloud services enable serviceusage.googleapis.com --project "${GCP_PROJECT_ID}"
gcloud services enable pubsub.googleapis.com --project "${GCP_PROJECT_ID}"
gcloud beta services identity create --service=pubsub.googleapis.com --project="${GCP_PROJECT_ID}"
gcloud storage buckets create "gs://${TF_STATE_BUCKET}" \
  --project="${GCP_PROJECT_ID}" \
  --location="${GCP_REGION}" \
  --uniform-bucket-level-access \
  --public-access-prevention
gcloud storage buckets update "gs://${TF_STATE_BUCKET}" --versioning
echo "Terraform remote state bucket ready: gs://${TF_STATE_BUCKET}"
