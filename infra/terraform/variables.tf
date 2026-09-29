variable "project_id" {
  description = "GCP project ID for this environment."
  type        = string
}

variable "region" {
  description = "Region for Cloud Run, Storage, BigQuery and Artifact Registry."
  type        = string
  default     = "southamerica-east1"
}

variable "environment" {
  type    = string
  default = "dev"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "environment must be dev, staging or prod."
  }
}

variable "github_owner" {
  description = "GitHub organization or user allowed to authenticate with WIF."
  type        = string
}

variable "github_repository" {
  description = "GitHub repository name allowed to authenticate with WIF."
  type        = string
}

variable "data_bucket" {
  type    = string
  default = ""
}

variable "state_bucket_name" {
  description = "Bootstrap-created GCS bucket used by the remote Terraform backend."
  type        = string
  default     = ""
}

variable "container_image" {
  description = "Immutable Artifact Registry image digest for the API."
  type        = string
}

variable "model_uri" {
  description = "Versioned gs:// URI for channel_reward_model.joblib."
  type        = string
}

variable "git_sha" {
  description = "Commit SHA of the training image used by the scheduled consolidation job."
  type        = string
  default     = ""
}

variable "firestore_location" {
  description = "Firestore database location; confirm regional availability before apply."
  type        = string
  default     = "southamerica-east1"
}

variable "cloud_run_max_instances" {
  type    = number
  default = 5
}

variable "terraform_apply_service_account_roles" {
  description = "Project roles for the protected GitHub Terraform apply identity."
  type        = set(string)
  default = [
    "roles/artifactregistry.admin",
    "roles/bigquery.admin",
    "roles/datastore.owner",
    "roles/iam.serviceAccountAdmin",
    "roles/iam.serviceAccountUser",
    "roles/iam.workloadIdentityPoolAdmin",
    "roles/pubsub.admin",
    "roles/resourcemanager.projectIamAdmin",
    "roles/cloudscheduler.admin",
    "roles/logging.viewer",
    "roles/monitoring.editor",
    "roles/run.admin",
    "roles/serviceusage.serviceUsageAdmin",
    "roles/storage.admin",
  ]
}

variable "consolidation_schedule" {
  description = "Cloud Scheduler cron for the policy consolidation job. Weekly by default."
  type        = string
  default     = "0 9 * * 1"
}

variable "consolidation_paused" {
  description = "Keep the bootstrap schedule paused until a validated image is deployed."
  type        = bool
  default     = true
}

variable "labels" {
  type = map(string)
  default = {
    app         = "adaptive-offers"
    owner       = "fiap-datathon"
    cost_center = "academic"
    managed_by  = "terraform"
  }
}
