locals {
  name             = "adaptive-offers-${var.environment}"
  data_bucket_name = var.data_bucket != "" ? var.data_bucket : "${var.project_id}-${local.name}-data"
  apis = toset([
    "artifactregistry.googleapis.com",
    "bigquery.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "firestore.googleapis.com",
    "iam.googleapis.com",
    "iamcredentials.googleapis.com",
    "logging.googleapis.com",
    "monitoring.googleapis.com",
    "cloudscheduler.googleapis.com",
    "pubsub.googleapis.com",
    "run.googleapis.com",
    "secretmanager.googleapis.com",
    "serviceusage.googleapis.com",
    "sts.googleapis.com",
    "storage.googleapis.com",
  ])
  pubsub_service_agent_email = "service-${data.google_project.current.number}@gcp-sa-pubsub.iam.gserviceaccount.com"
}

data "google_project" "current" {
  project_id = var.project_id
}

resource "google_project_service" "enabled" {
  for_each                   = local.apis
  project                    = var.project_id
  service                    = each.value
  disable_on_destroy         = false
  disable_dependent_services = false
}

resource "google_storage_bucket" "data" {
  name                        = local.data_bucket_name
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  force_destroy               = false
  labels                      = var.labels

  versioning { enabled = true }
  lifecycle_rule {
    condition { age = 365 }
    action { type = "Delete" }
  }
  depends_on = [google_project_service.enabled]
}

resource "google_bigquery_dataset" "analytics" {
  dataset_id                  = "adaptive_offers_${var.environment}"
  friendly_name               = "Adaptive Offers ${var.environment}"
  description                 = "Versioned decision and feedback analytics."
  location                    = var.region
  default_table_expiration_ms = 31536000000
  delete_contents_on_destroy  = false
  labels                      = var.labels
  depends_on                  = [google_project_service.enabled]
}

resource "google_bigquery_table" "feedback" {
  dataset_id = google_bigquery_dataset.analytics.dataset_id
  table_id   = "feedback_events"
  schema = jsonencode([
    { name = "request_id", type = "STRING", mode = "REQUIRED" },
    { name = "reward", type = "INTEGER", mode = "REQUIRED" },
    { name = "timestamp", type = "TIMESTAMP", mode = "REQUIRED" },
    { name = "received_at", type = "TIMESTAMP", mode = "REQUIRED" },
  ])
  time_partitioning {
    type  = "DAY"
    field = "received_at"
  }
  deletion_protection = true
}

resource "google_artifact_registry_repository" "containers" {
  location      = var.region
  repository_id = "adaptive-offers-${var.environment}"
  description   = "Immutable API images for Adaptive Offers."
  format        = "DOCKER"
  docker_config { immutable_tags = true }
  labels     = var.labels
  depends_on = [google_project_service.enabled]
}

resource "google_firestore_database" "default" {
  project                 = var.project_id
  name                    = "(default)"
  location_id             = var.firestore_location
  type                    = "FIRESTORE_NATIVE"
  concurrency_mode        = "PESSIMISTIC"
  delete_protection_state = "DELETE_PROTECTION_ENABLED"
  deletion_policy         = "ABANDON"
  depends_on              = [google_project_service.enabled]
}

resource "google_pubsub_topic" "feedback" {
  name                       = "${local.name}-feedback"
  message_retention_duration = "604800s"
  labels                     = var.labels
  depends_on                 = [google_project_service.enabled]
}

resource "google_pubsub_topic" "feedback_dead_letter" {
  name   = "${local.name}-feedback-dead-letter"
  labels = var.labels
}

resource "google_pubsub_subscription" "feedback" {
  name                       = "${local.name}-feedback-worker"
  topic                      = google_pubsub_topic.feedback.id
  ack_deadline_seconds       = 30
  message_retention_duration = "604800s"

  dead_letter_policy {
    dead_letter_topic     = google_pubsub_topic.feedback_dead_letter.id
    max_delivery_attempts = 5
  }
  push_config {
    push_endpoint = "${google_cloud_run_v2_service.api.uri}/events/feedback"
    oidc_token {
      service_account_email = google_service_account.pubsub_push.email
      audience              = google_cloud_run_v2_service.api.uri
    }
  }
  expiration_policy { ttl = "" }
  labels = var.labels
}

resource "google_service_account" "pubsub_push" {
  account_id   = "ao-${var.environment}-pubsub"
  display_name = "Pub/Sub push identity for the feedback receiver"
  depends_on   = [google_project_service.enabled]
}

resource "google_service_account_iam_member" "pubsub_mint_push_token" {
  service_account_id = google_service_account.pubsub_push.name
  role               = "roles/iam.serviceAccountTokenCreator"
  member             = "serviceAccount:${local.pubsub_service_agent_email}"
}

resource "google_service_account" "runtime" {
  account_id   = "ao-${var.environment}-runtime"
  display_name = "Adaptive Offers Cloud Run runtime"
  depends_on   = [google_project_service.enabled]
}

resource "google_project_iam_member" "runtime_roles" {
  for_each = toset([
    "roles/datastore.user",
  ])
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.runtime.email}"
}

resource "google_bigquery_dataset_iam_member" "runtime_feedback_writer" {
  dataset_id = google_bigquery_dataset.analytics.dataset_id
  role       = "roles/bigquery.dataEditor"
  member     = "serviceAccount:${google_service_account.runtime.email}"
}

resource "google_pubsub_topic_iam_member" "runtime_feedback_publisher" {
  topic  = google_pubsub_topic.feedback.name
  role   = "roles/pubsub.publisher"
  member = "serviceAccount:${google_service_account.runtime.email}"
}

resource "google_storage_bucket_iam_member" "model_reader" {
  bucket = google_storage_bucket.data.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.runtime.email}"
}

resource "google_service_account" "github_deploy" {
  account_id   = "ao-${var.environment}-deploy"
  display_name = "GitHub Actions deploy identity"
  depends_on   = [google_project_service.enabled]
}

resource "google_artifact_registry_repository_iam_member" "github_deploy_writer" {
  project    = var.project_id
  location   = google_artifact_registry_repository.containers.location
  repository = google_artifact_registry_repository.containers.repository_id
  role       = "roles/artifactregistry.writer"
  member     = "serviceAccount:${google_service_account.github_deploy.email}"
}

resource "google_storage_bucket_iam_member" "github_model_writer" {
  bucket = google_storage_bucket.data.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.github_deploy.email}"
}

resource "google_service_account" "github_terraform" {
  account_id   = "ao-${var.environment}-terraform"
  display_name = "Protected GitHub Actions Terraform identity"
  depends_on   = [google_project_service.enabled]
}

resource "google_project_iam_member" "github_terraform_roles" {
  for_each = var.terraform_apply_service_account_roles
  project  = var.project_id
  role     = each.value
  member   = "serviceAccount:${google_service_account.github_terraform.email}"
}

resource "google_service_account" "github_plan" {
  account_id   = "ao-${var.environment}-plan"
  display_name = "Read-only GitHub Actions Terraform plan identity"
  depends_on   = [google_project_service.enabled]
}

resource "google_storage_bucket_iam_member" "github_plan_state_lock" {
  count  = var.state_bucket_name == "" ? 0 : 1
  bucket = var.state_bucket_name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.github_plan.email}"
}

resource "google_project_iam_member" "github_plan_roles" {
  for_each = toset([
    "roles/artifactregistry.reader",
    "roles/bigquery.metadataViewer",
    "roles/datastore.viewer",
    "roles/iam.securityReviewer",
    "roles/logging.viewer",
    "roles/monitoring.viewer",
    "roles/pubsub.viewer",
    "roles/run.viewer",
    "roles/serviceusage.serviceUsageViewer",
    "roles/viewer",
  ])
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.github_plan.email}"
}

resource "google_iam_workload_identity_pool" "github" {
  workload_identity_pool_id = "github-${var.environment}"
  display_name              = "GitHub Actions ${var.environment}"
  description               = "Keyless CI identity for ${var.github_owner}/${var.github_repository}."
  depends_on                = [google_project_service.enabled]
}

resource "google_iam_workload_identity_pool_provider" "github" {
  workload_identity_pool_id          = google_iam_workload_identity_pool.github.workload_identity_pool_id
  workload_identity_pool_provider_id = "github-oidc"
  display_name                       = "GitHub Actions OIDC"
  attribute_mapping = {
    "google.subject"       = "assertion.sub"
    "attribute.repository" = "assertion.repository"
    "attribute.ref"        = "assertion.ref"
  }
  # Do not mint GCP credentials from pull-request or feature-branch workflows.
  # Terraform plans with cloud credentials are dispatched from protected main.
  attribute_condition = "assertion.repository_owner == '${var.github_owner}' && assertion.repository == '${var.github_owner}/${var.github_repository}' && assertion.ref == 'refs/heads/main'"
  oidc { issuer_uri = "https://token.actions.githubusercontent.com" }
}

resource "google_service_account_iam_member" "github_wif_deploy" {
  service_account_id = google_service_account.github_deploy.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github.name}/attribute.repository/${var.github_owner}/${var.github_repository}"
}

resource "google_service_account_iam_member" "github_wif_terraform" {
  service_account_id = google_service_account.github_terraform.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github.name}/attribute.repository/${var.github_owner}/${var.github_repository}"
}

resource "google_service_account_iam_member" "github_terraform_self_token_creator" {
  service_account_id = google_service_account.github_terraform.name
  role               = "roles/iam.serviceAccountTokenCreator"
  member             = "serviceAccount:${google_service_account.github_terraform.email}"
}

resource "google_service_account_iam_member" "github_wif_plan" {
  service_account_id = google_service_account.github_plan.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github.name}/attribute.repository/${var.github_owner}/${var.github_repository}"
}

resource "google_cloud_run_v2_service" "api" {
  name                = local.name
  location            = var.region
  ingress             = "INGRESS_TRAFFIC_ALL"
  deletion_protection = var.environment == "prod"
  labels              = var.labels

  template {
    service_account                  = google_service_account.runtime.email
    max_instance_request_concurrency = 40
    scaling {
      min_instance_count = 0
      max_instance_count = var.cloud_run_max_instances
    }
    containers {
      image = var.container_image
      ports { container_port = 8080 }
      resources { limits = { cpu = "1", memory = "1Gi" } }
      env {
        name  = "GOOGLE_CLOUD_PROJECT"
        value = var.project_id
      }
      env {
        name  = "ADAPTIVE_STORE"
        value = "firestore"
      }
      env {
        name  = "MODEL_URI"
        value = var.model_uri
      }
      env {
        name  = "FEEDBACK_TOPIC"
        value = google_pubsub_topic.feedback.name
      }
      env {
        name  = "FEEDBACK_BIGQUERY_TABLE"
        value = "${var.project_id}.${google_bigquery_dataset.analytics.dataset_id}.${google_bigquery_table.feedback.table_id}"
      }
      env {
        name  = "POLICY_VERSIONS_URI"
        value = "gs://${google_storage_bucket.data.name}/policy_versions"
      }
      startup_probe {
        initial_delay_seconds = 5
        timeout_seconds       = 5
        period_seconds        = 10
        failure_threshold     = 12
        http_get { path = "/" }
      }
      liveness_probe {
        timeout_seconds   = 5
        period_seconds    = 30
        failure_threshold = 3
        http_get { path = "/" }
      }
    }
  }
  depends_on = [
    google_project_service.enabled,
    google_project_iam_member.runtime_roles,
    google_bigquery_dataset_iam_member.runtime_feedback_writer,
    google_storage_bucket_iam_member.model_reader,
  ]

  lifecycle {
    # Revisions and traffic are rolled out by the protected deploy workflow.
    # Terraform continues to own service configuration and the scheduled job image.
    ignore_changes = [traffic, template[0].containers[0].image]
  }
}

resource "google_cloud_run_v2_service_iam_member" "pubsub_feedback_invoker" {
  project  = var.project_id
  location = google_cloud_run_v2_service.api.location
  name     = google_cloud_run_v2_service.api.name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.pubsub_push.email}"
}

resource "google_cloud_run_v2_service_iam_member" "terraform_health_invoker" {
  project  = var.project_id
  location = google_cloud_run_v2_service.api.location
  name     = google_cloud_run_v2_service.api.name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.github_terraform.email}"
}

resource "google_pubsub_topic_iam_member" "pubsub_dead_letter_writer" {
  topic  = google_pubsub_topic.feedback_dead_letter.name
  role   = "roles/pubsub.publisher"
  member = "serviceAccount:${local.pubsub_service_agent_email}"
}

resource "google_service_account" "consolidate" {
  account_id   = "ao-${var.environment}-consolidate"
  display_name = "Weekly policy consolidation job"
  depends_on   = [google_project_service.enabled]
}

resource "google_project_iam_member" "consolidate_firestore" {
  project = var.project_id
  role    = "roles/datastore.user"
  member  = "serviceAccount:${google_service_account.consolidate.email}"
}

resource "google_storage_bucket_iam_member" "consolidate_versions" {
  bucket = google_storage_bucket.data.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.consolidate.email}"
}

resource "google_cloud_run_v2_job" "consolidate" {
  name                = "${local.name}-consolidate"
  location            = var.region
  labels              = var.labels
  deletion_protection = var.environment == "prod"

  template {
    template {
      service_account = google_service_account.consolidate.email
      timeout         = "600s"
      max_retries     = 1
      containers {
        image   = var.container_image
        command = ["python", "scripts/consolidate_policy.py"]
        resources {
          limits = { cpu = "1", memory = "1Gi" }
        }
        env {
          name  = "GOOGLE_CLOUD_PROJECT"
          value = var.project_id
        }
        env {
          name  = "ADAPTIVE_STORE"
          value = "firestore"
        }
        env {
          name  = "GIT_SHA"
          value = var.git_sha
        }
        env {
          name  = "POLICY_VERSIONS_URI"
          value = "gs://${google_storage_bucket.data.name}/policy_versions"
        }
      }
    }
  }
  depends_on = [
    google_project_service.enabled,
    google_project_iam_member.consolidate_firestore,
    google_storage_bucket_iam_member.consolidate_versions,
  ]
}

resource "google_service_account" "scheduler" {
  account_id   = "ao-${var.environment}-scheduler"
  display_name = "Cloud Scheduler identity for policy consolidation"
  depends_on   = [google_project_service.enabled]
}

resource "google_cloud_run_v2_job_iam_member" "scheduler_invoker" {
  project  = var.project_id
  location = google_cloud_run_v2_job.consolidate.location
  name     = google_cloud_run_v2_job.consolidate.name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.scheduler.email}"
}

resource "google_service_account_iam_member" "scheduler_token_creator" {
  service_account_id = google_service_account.scheduler.name
  role               = "roles/iam.serviceAccountTokenCreator"
  member             = "serviceAccount:service-${data.google_project.current.number}@gcp-sa-cloudscheduler.iam.gserviceaccount.com"
}

resource "google_cloud_scheduler_job" "consolidate_weekly" {
  name             = "${local.name}-consolidate-weekly"
  description      = "Starts the policy consolidation job. The script decides whether a candidate is due."
  schedule         = var.consolidation_schedule
  paused           = var.consolidation_paused
  time_zone        = "America/Sao_Paulo"
  region           = var.region
  attempt_deadline = "180s"

  http_target {
    http_method = "POST"
    uri         = "https://run.googleapis.com/v2/projects/${var.project_id}/locations/${google_cloud_run_v2_job.consolidate.location}/jobs/${google_cloud_run_v2_job.consolidate.name}:run"
    body        = base64encode("{}")
    headers = {
      "Content-Type" = "application/json"
    }
    oauth_token {
      service_account_email = google_service_account.scheduler.email
    }
  }
  depends_on = [
    google_project_service.enabled,
    google_cloud_run_v2_job_iam_member.scheduler_invoker,
    google_service_account_iam_member.scheduler_token_creator,
  ]
}

resource "google_monitoring_alert_policy" "api_5xx" {
  display_name = "${local.name} API 5xx"
  combiner     = "OR"
  conditions {
    display_name = "Cloud Run responses in class 5xx"
    condition_threshold {
      filter          = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${local.name}\" AND metric.type=\"run.googleapis.com/request_count\" AND metric.labels.response_code_class=\"5xx\""
      comparison      = "COMPARISON_GT"
      threshold_value = 0
      duration        = "60s"
      aggregations {
        alignment_period   = "300s"
        per_series_aligner = "ALIGN_DELTA"
      }
    }
  }
  alert_strategy {
    auto_close = "1800s"
  }
  depends_on = [google_project_service.enabled]
}

resource "google_monitoring_alert_policy" "consolidate_failed" {
  display_name = "${local.name} consolidation job failed"
  combiner     = "OR"
  conditions {
    display_name = "Cloud Run job execution failed"
    condition_threshold {
      filter          = "resource.type=\"cloud_run_job\" AND resource.labels.job_name=\"${google_cloud_run_v2_job.consolidate.name}\" AND metric.type=\"run.googleapis.com/job/completed_execution_count\" AND metric.labels.result=\"failed\""
      comparison      = "COMPARISON_GT"
      threshold_value = 0
      duration        = "60s"
      aggregations {
        alignment_period   = "300s"
        per_series_aligner = "ALIGN_DELTA"
      }
    }
  }
  alert_strategy {
    auto_close = "1800s"
  }
  depends_on = [google_project_service.enabled]
}
