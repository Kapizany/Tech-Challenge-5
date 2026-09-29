output "api_url" {
  value       = google_cloud_run_v2_service.api.uri
  description = "Cloud Run URL; invocation still requires authenticated IAM access."
}

output "artifact_repository" {
  value = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.containers.repository_id}"
}

output "model_bucket" {
  value = google_storage_bucket.data.name
}

output "wif_provider" {
  value = google_iam_workload_identity_pool_provider.github.name
}

output "github_deploy_service_account" {
  value = google_service_account.github_deploy.email
}

output "github_terraform_service_account" {
  value = google_service_account.github_terraform.email
}

output "github_plan_service_account" {
  value = google_service_account.github_plan.email
}

output "data_bucket" {
  value = google_storage_bucket.data.name
}

output "consolidation_job" {
  value = google_cloud_run_v2_job.consolidate.name
}

output "consolidation_schedule" {
  value = google_cloud_scheduler_job.consolidate_weekly.schedule
}
