#!/usr/bin/env python3
"""Verify that training and policy comparison are recorded in MLflow."""

from __future__ import annotations

import argparse

import mlflow
from mlflow.tracking import MlflowClient


def _runs(client: MlflowClient, experiment_name: str):
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return []
    return client.search_runs([experiment.experiment_id], max_results=100, order_by=["start_time DESC"])


def verify_tracking(minimum_seeds: int) -> dict[str, str]:
    client = MlflowClient()
    train_runs = _runs(client, "adaptive-offers-phase3-4")
    train = next(
        (
            run for run in train_runs
            if run.info.status == "FINISHED"
            and {"git_sha", "data_sha256"} <= set(run.data.tags)
            and "best_train_action" in run.data.params
            and "thompson_model_strength" in run.data.params
            and any(key.startswith("train_conversion_") for key in run.data.metrics)
            and "test_roc_auc" in run.data.metrics
            and "test_pr_auc" in run.data.metrics
        ),
        None,
    )
    evaluation_runs = _runs(client, "adaptive-offers-phase5-evaluation")
    evaluation = next(
        (
            run for run in evaluation_runs
            if run.info.status == "FINISHED"
            and {"git_sha", "data_sha256"} <= set(run.data.tags)
            and int(run.data.params.get("seeds", 0)) >= minimum_seeds
            and "gate_deployment_claim" in run.data.metrics
            and "simulation_fixed_best_conversion_rate" in run.data.metrics
            and "simulation_thompson_sampling_uplift_absolute" in run.data.metrics
            and "ope_thompson_sampling_clip_rate" in run.data.metrics
        ),
        None,
    )
    if train is None or evaluation is None:
        raise RuntimeError(
            "MLflow incompleto: execute make train-policies e make evaluate "
            f"SEEDS={minimum_seeds} no mesmo tracking URI."
        )
    return {
        "tracking_uri": mlflow.get_tracking_uri(),
        "train_run_id": train.info.run_id,
        "evaluation_run_id": evaluation.info.run_id,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minimum-seeds", type=int, default=30)
    args = parser.parse_args()
    if args.minimum_seeds < 1:
        parser.error("--minimum-seeds deve ser positivo")
    print(verify_tracking(args.minimum_seeds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
