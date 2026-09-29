#!/usr/bin/env python3
"""Train reward models and initialise the baseline/adaptive policies."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adaptive_offers.baselines import stats_table
from adaptive_offers.config import DEFAULT_DATA_PATH
from adaptive_offers.data import load_bank_marketing, validate_schema
from adaptive_offers.policies import (
    EpsilonGreedyPolicy,
    FixedBestPolicy,
    Guardrails,
    SegmentBaselinePolicy,
    ThompsonSamplingPolicy,
)
from adaptive_offers.preparation import modeling_frame, temporal_split
from adaptive_offers.reward_model import ChannelRewardModel
from adaptive_offers.tracking import lineage, mlflow_tags


def _json_safe(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts" / "phase3_4_report.json")
    parser.add_argument("--mlflow-experiment", default="adaptive-offers-phase3-4")
    args = parser.parse_args()

    raw = load_bank_marketing(args.input)
    validate_schema(raw)
    frame = modeling_frame(raw)
    splits = temporal_split(frame)
    train, validation, test = splits.train, splits.validation, splits.test

    model = ChannelRewardModel.fit(train)
    validation_metrics = model.evaluate_observed(validation)
    test_metrics = model.evaluate_observed(test)
    actions = model.actions

    fixed = FixedBestPolicy().fit(train)
    segment = SegmentBaselinePolicy(min_segment_observations=100).fit(train)
    thompson = ThompsonSamplingPolicy(
        actions, seed=42, guardrails=Guardrails(), reward_model=model,
    ).fit(train)
    epsilon = EpsilonGreedyPolicy(
        actions, epsilon=0.10, seed=42, guardrails=Guardrails(), reward_model=model,
    ).fit(train)

    example = validation.iloc[0]
    context = example.drop(labels=["y", "contact", "duration"])
    decisions = {}
    for name, policy in {
        "fixed_best": fixed,
        "segment_baseline": segment,
        "thompson_sampling": thompson,
        "epsilon_greedy": epsilon,
    }.items():
        action = policy.select_action(context)
        decisions[name] = {
            "action": action,
            "action_probability": policy.action_probability(context, action),
        }

    report = {
        "train_action_statistics": stats_table(train).to_dict(orient="records"),
        "best_train_action": fixed.best,
        "reward_model": {
            "actions": list(actions),
            "validation_observed_metrics": validation_metrics,
            "test_observed_metrics": test_metrics,
            "model_type": "calibrated logistic regression with shared context and action feature",
        },
        "policies": {
            "thompson_sampling": {
                "prior": [1.0, 1.0],
                "model_strength": 40.0,
                "context": "recompensa esperada do modelo calibrado por canal",
                "version": thompson.policy_version,
            },
            "epsilon_greedy": {"epsilon": 0.10, "version": epsilon.policy_version},
            "guardrails": {"min_action_share": 0.05, "max_action_share": 0.95},
        },
        "example_decisions": decisions,
        "evaluation_note": "As métricas do reward model usam somente a ação observada; ganho de política exige replay/OPE ou simulação.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model_path = args.output.parent / "channel_reward_model.joblib"
    joblib.dump(model, model_path)
    args.output.write_text(json.dumps(_json_safe(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    try:
        import mlflow

        mlflow.set_experiment(args.mlflow_experiment)
        with mlflow.start_run(run_name="phase3-4-reward-model-and-policies"):
            mlflow.set_tags(mlflow_tags(lineage(args.input)))
            mlflow.log_params({
                "train_rows": len(train), "validation_rows": len(validation), "test_rows": len(test),
                "best_train_action": fixed.best, "epsilon": 0.10, "thompson_prior_alpha": 1.0,
                "thompson_prior_beta": 1.0, "thompson_model_strength": 40.0,
                "guardrail_min_action_share": 0.05, "guardrail_max_action_share": 0.95,
                "min_segment_observations": 100,
            })
            mlflow.log_metrics({
                f"train_conversion_{row['action']}": float(row["conversion_rate"])
                for row in report["train_action_statistics"]
            })
            mlflow.log_metrics({
                f"validation_{key}": value for key, value in validation_metrics.items()
                if math.isfinite(value)
            })
            mlflow.log_metrics({
                f"test_{key}": value for key, value in test_metrics.items()
                if math.isfinite(value)
            })
            mlflow.log_artifact(str(model_path))
            mlflow.log_artifact(str(args.output))
    except ImportError as error:
        raise RuntimeError("MLflow é obrigatório para treinar; execute make install.") from error

    print(f"Reward model e políticas inicializados. Relatório: {args.output}")


if __name__ == "__main__":
    main()
