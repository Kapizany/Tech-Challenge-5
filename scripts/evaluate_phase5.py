#!/usr/bin/env python3
"""Evaluate policies with semisynthetic simulation and observational OPE."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adaptive_offers.baselines import best_action
from adaptive_offers.config import DEFAULT_DATA_PATH
from adaptive_offers.data import load_bank_marketing, validate_schema
from adaptive_offers.eda import channel_by_euribor, channel_by_period, conversion_by
from adaptive_offers.evaluation import evaluate_policy_suite
from adaptive_offers.preparation import modeling_frame, temporal_split
from adaptive_offers.reward_model import ChannelRewardModel
from adaptive_offers.tracking import lineage, mlflow_tags


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts")
    parser.add_argument("--seeds", type=int, default=30)
    args = parser.parse_args()

    raw = load_bank_marketing(args.input)
    validate_schema(raw)
    frame = modeling_frame(raw)
    splits = temporal_split(frame)
    reward_model = ChannelRewardModel.fit(splits.train)
    policy_names = ["fixed_best", "random", "segment_baseline", "epsilon_greedy", "thompson_sampling"]
    print(
        f"Avaliando {len(splits.test):,} linhas, {len(policy_names)} políticas e "
        f"{args.seeds} seeds...",
        flush=True,
    )
    simulation_summary, ope_summary = evaluate_policy_suite(
        splits.train,
        splits.test,
        reward_model,
        policy_names=policy_names,
        seeds=tuple(range(args.seeds)),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    simulation_path = args.output_dir / "phase5_simulation_summary.csv"
    ope_path = args.output_dir / "phase5_ope_summary.csv"
    simulation_summary.to_csv(simulation_path, index=False)
    ope_summary.to_csv(ope_path, index=False)
    scores = reward_model.predict_all(splits.test)
    baseline_action = best_action(splits.train)
    chosen = scores.idxmax(axis=1)
    observed_test = conversion_by(splits.test, "contact")
    observed_best = str(observed_test.sort_values("conversion_rate", ascending=False).iloc[0]["contact"])
    model_best = str(scores.mean().idxmax())
    thompson = simulation_summary.set_index("policy").loc["thompson_sampling"]
    clip_rate = float(ope_summary.set_index("policy").loc["thompson_sampling"]["clip_rate"])
    test_roc = float(reward_model.evaluate_observed(splits.test)["roc_auc"])
    simulation_gain = float(thompson["uplift_ci_low"]) > 0
    ope_stable = clip_rate < 0.1
    ranking_matches_test = model_best == observed_best
    run_lineage = lineage(args.input)
    report = {
        **run_lineage,
        "simulation": simulation_summary.to_dict(orient="records"),
        "ope": ope_summary.to_dict(orient="records"),
        "context_disagreement_rate": float((chosen != baseline_action).mean()),
        "baseline_action": baseline_action,
        "observed_best_test_action": observed_best,
        "model_preferred_action_on_test": model_best,
        "mean_predicted_reward_on_test": {action: float(scores[action].mean()) for action in scores.columns},
        "test_observed_roc_auc": test_roc,
        "claim": {
            "simulation_uplift_ci_above_zero": simulation_gain,
            "ope_clip_rate_below_10_percent": ope_stable,
            "model_ranking_matches_observed_test": ranking_matches_test,
            "deployment_claim": bool(simulation_gain and ope_stable and ranking_matches_test and test_roc >= 0.5),
        },
        "confounding": {
            "marginal_channel": conversion_by(frame, "contact").to_dict(orient="records"),
            "channel_by_period": channel_by_period(frame).to_dict(orient="records"),
            "channel_by_euribor": channel_by_euribor(frame).assign(
                euribor_bin=lambda table: table["euribor_bin"].astype(str)
            ).to_dict(orient="records"),
        },
        "protocol": {
            "simulation": "recompensas Bernoulli amostradas a partir do reward model ajustado no treino",
            "ope": "IPS, SNIPS e Doubly Robust com propensão comportamental estimada e clipping",
            "policy_gain_claim": "somente se o IC do uplift contra fixed_best ultrapassar zero e OPE tiver overlap adequado",
        },
    }
    report_path = args.output_dir / "phase5_evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    try:
        import mlflow

        mlflow.set_experiment("adaptive-offers-phase5-evaluation")
        with mlflow.start_run(run_name="simulation-and-ope"):
            mlflow.set_tags(mlflow_tags(run_lineage))
            mlflow.set_tag("baseline_action", baseline_action)
            mlflow.log_params({"seeds": args.seeds, "evaluation_rows": len(splits.test), "propensity_clip": 0.01})
            mlflow.log_metric("test_observed_roc_auc", test_roc)
            mlflow.log_metrics({
                f"gate_{key}": float(value) for key, value in report["claim"].items()
            })
            for row in simulation_summary.to_dict(orient="records"):
                prefix = f"simulation_{row['policy']}"
                metrics = {
                    key: value for key, value in row.items()
                    if key != "policy" and isinstance(value, (int, float)) and pd.notna(value)
                }
                mlflow.log_metrics({f"{prefix}_{key}": float(value) for key, value in metrics.items()})
            for row in ope_summary.to_dict(orient="records"):
                prefix = f"ope_{row['policy']}"
                metrics = {
                    key: value for key, value in row.items()
                    if key != "policy" and isinstance(value, (int, float)) and pd.notna(value)
                }
                mlflow.log_metrics({f"{prefix}_{key}": float(value) for key, value in metrics.items()})
            mlflow.log_artifact(str(simulation_path))
            mlflow.log_artifact(str(ope_path))
            mlflow.log_artifact(str(report_path))
    except ImportError as error:
        raise RuntimeError("MLflow é obrigatório para avaliar; execute make install.") from error
    print("Resumo da simulação semissintética:")
    print(simulation_summary.to_string(index=False))
    print("\nResumo OPE:")
    print(ope_summary[["policy", "ips", "snips", "doubly_robust", "effective_sample_size", "clip_rate"]].to_string(index=False))
    print(f"\nArtefatos: {report_path}, {simulation_path}, {ope_path}")


if __name__ == "__main__":
    main()
