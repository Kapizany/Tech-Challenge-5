#!/usr/bin/env python3
"""Fail promotion unless predeclared evaluation gates pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def evaluate_gates(report: dict, *, minimum_seeds: int = 30, max_clip_rate: float = 0.10) -> list[str]:
    failures: list[str] = []
    claims = report.get("claim", {})
    if not claims.get("simulation_uplift_ci_above_zero", False):
        failures.append("Thompson Sampling uplift lower 95% CI is not above zero")
    if not claims.get("ope_clip_rate_below_10_percent", False):
        failures.append(f"OPE clipping rate is not below {max_clip_rate:.0%}")
    if not claims.get("model_ranking_matches_observed_test", False):
        failures.append("reward model ranking disagrees with observed temporal-test winner")
    if float(report.get("test_observed_roc_auc", 0.0)) < 0.5:
        failures.append("reward model temporal-test ROC-AUC is below 0.5")

    thompson = next(
        (row for row in report.get("simulation", []) if row.get("policy") == "thompson_sampling"),
        None,
    )
    if thompson is None or int(thompson.get("seeds", 0)) < minimum_seeds:
        failures.append(f"simulation requires at least {minimum_seeds} seeds")
    thompson_ope = next(
        (row for row in report.get("ope", []) if row.get("policy") == "thompson_sampling"),
        None,
    )
    if thompson_ope is None or float(thompson_ope.get("clip_rate", 1.0)) >= max_clip_rate:
        failures.append(f"Thompson OPE clipping rate must be below {max_clip_rate:.0%}")
    if not claims.get("deployment_claim", False):
        failures.append("report deployment_claim is false")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "phase5_evaluation_report.json",
    )
    parser.add_argument("--minimum-seeds", type=int, default=30)
    parser.add_argument("--max-clip-rate", type=float, default=0.10)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    failures = evaluate_gates(
        report, minimum_seeds=args.minimum_seeds, max_clip_rate=args.max_clip_rate
    )
    if failures:
        print("Promotion blocked:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("All promotion gates passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
