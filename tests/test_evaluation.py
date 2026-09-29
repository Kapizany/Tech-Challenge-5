import pandas as pd

from adaptive_offers.config import COLUMNS
from adaptive_offers.evaluation import (
    BehaviorPropensityModel,
    evaluate_ope,
    simulate_policies,
    summarize_simulation,
)
from adaptive_offers.preparation import temporal_split
from adaptive_offers.reward_model import ChannelRewardModel


def sample_frame(rows: int = 30) -> pd.DataFrame:
    row = {
        "age": 35, "job": "admin.", "marital": "married", "education": "university.degree",
        "default": "no", "housing": "yes", "loan": "no", "contact": "cellular",
        "month": "may", "day_of_week": "mon", "duration": 100, "campaign": 1,
        "pdays": -1, "previous": 0, "poutcome": "nonexistent", "emp.var.rate": 1.1,
        "cons.price.idx": 93.9, "cons.conf.idx": -36.4, "euribor3m": 4.8,
        "nr.employed": 5191.0, "y": "no",
    }
    records = [row.copy() for _ in range(rows)]
    for index, item in enumerate(records):
        item["age"] = 20 + index
        item["contact"] = "cellular" if index % 2 == 0 else "telephone"
        item["y"] = "yes" if index % 3 == 0 else "no"
    return pd.DataFrame(records, columns=COLUMNS)


def test_simulation_has_policy_conversion_and_regret() -> None:
    splits = temporal_split(sample_frame())
    model = ChannelRewardModel.fit(splits.train)
    simulation = simulate_policies(
        splits.train, splits.test, model,
        policy_names=["fixed_best", "thompson_sampling"], seeds=[0, 1],
    )
    summary = summarize_simulation(simulation)
    assert set(summary["policy"]) == {"fixed_best", "thompson_sampling"}
    assert {
        "conversion_rate", "cumulative_regret", "conversion_ci_low", "uplift_absolute",
    }.issubset(summary.columns)


def test_ope_reports_estimators_overlap_and_ess() -> None:
    splits = temporal_split(sample_frame())
    model = ChannelRewardModel.fit(splits.train)
    behavior = BehaviorPropensityModel.fit(splits.train, model.actions)
    result = evaluate_ope(splits.train, splits.test, "thompson_sampling", behavior, model)
    assert {"ips", "snips", "doubly_robust", "effective_sample_size", "clip_rate"}.issubset(result)
    assert result["effective_sample_size"] > 0
