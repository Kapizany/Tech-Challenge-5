import pandas as pd

from adaptive_offers.config import COLUMNS
from adaptive_offers.golden_set import build_golden_set
from adaptive_offers.policies import ThompsonSamplingPolicy
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
        item["campaign"] = index + 1
        item["contact"] = "cellular" if index % 2 == 0 else "telephone"
        item["y"] = "yes" if index % 3 == 0 else "no"
    records[-1]["job"] = "unknown"
    records[-2]["poutcome"] = "success"
    return pd.DataFrame(records, columns=COLUMNS)


def test_golden_set_has_five_explainable_cases_without_leakage() -> None:
    splits = temporal_split(sample_frame())
    model = ChannelRewardModel.fit(splits.train)
    policy = ThompsonSamplingPolicy(model.actions, seed=42).fit(splits.train)
    records = build_golden_set(splits.test, model, policy)
    assert len(records) == 5
    assert len({record["case_id"] for record in records}) == 5
    for record in records:
        assert {"duration", "contact", "y"}.isdisjoint(record["context"])
        assert record["recommendation"]["action"] in model.actions
        assert record["human_review"]["status"] == "required"
