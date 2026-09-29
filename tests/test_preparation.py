import numpy as np
import pandas as pd
import pytest

from adaptive_offers.config import COLUMNS
from adaptive_offers.preparation import (
    LEAKAGE_COLUMNS,
    PROXY_COLUMNS,
    context_columns,
    derive_decision_features,
    prepare_temporal_data,
    temporal_split,
)


def sample_frame(rows: int = 10) -> pd.DataFrame:
    row = {
        "age": 35, "job": "admin.", "marital": "married", "education": "university.degree",
        "default": "no", "housing": "yes", "loan": "no", "contact": "cellular",
        "month": "may", "day_of_week": "mon", "duration": 100, "campaign": 1,
        "pdays": -1, "previous": 0, "poutcome": "nonexistent", "emp.var.rate": 1.1,
        "cons.price.idx": 93.9, "cons.conf.idx": -36.4, "euribor3m": 4.8,
        "nr.employed": 5191.0, "y": "no",
    }
    frame = pd.DataFrame([row] * rows, columns=COLUMNS)
    frame["age"] = np.arange(20, 20 + rows)
    frame["y"] = np.where(np.arange(rows) % 2, "yes", "no")
    frame["contact"] = np.where(np.arange(rows) % 2, "telephone", "cellular")
    return frame


def test_temporal_split_preserves_order_and_sizes() -> None:
    splits = temporal_split(sample_frame(10))
    assert [len(splits.train), len(splits.validation), len(splits.test)] == [6, 2, 2]
    assert splits.train.index.tolist() == [0, 1, 2, 3, 4, 5]
    assert splits.validation.index.tolist() == [6, 7]
    assert splits.test.index.tolist() == [8, 9]


def test_context_has_no_action_target_or_post_event_duration() -> None:
    columns = set(context_columns())
    assert columns.isdisjoint(LEAKAGE_COLUMNS)
    assert {"duration", "contact", "y", "campaign", "pdays"}.isdisjoint(columns)
    assert {"contacts_before", "never_contacted", "pdays_contacted"}.issubset(columns)


def test_proxy_sensitivity_excludes_declared_proxies() -> None:
    columns = set(context_columns(include_proxies=False))
    assert columns.isdisjoint(PROXY_COLUMNS)
    assert "default" in columns
    assert "contacts_before" in columns
    assert "campaign" not in columns


def test_pipeline_fits_on_train_and_handles_unseen_categories() -> None:
    frame = sample_frame(10)
    frame.loc[8, "job"] = "retired"
    prepared = prepare_temporal_data(frame)
    names = set(prepared.pipeline.get_feature_names_out())
    assert {"duration", "contact", "y", "campaign", "pdays"}.isdisjoint(names)
    assert "contacts_before" in names
    assert "never_contacted" in names
    assert prepared.matrices["train"].shape[0] == 6
    assert prepared.matrices["validation"].shape[0] == 2
    assert prepared.matrices["test"].shape[0] == 2


def test_pdays_sentinel_and_current_campaign_are_rewritten() -> None:
    frame = sample_frame(4)
    frame.loc[0, "pdays"] = 999
    frame.loc[0, "campaign"] = 3
    frame.loc[1, "pdays"] = 12
    derived = derive_decision_features(frame)
    assert derived.loc[0, "never_contacted"] == 1
    assert derived.loc[0, "pdays_contacted"] == 0
    assert derived.loc[0, "contacts_before"] == 2
    assert derived.loc[1, "never_contacted"] == 0
    assert derived.loc[1, "pdays_contacted"] == 12
    assert derived.loc[2, "never_contacted"] == 1


def test_invalid_split_fractions_are_rejected() -> None:
    with pytest.raises(ValueError):
        temporal_split(sample_frame(), train_fraction=0.8, validation_fraction=0.3)
