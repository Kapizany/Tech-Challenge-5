"""Exploratory tables used by the notebook and the written report.

The functions describe the public Bank Marketing file. They do not fit the
serving policy. Duration appears only in the leakage comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import ACTION_COLUMN, TARGET_COLUMN
from .preparation import temporal_split

MONTH_ORDER = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


def _reward(frame: pd.DataFrame) -> pd.Series:
    return (frame[TARGET_COLUMN].astype(str) == "yes").astype(int)


def conversion_by(frame: pd.DataFrame, columns: str | list[str]) -> pd.DataFrame:
    """Count contacts and conversion rate for one or more grouping columns."""

    keys = [columns] if isinstance(columns, str) else list(columns)
    grouped = frame.groupby(keys, observed=True)[TARGET_COLUMN]
    table = grouped.agg(
        conversions=lambda series: int((series.astype(str) == "yes").sum()),
        contacts="size",
    ).reset_index()
    table["conversion_rate"] = table["conversions"] / table["contacts"]
    return table.sort_values(["conversion_rate", "contacts"], ascending=[False, False]).reset_index(drop=True)


def campaign_fatigue(frame: pd.DataFrame) -> pd.DataFrame:
    """Conversion by contacts already performed, including the current one."""

    work = frame.copy()
    campaign = pd.to_numeric(work["campaign"], errors="coerce")
    work["campaign_bin"] = pd.cut(
        campaign,
        bins=[0, 1, 2, 3, 5, np.inf],
        labels=["1", "2", "3", "4-5", "6+"],
        right=True,
    )
    return conversion_by(work, "campaign_bin")


def channel_by_month(frame: pd.DataFrame) -> pd.DataFrame:
    table = conversion_by(frame, [ACTION_COLUMN, "month"])
    table["month"] = pd.Categorical(table["month"], categories=MONTH_ORDER, ordered=True)
    return table.sort_values(["month", ACTION_COLUMN]).reset_index(drop=True)


def channel_by_period(frame: pd.DataFrame, bins: int = 5) -> pd.DataFrame:
    """Cross channel with file order. The UCI file is sorted from May 2008 to Nov 2010."""

    work = frame.copy()
    labels = [f"P{index + 1}" for index in range(bins)]
    work["periodo"] = pd.qcut(np.arange(len(work)), bins, labels=labels)
    return conversion_by(work, ["periodo", ACTION_COLUMN])


def channel_by_euribor(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["euribor_bin"] = pd.qcut(pd.to_numeric(work["euribor3m"], errors="coerce"), 4, duplicates="drop")
    table = conversion_by(work, ["euribor_bin", ACTION_COLUMN])
    table["euribor_bin"] = table["euribor_bin"].astype(str)
    return table


def macro_correlation(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
    return frame[columns].apply(pd.to_numeric, errors="coerce").corr()


def unknown_rates(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["job", "marital", "education", "default", "housing", "loan"]
    rows = []
    for column in columns:
        if column not in frame.columns:
            continue
        share = float((frame[column].astype(str) == "unknown").mean())
        rows.append({"column": column, "unknown_rate": share, "unknown_rows": int((frame[column].astype(str) == "unknown").sum())})
    return pd.DataFrame(rows)


def _propensity_auc(train: pd.DataFrame, test: pd.DataFrame, *, include_duration: bool) -> dict[str, float]:
    categorical = [
        "job", "marital", "education", "default", "housing", "loan",
        "month", "day_of_week", "poutcome", ACTION_COLUMN,
    ]
    numeric = [
        "age", "campaign", "pdays", "previous", "emp.var.rate",
        "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed",
    ]
    if include_duration:
        numeric = numeric + ["duration"]
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical),
    ])
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("logistic", LogisticRegression(max_iter=500, random_state=42)),
    ])
    features = numeric + categorical
    model.fit(train[features], _reward(train))
    probability = model.predict_proba(test[features])[:, 1]
    target = _reward(test)
    return {
        "test_roc_auc": float(roc_auc_score(target, probability)),
        "test_pr_auc": float(average_precision_score(target, probability)),
        "test_rows": int(len(test)),
    }


def duration_leakage_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare propensity models with and without call duration.

    The temporal split is the project protocol. The random split reproduces the
    higher AUCs common in public notebooks, which mix later campaign months into
    training. Neither model is the served policy.
    """

    temporal = temporal_split(frame)
    random_train, random_test = train_test_split(
        frame, test_size=0.2, random_state=42, stratify=frame[TARGET_COLUMN],
    )
    rows = []
    for split_name, train, test in (
        ("temporal", temporal.train, temporal.test),
        ("aleatorio", random_train, random_test),
    ):
        for include_duration in (False, True):
            metrics = _propensity_auc(train, test, include_duration=include_duration)
            rows.append({
                "split": split_name,
                "include_duration": include_duration,
                **metrics,
            })
    return pd.DataFrame(rows)
