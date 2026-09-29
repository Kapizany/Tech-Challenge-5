"""Leakage-safe temporal splitting and context feature preparation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import ACTION_COLUMN, COLUMNS, TARGET_COLUMN

LEAKAGE_COLUMNS = frozenset({"duration", ACTION_COLUMN, TARGET_COLUMN})
# `campaign` includes the current contact; `pdays == 999` is the UCI sentinel.
REPLACED_CONTEXT_COLUMNS = frozenset({"campaign", "pdays"})
PROXY_COLUMNS = frozenset({"age", "job", "marital", "education"})
PDAYS_SENTINEL = 999
DERIVED_CONTEXT_COLUMNS = ("contacts_before", "never_contacted", "pdays_contacted")

NUMERIC_CONTEXT_COLUMNS = [
    "age", "contacts_before", "never_contacted", "pdays_contacted", "previous",
    "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed",
]
CATEGORICAL_CONTEXT_COLUMNS = [
    "job", "marital", "education", "default", "housing", "loan",
    "month", "day_of_week", "poutcome",
]


@dataclass(frozen=True)
class TemporalSplits:
    """Three chronological partitions retaining action and target columns."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    def as_dict(self) -> dict[str, pd.DataFrame]:
        return {"train": self.train, "validation": self.validation, "test": self.test}


@dataclass(frozen=True)
class PreparedTemporalData:
    """Prepared matrices and observed outcomes for a bandit/reward model."""

    splits: TemporalSplits
    pipeline: Pipeline
    matrices: dict[str, np.ndarray]
    actions: dict[str, np.ndarray]
    rewards: dict[str, np.ndarray]
    context_columns: tuple[str, ...]
    include_proxies: bool
    dropped_duplicates: int = 0


def context_columns(*, include_proxies: bool = True) -> list[str]:
    """Return columns available before the decision, excluding known leakage.

    Raw ``campaign`` and ``pdays`` are replaced by decision-time features:
    contacts already made in this campaign, a never-contacted flag, and the
    day count only when a previous contact exists.
    """

    excluded = LEAKAGE_COLUMNS | REPLACED_CONTEXT_COLUMNS
    columns = [column for column in COLUMNS if column not in excluded]
    if not include_proxies:
        columns = [column for column in columns if column not in PROXY_COLUMNS]
    derived = [column for column in DERIVED_CONTEXT_COLUMNS if column not in columns]
    if "previous" in columns:
        index = columns.index("previous")
        for offset, name in enumerate(derived):
            columns.insert(index + offset, name)
    else:
        columns.extend(derived)
    return columns


def modeling_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows while preserving campaign order."""

    return frame.drop_duplicates(keep="first")


def derive_decision_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add features that exist at decision time and remove sentinel distortion.

    ``campaign`` counts the contact being decided, so the known history is
    ``campaign - 1``. ``pdays`` uses 999 (this file) or a negative value (the
    other UCI extract) to mean "never contacted"; that sentinel is not a distance.
    """

    if "campaign" not in frame.columns or "pdays" not in frame.columns:
        raise ValueError("campaign e pdays são necessários para derivar o contexto")
    derived = frame.copy()
    campaign = pd.to_numeric(derived["campaign"], errors="coerce").fillna(1.0)
    pdays = pd.to_numeric(derived["pdays"], errors="coerce")
    never = pdays.isna() | (pdays >= PDAYS_SENTINEL) | (pdays < 0)
    derived["contacts_before"] = (campaign - 1.0).clip(lower=0.0)
    derived["never_contacted"] = never.astype(int)
    # Zero means "not applicable" when never_contacted is 1, not "contacted today".
    derived["pdays_contacted"] = pdays.where(~never, 0.0).fillna(0.0)
    return derived


class DecisionFeatureBuilder(BaseEstimator, TransformerMixin):
    """Sklearn step that adds decision-time columns before encoding."""

    def fit(self, features, target=None):
        return self

    def transform(self, features):
        return derive_decision_features(features)

    def get_feature_names_out(self, input_features=None):
        names = list(input_features) if input_features is not None else list(COLUMNS)
        for name in DERIVED_CONTEXT_COLUMNS:
            if name not in names:
                names.append(name)
        return np.asarray(names, dtype=object)


def temporal_split(
    frame: pd.DataFrame,
    *,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
) -> TemporalSplits:
    """Split in source order without shuffling.

    The full Bank Marketing file is ordered by campaign date. It has no complete
    timestamp, so source row order is the temporal proxy and is preserved exactly.
    """

    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction deve estar entre 0 e 1")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction deve estar entre 0 e 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction deve ser menor que 1")
    if len(frame) < 3:
        raise ValueError("São necessárias pelo menos 3 linhas para o split temporal")

    train_end = int(len(frame) * train_fraction)
    validation_end = int(len(frame) * (train_fraction + validation_fraction))
    if train_end < 1 or validation_end <= train_end or validation_end >= len(frame):
        raise ValueError("frações produzem partições vazias")
    return TemporalSplits(
        train=frame.iloc[:train_end].copy(),
        validation=frame.iloc[train_end:validation_end].copy(),
        test=frame.iloc[validation_end:].copy(),
    )


def build_context_preprocessor(*, include_proxies: bool = True, include_action: bool = False) -> Pipeline:
    """Build a transformation pipeline without fitting it.

    ``include_action`` is only for the reward model, where the channel is an
    input. The bandit context leaves the action out.
    """

    selected = set(context_columns(include_proxies=include_proxies))
    numeric = [column for column in NUMERIC_CONTEXT_COLUMNS if column in selected]
    categorical = [column for column in CATEGORICAL_CONTEXT_COLUMNS if column in selected]
    if include_action:
        categorical = categorical + [ACTION_COLUMN]
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    transformer = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline([
        ("decision_features", DecisionFeatureBuilder()),
        ("context_preprocessor", transformer),
    ])


def fit_context_preprocessor(
    train_frame: pd.DataFrame, *, include_proxies: bool = True, include_action: bool = False,
) -> Pipeline:
    """Fit transformations on ``train_frame`` only."""

    pipeline = build_context_preprocessor(
        include_proxies=include_proxies, include_action=include_action,
    )
    pipeline.fit(train_frame)
    return pipeline


def _observed_outcomes(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    actions = frame[ACTION_COLUMN].astype(str).to_numpy()
    rewards = (frame[TARGET_COLUMN].astype(str).to_numpy() == "yes").astype(np.int8)
    return actions, rewards


def prepare_temporal_data(
    frame: pd.DataFrame,
    *,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
    include_proxies: bool = True,
) -> PreparedTemporalData:
    """Split, fit on train and transform all periods without leakage."""

    cleaned = modeling_frame(frame)
    dropped_duplicates = int(len(frame) - len(cleaned))
    splits = temporal_split(
        cleaned,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )
    pipeline = fit_context_preprocessor(splits.train, include_proxies=include_proxies)
    columns = context_columns(include_proxies=include_proxies)
    matrices: dict[str, np.ndarray] = {}
    actions: dict[str, np.ndarray] = {}
    rewards: dict[str, np.ndarray] = {}
    for name, partition in splits.as_dict().items():
        matrices[name] = pipeline.transform(partition)
        actions[name], rewards[name] = _observed_outcomes(partition)
    return PreparedTemporalData(
        splits=splits, pipeline=pipeline, matrices=matrices, actions=actions,
        rewards=rewards, context_columns=tuple(columns), include_proxies=include_proxies,
        dropped_duplicates=dropped_duplicates,
    )


def save_preparation_artifacts(prepared: PreparedTemporalData, output_dir: str | Path) -> dict:
    """Persist fitted pipeline, matrices and a JSON-friendly manifest."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    joblib.dump(prepared.pipeline, destination / "context_preprocessor.joblib")
    for name, matrix in prepared.matrices.items():
        np.save(destination / f"X_{name}.npy", matrix)
        np.save(destination / f"actions_{name}.npy", prepared.actions[name])
        np.save(destination / f"rewards_{name}.npy", prepared.rewards[name])
    manifest = {
        "split": {
            name: {
                "rows": int(len(partition)),
                "first_index": str(partition.index[0]),
                "last_index": str(partition.index[-1]),
            }
            for name, partition in prepared.splits.as_dict().items()
        },
        "context_columns": list(prepared.context_columns),
        "excluded_columns": sorted(set(COLUMNS) - set(prepared.context_columns)),
        "include_proxies": prepared.include_proxies,
        "dropped_duplicates": prepared.dropped_duplicates,
        "replaced_columns": {
            "campaign": "contacts_before = campaign - 1, porque campaign inclui o contato atual",
            "pdays": "never_contacted e pdays_contacted; 999 ou valor negativo é sentinela",
        },
        "feature_count": int(prepared.matrices["train"].shape[1]),
        "fit_period": "train only",
    }
    (destination / "preparation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest
