"""One calibrated reward model shared by both contact channels."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from .config import ACTION_COLUMN, TARGET_COLUMN
from .preparation import context_columns, fit_context_preprocessor


class _ConstantProbabilityModel:
    """Fallback for an action with one class or too few observations."""

    def __init__(self, probability: float):
        self.probability = float(probability)

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        probability = np.full(len(matrix), self.probability, dtype=float)
        return np.column_stack([1 - probability, probability])


@dataclass
class ChannelRewardModel:
    """Shared logistic model. The channel is a feature, so macro effects are not fit twice."""

    actions: tuple[str, ...]
    preprocessor: Any
    estimator: Any
    fallback_rates: dict[str, float] = field(default_factory=dict)
    context_columns: tuple[str, ...] = ()

    @classmethod
    def fit(cls, train_frame: pd.DataFrame, *, include_proxies: bool = True) -> "ChannelRewardModel":
        columns = context_columns(include_proxies=include_proxies)
        preprocessor = fit_context_preprocessor(
            train_frame, include_proxies=include_proxies, include_action=True,
        )
        actions = tuple(sorted(train_frame[ACTION_COLUMN].astype(str).unique()))
        target = (train_frame[TARGET_COLUMN].astype(str).to_numpy() == "yes").astype(int)
        observed = train_frame[ACTION_COLUMN].astype(str)
        fallback_rates = {
            action: float(target[observed.to_numpy() == action].mean()) if (observed == action).any() else 0.0
            for action in actions
        }
        counts = np.bincount(target, minlength=2)
        if len(target) < 6 or (counts < 2).any():
            estimator: Any = _ConstantProbabilityModel(float(target.mean()) if len(target) else 0.0)
        else:
            estimator = CalibratedClassifierCV(
                estimator=LogisticRegression(max_iter=1000, random_state=42),
                method="sigmoid",
                cv=min(3, int(counts.min())),
            )
            estimator.fit(preprocessor.transform(train_frame), target)
        return cls(
            actions=actions,
            preprocessor=preprocessor,
            estimator=estimator,
            fallback_rates=fallback_rates,
            context_columns=tuple(columns),
        )

    def _score_action(self, frame: pd.DataFrame, action: str) -> np.ndarray:
        scored = frame.copy()
        scored[ACTION_COLUMN] = action
        return self.estimator.predict_proba(self.preprocessor.transform(scored))[:, 1]

    def predict_proba(self, frame: pd.DataFrame, action: str) -> np.ndarray:
        """Estimate P(y=yes | context, action) for each row."""

        if action not in self.actions:
            raise ValueError(f"Ação não conhecida pelo modelo: {action}")
        return self._score_action(frame, action)

    def predict_all(self, frame: pd.DataFrame) -> pd.DataFrame:
        stacked = pd.concat(
            [frame.assign(**{ACTION_COLUMN: action}) for action in self.actions],
            ignore_index=True,
        )
        probabilities = self.estimator.predict_proba(self.preprocessor.transform(stacked))[:, 1]
        width = len(frame)
        return pd.DataFrame(
            {
                action: probabilities[index * width:(index + 1) * width]
                for index, action in enumerate(self.actions)
            },
            index=frame.index,
        )

    def evaluate_observed(self, frame: pd.DataFrame) -> dict[str, float]:
        """Evaluate only rows where the corresponding action was actually observed."""

        target = (frame[TARGET_COLUMN].astype(str) == "yes").astype(int).to_numpy()
        probabilities = np.zeros(len(frame), dtype=float)
        actions = frame[ACTION_COLUMN].astype(str).to_numpy()
        for action in self.actions:
            mask = actions == action
            if mask.any():
                probabilities[mask] = self.predict_proba(frame.loc[mask], action)
        metrics: dict[str, float] = {
            "brier_score": float(brier_score_loss(target, probabilities)),
            "log_loss": float(log_loss(target, np.clip(probabilities, 1e-6, 1 - 1e-6), labels=[0, 1])),
            "pr_auc": float(average_precision_score(target, probabilities)),
        }
        metrics["roc_auc"] = float(roc_auc_score(target, probabilities)) if len(np.unique(target)) == 2 else float("nan")
        return metrics
