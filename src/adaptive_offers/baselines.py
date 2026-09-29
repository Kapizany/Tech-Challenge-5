"""Deterministic and random policies plus train-only baseline statistics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from .config import ACTION_COLUMN, TARGET_COLUMN


@dataclass(frozen=True)
class ActionStats:
    action: str
    successes: int
    observations: int
    conversion_rate: float
    ci_low: float
    ci_high: float


def _wilson_interval(successes: int, observations: int, z: float = 1.96) -> tuple[float, float]:
    if observations == 0:
        return 0.0, 1.0
    p = successes / observations
    denominator = 1 + z**2 / observations
    centre = (p + z**2 / (2 * observations)) / denominator
    radius = z * np.sqrt((p * (1 - p) / observations) + z**2 / (4 * observations**2)) / denominator
    return max(0.0, centre - radius), min(1.0, centre + radius)


def action_statistics(frame: pd.DataFrame) -> dict[str, ActionStats]:
    """Calculate channel conversion and Wilson 95% intervals."""

    result: dict[str, ActionStats] = {}
    for action, group in frame.groupby(ACTION_COLUMN, observed=True):
        observations = int(len(group))
        successes = int((group[TARGET_COLUMN].astype(str) == "yes").sum())
        rate = successes / observations if observations else 0.0
        low, high = _wilson_interval(successes, observations)
        result[str(action)] = ActionStats(
            action=str(action), successes=successes, observations=observations,
            conversion_rate=rate, ci_low=low, ci_high=high,
        )
    return result


def stats_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Return serialisable baseline statistics ordered by conversion."""

    stats = action_statistics(frame)
    return pd.DataFrame([
        {
            "action": item.action,
            "successes": item.successes,
            "observations": item.observations,
            "conversion_rate": item.conversion_rate,
            "ci_low": item.ci_low,
            "ci_high": item.ci_high,
        }
        for item in sorted(stats.values(), key=lambda item: item.conversion_rate, reverse=True)
    ])


def best_action(frame: pd.DataFrame) -> str:
    """Select the best historical action using the supplied frame only."""

    stats = action_statistics(frame)
    if not stats:
        raise ValueError("Não há ações observadas para calcular o baseline")
    return max(stats.values(), key=lambda item: (item.conversion_rate, item.action)).action


def _segment_value(context: Mapping[str, object], field: str, default: str = "unknown") -> str:
    value = context.get(field, default)
    if value is None or pd.isna(value):
        return default
    return str(value)


def default_segment(context: Mapping[str, object]) -> str:
    """Create broad, non-sensitive segments with pooling-friendly cardinality."""

    cached = context.get("_default_segment")
    if cached is not None:
        return str(cached)

    if "contacts_before" in context and context.get("contacts_before") is not None:
        try:
            contacts_before = float(context["contacts_before"])
        except (TypeError, ValueError):
            contacts_before = 0.0
    else:
        try:
            contacts_before = max(float(context.get("campaign", 1)) - 1.0, 0.0)
        except (TypeError, ValueError):
            contacts_before = 0.0
    campaign_bin = "0" if contacts_before <= 0 else "1-2" if contacts_before <= 2 else "3+"
    return "|".join([
        f"poutcome={_segment_value(context, 'poutcome')}",
        f"contacts_before={campaign_bin}",
        f"housing={_segment_value(context, 'housing')}",
    ])


def segment_keys(frame: pd.DataFrame) -> pd.Series:
    return frame.apply(lambda row: default_segment(row.to_dict()), axis=1)
