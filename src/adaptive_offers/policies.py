"""Baseline and adaptive policies with a common serving interface."""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.special import betainc, betaln

from .baselines import best_action, default_segment, segment_keys
from .config import ACTION_COLUMN, TARGET_COLUMN

_LEGENDRE_NODES, _LEGENDRE_WEIGHTS = np.polynomial.legendre.leggauss(96)
_BETA_INTEGRATION_POINTS = (_LEGENDRE_NODES + 1.0) / 2.0
_BETA_INTEGRATION_WEIGHTS = _LEGENDRE_WEIGHTS / 2.0


@dataclass(frozen=True)
class Guardrails:
    """Exposure constraints shared by adaptive policies."""

    min_action_share: float = 0.05
    max_action_share: float = 0.95


def _context_dict(context: Mapping[str, object] | pd.Series) -> dict[str, object]:
    payload = context.to_dict() if isinstance(context, pd.Series) else dict(context)
    payload.pop("_default_segment", None)
    return payload


def lookup_expected_rewards(policy: object, context: Mapping[str, object] | pd.Series) -> dict[str, float] | None:
    """Read P(reward | context, action) from a bound table or the reward model.

    Offline evaluation binds the full score table so replay does not transform
    one row at a time. A single API request uses the reward model directly.
    """

    scores = getattr(policy, "_bound_scores", None)
    actions = tuple(getattr(policy, "actions", ()))
    if scores is not None and isinstance(context, pd.Series) and context.name in scores.index:
        row = scores.loc[context.name]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        return {action: float(row[action]) for action in actions}
    model = getattr(policy, "reward_model", None)
    if model is None:
        return None
    predicted = model.predict_all(pd.DataFrame([_context_dict(context)])).iloc[0]
    return {action: float(predicted[action]) for action in actions}


def _seed_from_segment(segment: str) -> int:
    return int(hashlib.sha256(segment.encode("utf-8")).hexdigest()[:8], 16)


class FixedBestPolicy:
    """Always select the best action measured on the training period."""

    def __init__(self, actions: Sequence[str] | None = None):
        self.actions = tuple(actions or ())
        self.best: str | None = None

    def fit(self, train_frame: pd.DataFrame) -> "FixedBestPolicy":
        self.best = best_action(train_frame)
        self.actions = tuple(sorted(train_frame[ACTION_COLUMN].astype(str).unique()))
        return self

    def select_action(self, context: Mapping[str, object] | pd.Series) -> str:
        if self.best is None:
            raise RuntimeError("Política não ajustada")
        return self.best

    def update(self, context: Mapping[str, object] | pd.Series, action: str, reward: float) -> None:
        return None

    def action_probability(self, context: Mapping[str, object] | pd.Series, action: str) -> float:
        return 1.0 if action == self.best else 0.0


class RandomPolicy:
    def __init__(self, actions: Sequence[str], seed: int = 42):
        self.actions = tuple(actions)
        self.rng = np.random.default_rng(seed)

    def select_action(self, context: Mapping[str, object] | pd.Series) -> str:
        return str(self.rng.choice(self.actions))

    def update(self, context: Mapping[str, object] | pd.Series, action: str, reward: float) -> None:
        return None

    def action_probability(self, context: Mapping[str, object] | pd.Series, action: str) -> float:
        return 1.0 / len(self.actions) if action in self.actions else 0.0


class SegmentBaselinePolicy(FixedBestPolicy):
    """Deterministic segment policy with fallback to the global best action."""

    def __init__(self, actions: Sequence[str] | None = None, min_segment_observations: int = 100):
        super().__init__(actions)
        self.min_segment_observations = min_segment_observations
        self.segment_best: dict[str, str] = {}

    def fit(self, train_frame: pd.DataFrame) -> "SegmentBaselinePolicy":
        super().fit(train_frame)
        segments = segment_keys(train_frame)
        for segment, group in train_frame.assign(_segment=segments).groupby("_segment", observed=True):
            if len(group) >= self.min_segment_observations:
                self.segment_best[str(segment)] = best_action(group)
        return self

    def select_action(self, context: Mapping[str, object] | pd.Series) -> str:
        segment = default_segment(_context_dict(context))
        return self.segment_best.get(segment, self.best)  # type: ignore[return-value]

    def action_probability(self, context: Mapping[str, object] | pd.Series, action: str) -> float:
        return 1.0 if action == self.select_action(context) else 0.0


class EpsilonGreedyPolicy:
    """Contextual empirical policy with explicit exploration probability."""

    def __init__(
        self,
        actions: Sequence[str],
        epsilon: float = 0.10,
        seed: int = 42,
        guardrails: Guardrails | None = None,
        policy_version: str = "epsilon-greedy-v1",
        reward_model: object | None = None,
    ):
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon deve estar entre 0 e 1")
        self.actions = tuple(actions)
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.guardrails = guardrails or Guardrails()
        self.reward_model = reward_model
        self._bound_scores: pd.DataFrame | None = None
        self.policy_version = (
            "epsilon-greedy-contextual-v1"
            if reward_model is not None and policy_version == "epsilon-greedy-v1"
            else policy_version
        )
        self.global_successes = {action: 0 for action in self.actions}
        self.global_counts = {action: 0 for action in self.actions}
        self.segment_successes: dict[str, dict[str, int]] = {}
        self.segment_counts: dict[str, dict[str, int]] = {}
        self.selection_counts = {action: 0 for action in self.actions}
        self.decision_log: list[dict[str, object]] = []

    def fit(self, train_frame: pd.DataFrame) -> "EpsilonGreedyPolicy":
        for action, group in train_frame.groupby(ACTION_COLUMN, observed=True):
            action = str(action)
            self.global_counts[action] = int(len(group))
            self.global_successes[action] = int((group[TARGET_COLUMN].astype(str) == "yes").sum())
        for segment, group in train_frame.assign(_segment=segment_keys(train_frame)).groupby("_segment", observed=True):
            key = str(segment)
            self.segment_counts[key] = {action: 0 for action in self.actions}
            self.segment_successes[key] = {action: 0 for action in self.actions}
            for action, action_group in group.groupby(ACTION_COLUMN, observed=True):
                action = str(action)
                self.segment_counts[key][action] = int(len(action_group))
                self.segment_successes[key][action] = int((action_group[TARGET_COLUMN].astype(str) == "yes").sum())
        return self

    def bind_expected_rewards(self, scores: pd.DataFrame) -> None:
        self._bound_scores = scores

    def _rates(self, context: Mapping[str, object] | pd.Series) -> dict[str, float]:
        expected = lookup_expected_rewards(self, context)
        if expected is not None:
            return expected
        segment = default_segment(_context_dict(context))
        counts = self.segment_counts.get(segment, self.global_counts)
        successes = self.segment_successes.get(segment, self.global_successes)
        return {action: (successes[action] + 1) / (counts[action] + 2) for action in self.actions}

    def select_action(
        self,
        context: Mapping[str, object] | pd.Series,
        request_id: str | None = None,
        *,
        log: bool = True,
    ) -> str:
        probabilities = self._probabilities(context)
        action = str(self.rng.choice(self.actions, p=[probabilities[a] for a in self.actions]))
        self.selection_counts[action] += 1
        if log:
            self.decision_log.append({
                "request_id": request_id or str(uuid.uuid4()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "policy_version": self.policy_version,
                "segment": default_segment(_context_dict(context)),
                "action": action,
                "action_probability": probabilities[action],
                "reason": "epsilon_greedy",
            })
        return action

    def _probabilities(self, context: Mapping[str, object] | pd.Series) -> dict[str, float]:
        rates = self._rates(context)
        best = max(self.actions, key=lambda action: (rates[action], action))
        probabilities = {action: self.epsilon / len(self.actions) for action in self.actions}
        probabilities[best] += 1 - self.epsilon
        return self._apply_guardrails(probabilities)

    def action_probabilities(self, context: Mapping[str, object] | pd.Series) -> dict[str, float]:
        """Return all action probabilities in one computation for OPE."""
        return self._probabilities(context)

    def _apply_guardrails(self, probabilities: dict[str, float]) -> dict[str, float]:
        floor = min(self.guardrails.min_action_share, 1 / len(self.actions))
        probabilities = {action: max(floor, value) for action, value in probabilities.items()}
        total = sum(probabilities.values())
        return {action: value / total for action, value in probabilities.items()}

    def update(self, context: Mapping[str, object] | pd.Series, action: str, reward: float) -> None:
        if action not in self.actions:
            raise ValueError(f"Ação não conhecida: {action}")
        segment = default_segment(_context_dict(context))
        self.global_counts[action] += 1
        self.global_successes[action] += int(reward > 0)
        self.segment_counts.setdefault(segment, {item: 0 for item in self.actions})[action] += 1
        self.segment_successes.setdefault(segment, {item: 0 for item in self.actions})[action] += int(reward > 0)

    def action_probability(self, context: Mapping[str, object] | pd.Series, action: str) -> float:
        return self._probabilities(context).get(action, 0.0)


class ThompsonSamplingPolicy:
    """Contextual Beta-Bernoulli Thompson Sampling with pooling and guardrails."""

    def __init__(
        self,
        actions: Sequence[str],
        *,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        min_segment_observations: int = 100,
        seed: int = 42,
        guardrails: Guardrails | None = None,
        policy_version: str = "thompson-sampling-v1",
        reward_model: object | None = None,
        model_strength: float = 40.0,
    ):
        if prior_alpha <= 0 or prior_beta <= 0:
            raise ValueError("priors Beta devem ser positivos")
        if model_strength <= 0:
            raise ValueError("model_strength deve ser positivo")
        self.actions = tuple(actions)
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.min_segment_observations = min_segment_observations
        self.rng = np.random.default_rng(seed)
        self.guardrails = guardrails or Guardrails()
        self.reward_model = reward_model
        self.model_strength = model_strength
        self._bound_scores: pd.DataFrame | None = None
        self.policy_version = (
            "thompson-sampling-contextual-v1"
            if reward_model is not None and policy_version == "thompson-sampling-v1"
            else policy_version
        )
        self.global_alpha = {action: prior_alpha for action in self.actions}
        self.global_beta = {action: prior_beta for action in self.actions}
        self.segment_alpha: dict[str, dict[str, float]] = {}
        self.segment_beta: dict[str, dict[str, float]] = {}
        self.segment_counts: dict[str, int] = {}
        self.online_successes: dict[str, dict[str, float]] = {}
        self.online_counts: dict[str, dict[str, float]] = {}
        self.selection_counts = {action: 0 for action in self.actions}
        self.last_decision: dict[str, object] | None = None
        self.decision_log: list[dict[str, object]] = []

    def bind_expected_rewards(self, scores: pd.DataFrame) -> None:
        self._bound_scores = scores

    def fit(self, train_frame: pd.DataFrame) -> "ThompsonSamplingPolicy":
        if self.reward_model is not None:
            # Train outcomes already sit inside the reward model. Counting them
            # again in the Beta posterior would hide the client context.
            return self
        for action, group in train_frame.groupby(ACTION_COLUMN, observed=True):
            action = str(action)
            successes = int((group[TARGET_COLUMN].astype(str) == "yes").sum())
            self.global_alpha[action] += successes
            self.global_beta[action] += len(group) - successes
        segments = train_frame.assign(_segment=segment_keys(train_frame))
        for segment, group in segments.groupby("_segment", observed=True):
            key = str(segment)
            self.segment_counts[key] = len(group)
            if len(group) < self.min_segment_observations:
                continue
            self.segment_alpha[key] = {action: self.prior_alpha for action in self.actions}
            self.segment_beta[key] = {action: self.prior_beta for action in self.actions}
            for action, action_group in group.groupby(ACTION_COLUMN, observed=True):
                action = str(action)
                successes = int((action_group[TARGET_COLUMN].astype(str) == "yes").sum())
                self.segment_alpha[key][action] += successes
                self.segment_beta[key][action] += len(action_group) - successes
        return self

    def _posterior(self, context: Mapping[str, object] | pd.Series) -> tuple[dict[str, float], dict[str, float]]:
        segment = default_segment(_context_dict(context))
        if segment in self.segment_alpha:
            return self.segment_alpha[segment], self.segment_beta[segment]
        return self.global_alpha, self.global_beta

    def _guardrail_forced_action(self) -> str | None:
        total = sum(self.selection_counts.values())
        if total == 0:
            return None
        shares = {action: count / total for action, count in self.selection_counts.items()}
        underexposed = [action for action, share in shares.items() if share < self.guardrails.min_action_share]
        if underexposed:
            return min(underexposed, key=lambda action: (self.selection_counts[action], action))
        overexposed = [action for action, share in shares.items() if share > self.guardrails.max_action_share]
        if overexposed:
            return min(self.actions, key=lambda action: (self.selection_counts[action], action))
        return None

    def _sampling_parameters(
        self, context: Mapping[str, object] | pd.Series
    ) -> tuple[dict[str, float], dict[str, float]]:
        expected = lookup_expected_rewards(self, context)
        if expected is None:
            return self._posterior(context)
        segment = default_segment(_context_dict(context))
        successes = self.online_successes.get(segment, {})
        counts = self.online_counts.get(segment, {})
        alpha: dict[str, float] = {}
        beta: dict[str, float] = {}
        for action in self.actions:
            probability = float(np.clip(expected[action], 1e-4, 1 - 1e-4))
            observed_successes = float(successes.get(action, 0.0))
            observed_trials = float(counts.get(action, 0.0))
            alpha[action] = self.prior_alpha + self.model_strength * probability + observed_successes
            beta[action] = (
                self.prior_beta
                + self.model_strength * (1 - probability)
                + observed_trials
                - observed_successes
            )
        return alpha, beta

    def _posterior_means(self, context: Mapping[str, object] | pd.Series) -> dict[str, float]:
        alpha, beta = self._sampling_parameters(context)
        return {action: alpha[action] / (alpha[action] + beta[action]) for action in self.actions}

    def select_action(
        self,
        context: Mapping[str, object] | pd.Series,
        request_id: str | None = None,
        *,
        log: bool = True,
    ) -> str:
        # Compute the propensity from the pre-decision state when a decision
        # record needs it. Replay callers pass log=False and calculate their
        # OPE propensity separately, avoiding duplicate integration work.
        probabilities = self.action_probabilities(context) if log else None
        alpha, beta = self._sampling_parameters(context)
        samples = {action: float(self.rng.beta(alpha[action], beta[action])) for action in self.actions}
        forced = self._guardrail_forced_action()
        action = forced or max(self.actions, key=lambda item: (samples[item], item))
        self.selection_counts[action] += 1
        segment = default_segment(_context_dict(context))
        decision = {
            "request_id": request_id or str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "policy_version": self.policy_version,
            "segment": segment,
            "action": action,
            "action_probability": probabilities[action] if probabilities is not None else None,
            "posterior_means": self._posterior_means(context) if log else {},
            "context_source": "reward_model" if self.reward_model is not None else "segment_posterior",
            "reason": "guardrail" if forced else "posterior_sample",
        }
        self.last_decision = decision
        if log:
            self.decision_log.append(decision)
        return action

    def update(self, context: Mapping[str, object] | pd.Series, action: str, reward: float) -> None:
        if action not in self.actions:
            raise ValueError(f"Ação não conhecida: {action}")
        success = int(reward > 0)
        segment = default_segment(_context_dict(context))
        if self.reward_model is not None:
            self.online_counts.setdefault(segment, {item: 0.0 for item in self.actions})
            self.online_successes.setdefault(segment, {item: 0.0 for item in self.actions})
            self.online_counts[segment][action] += 1
            self.online_successes[segment][action] += success
            return
        self.global_alpha[action] += success
        self.global_beta[action] += 1 - success
        if segment in self.segment_alpha:
            self.segment_alpha[segment][action] += success
            self.segment_beta[segment][action] += 1 - success

    def action_probabilities(
        self,
        context: Mapping[str, object] | pd.Series,
        *,
        samples: int = 0,
    ) -> dict[str, float]:
        """Return probabilities under the actual Thompson selector.

        Two arms use fixed Gauss-Legendre quadrature of their Beta densities.
        More than two arms use a deterministic-seed Monte Carlo estimate. An
        exposure guardrail that forces an action is represented as a point mass.
        """
        forced = self._guardrail_forced_action()
        if forced is not None:
            return {action: float(action == forced) for action in self.actions}

        alpha, beta = self._sampling_parameters(context)
        if len(self.actions) == 2 and samples == 0:
            first, second = self.actions
            points = _BETA_INTEGRATION_POINTS
            density = np.exp(
                (alpha[first] - 1) * np.log(points)
                + (beta[first] - 1) * np.log1p(-points)
                - betaln(alpha[first], beta[first])
            )
            first_probability = float(
                np.dot(
                    _BETA_INTEGRATION_WEIGHTS,
                    density * betainc(alpha[second], beta[second], points),
                )
            )
            probabilities = {
                first: float(np.clip(first_probability, 0.0, 1.0)),
                second: float(np.clip(1.0 - first_probability, 0.0, 1.0)),
            }
        elif samples > 0 or len(self.actions) > 2:
            segment = default_segment(_context_dict(context))
            sample_count = samples or 2048
            rng = np.random.default_rng(
                _seed_from_segment(segment) + sum(self.selection_counts.values())
            )
            draws = np.column_stack([
                rng.beta(alpha[item], beta[item], size=sample_count) for item in self.actions
            ])
            probabilities_array = np.mean(
                np.argmax(draws, axis=1)[:, None] == np.arange(len(self.actions)), axis=0
            )
            probabilities = dict(zip(self.actions, probabilities_array.astype(float)))
        else:
            probabilities = {self.actions[0]: 1.0}

        total = sum(probabilities.values())
        if total <= 0:
            return {action: 1.0 / len(self.actions) for action in self.actions}
        return {action: value / total for action, value in probabilities.items()}

    def action_probability(
        self,
        context: Mapping[str, object] | pd.Series,
        action: str,
        *,
        samples: int = 0,
    ) -> float:
        if action not in self.actions:
            return 0.0
        return self.action_probabilities(context, samples=samples)[action]
