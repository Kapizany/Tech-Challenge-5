"""Offline evaluation for adaptive policies.

The historical dataset only contains the reward for the action that happened.
This module therefore exposes two explicitly different tracks:

* a semi-synthetic simulator, where the fitted reward model supplies
  counterfactual probabilities and regret is measurable;
* observational off-policy evaluation (IPS, SNIPS and Doubly Robust), with
  propensity overlap and effective sample size reported alongside estimates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .baselines import default_segment
from .config import ACTION_COLUMN, TARGET_COLUMN
from .policies import (
    EpsilonGreedyPolicy,
    FixedBestPolicy,
    Guardrails,
    RandomPolicy,
    SegmentBaselinePolicy,
    ThompsonSamplingPolicy,
)
from .preparation import fit_context_preprocessor
from .reward_model import ChannelRewardModel

Policy = object


def make_policy(
    name: str,
    train_frame: pd.DataFrame,
    actions: Sequence[str],
    *,
    seed: int = 42,
    reward_model: ChannelRewardModel | None = None,
) -> Policy:
    """Build a fresh policy fitted only with the training period."""

    if name == "fixed_best":
        return FixedBestPolicy(actions).fit(train_frame)
    if name == "random":
        return RandomPolicy(actions, seed=seed)
    if name == "segment_baseline":
        return SegmentBaselinePolicy(actions, min_segment_observations=100).fit(train_frame)
    if name == "epsilon_greedy":
        return EpsilonGreedyPolicy(
            actions, epsilon=0.10, seed=seed, guardrails=Guardrails(),
            reward_model=reward_model,
        ).fit(train_frame)
    if name == "thompson_sampling":
        return ThompsonSamplingPolicy(
            actions, prior_alpha=1.0, prior_beta=1.0,
            min_segment_observations=100, seed=seed, guardrails=Guardrails(),
            reward_model=reward_model,
        ).fit(train_frame)
    raise ValueError(f"Política desconhecida: {name}")


def _bind_scores(policy: Policy, scores: pd.DataFrame) -> None:
    bind = getattr(policy, "bind_expected_rewards", None)
    if callable(bind):
        bind(scores)


def _context(row: pd.Series) -> pd.Series:
    return row.drop(labels=[TARGET_COLUMN, ACTION_COLUMN, "duration"], errors="ignore")


def _select(policy: Policy, context: pd.Series, request_id: str, *, log: bool = False) -> str:
    """Select an action while allowing replay callers to disable audit logs."""
    try:
        return policy.select_action(context, request_id=request_id, log=log)
    except TypeError:
        try:
            return policy.select_action(context, request_id=request_id)
        except TypeError:
            return policy.select_action(context)


def simulate_policies(
    train_frame: pd.DataFrame,
    evaluation_frame: pd.DataFrame,
    reward_model: ChannelRewardModel,
    *,
    policy_names: Sequence[str] = (
        "fixed_best", "random", "segment_baseline", "epsilon_greedy", "thompson_sampling"
    ),
    seeds: Sequence[int] = tuple(range(30)),
    compact: bool = False,
) -> pd.DataFrame:
    """Replay policies against semisynthetic Bernoulli rewards.

    Reward probabilities are produced by a model fitted on train, then sampled
    in the temporal order of ``evaluation_frame``. The fitted reward model is
    not refit on validation/test data.
    """

    reward_probabilities = reward_model.predict_all(evaluation_frame)
    actions = reward_model.actions
    evaluation_indices = evaluation_frame.index.to_numpy()
    contexts = []
    for _, row in evaluation_frame.iterrows():
        context = _context(row)
        context["_default_segment"] = default_segment(context)
        contexts.append(context)
    reward_matrix = reward_probabilities.to_numpy(dtype=float)
    action_positions = {action: position for position, action in enumerate(actions)}
    # Keeping every decision from 30 seeds would materialize millions of
    # dictionaries for the full UCI test split. The default remains useful for
    # diagnostics/tests; production evaluation requests compact aggregates.
    records: list[dict[str, object]] = []
    aggregate_records: list[dict[str, object]] = []
    for policy_name in policy_names:
        for seed in seeds:
            policy = make_policy(
                policy_name, train_frame, actions, seed=seed, reward_model=reward_model,
            )
            _bind_scores(policy, reward_probabilities)
            rng = np.random.default_rng(seed)
            reward_sum = 0
            expected_sum = 0.0
            regret_sum = 0.0
            for step, context in enumerate(contexts):
                action = _select(policy, context, f"sim-{policy_name}-{seed}-{step}", log=False)
                expected_reward = float(reward_matrix[step, action_positions[action]])
                reward = int(rng.binomial(1, expected_reward))
                oracle_reward = float(reward_matrix[step].max())
                policy.update(context, action, reward)
                reward_sum += reward
                expected_sum += expected_reward
                regret_sum += oracle_reward - expected_reward
                if not compact:
                    records.append({
                        "policy": policy_name,
                        "seed": seed,
                        "step": step,
                        "index": str(evaluation_indices[step]),
                        "action": action,
                        "reward": reward,
                        "expected_reward": expected_reward,
                        "oracle_reward": oracle_reward,
                        "regret": oracle_reward - expected_reward,
                    })
            if compact:
                aggregate_records.append({
                    "policy": policy_name,
                    "seed": seed,
                    "conversion_rate": reward_sum / len(contexts),
                    "expected_conversion": expected_sum / len(contexts),
                    "cumulative_regret": regret_sum,
                })
    return pd.DataFrame(aggregate_records if compact else records)


def _bootstrap_ci(values: np.ndarray, *, seed: int = 42, samples: int = 1000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def summarize_simulation(simulation: pd.DataFrame) -> pd.DataFrame:
    """Aggregate conversion and regret with seed-level bootstrap intervals."""

    if {"conversion_rate", "expected_conversion", "cumulative_regret"}.issubset(simulation.columns):
        by_seed = simulation[[
            "policy", "seed", "conversion_rate", "expected_conversion", "cumulative_regret"
        ]].copy()
    else:
        by_seed = simulation.groupby(["policy", "seed"], as_index=False).agg(
            conversion_rate=("reward", "mean"),
            expected_conversion=("expected_reward", "mean"),
            cumulative_regret=("regret", "sum"),
        )
    baseline_by_seed = (
        by_seed[by_seed["policy"] == "fixed_best"]
        .set_index("seed")["conversion_rate"]
        .to_dict()
    )
    rows = []
    for policy, group in by_seed.groupby("policy", observed=True):
        conversion_low, conversion_high = _bootstrap_ci(group["conversion_rate"].to_numpy())
        regret_low, regret_high = _bootstrap_ci(group["cumulative_regret"].to_numpy())
        uplift_values = np.asarray(
            [
                row.conversion_rate - baseline_by_seed[row.seed]
                for row in group.itertuples()
                if row.seed in baseline_by_seed
            ],
            dtype=float,
        )
        uplift_low, uplift_high = _bootstrap_ci(uplift_values)
        baseline_rate = float(np.mean(list(baseline_by_seed.values()))) if baseline_by_seed else float("nan")
        rows.append({
            "policy": policy,
            "conversion_rate": float(group["conversion_rate"].mean()),
            "conversion_ci_low": conversion_low,
            "conversion_ci_high": conversion_high,
            "expected_conversion": float(group["expected_conversion"].mean()),
            "baseline_conversion_rate": baseline_rate,
            "uplift_absolute": float(uplift_values.mean()) if len(uplift_values) else float("nan"),
            "uplift_ci_low": uplift_low,
            "uplift_ci_high": uplift_high,
            "uplift_relative": (
                float(uplift_values.mean() / baseline_rate)
                if len(uplift_values) and baseline_rate > 0 else float("nan")
            ),
            "cumulative_regret": float(group["cumulative_regret"].mean()),
            "regret_ci_low": regret_low,
            "regret_ci_high": regret_high,
            "seeds": int(group["seed"].nunique()),
        })
    return pd.DataFrame(rows).sort_values("conversion_rate", ascending=False).reset_index(drop=True)


@dataclass
class BehaviorPropensityModel:
    """Estimated logging-policy propensities for observational OPE."""

    actions: tuple[str, ...]
    preprocessor: object
    classifier: LogisticRegression | None
    empirical: dict[str, float]

    @classmethod
    def fit(cls, train_frame: pd.DataFrame, actions: Sequence[str]) -> "BehaviorPropensityModel":
        actions = tuple(actions)
        preprocessor = fit_context_preprocessor(train_frame)
        matrix = preprocessor.transform(train_frame)
        target = train_frame[ACTION_COLUMN].astype(str).to_numpy()
        empirical = {action: float(np.mean(target == action)) for action in actions}
        classifier: LogisticRegression | None = None
        if len(np.unique(target)) > 1:
            classifier = LogisticRegression(max_iter=1000, random_state=42)
            classifier.fit(matrix, target)
        return cls(actions=actions, preprocessor=preprocessor, classifier=classifier, empirical=empirical)

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.classifier is None:
            return pd.DataFrame(
                {action: self.empirical.get(action, 0.0) for action in self.actions}, index=frame.index
            )
        probabilities = self.classifier.predict_proba(self.preprocessor.transform(frame))
        result = pd.DataFrame(0.0, index=frame.index, columns=self.actions)
        for index, action in enumerate(self.classifier.classes_):
            if action in result.columns:
                result[action] = probabilities[:, index]
        return result


def _block_bootstrap(values: np.ndarray, *, block_size: int = 100, seed: int = 42, samples: int = 1000) -> np.ndarray:
    blocks = [values[start:start + block_size] for start in range(0, len(values), block_size)]
    if not blocks:
        return np.array([], dtype=float)
    rng = np.random.default_rng(seed)
    return np.concatenate([blocks[index] for index in rng.integers(0, len(blocks), size=len(blocks))])


def _ope_estimates(weights: np.ndarray, rewards: np.ndarray, dr_values: np.ndarray) -> dict[str, float]:
    denominator = float(weights.sum())
    return {
        "ips": float(np.mean(weights * rewards)),
        "snips": float((weights * rewards).sum() / denominator) if denominator else float("nan"),
        "doubly_robust": float(np.mean(dr_values)),
        "effective_sample_size": float((weights.sum() ** 2) / (np.square(weights).sum()))
        if np.square(weights).sum() else 0.0,
    }


def evaluate_ope(
    train_frame: pd.DataFrame,
    evaluation_frame: pd.DataFrame,
    policy_name: str,
    behavior_model: BehaviorPropensityModel,
    reward_model: ChannelRewardModel,
    *,
    seed: int = 42,
    propensity_clip: float = 0.01,
    block_size: int = 100,
    behavior: pd.DataFrame | None = None,
    q_values: pd.DataFrame | None = None,
) -> dict[str, float | str]:
    """Estimate a policy value with IPS, SNIPS and Doubly Robust OPE."""

    behavior = behavior if behavior is not None else behavior_model.predict(evaluation_frame)
    q_values = q_values if q_values is not None else reward_model.predict_all(evaluation_frame)
    policy = make_policy(
        policy_name, train_frame, reward_model.actions, seed=seed, reward_model=reward_model,
    )
    _bind_scores(policy, q_values)
    weights: list[float] = []
    rewards: list[float] = []
    dr_values: list[float] = []
    clipped = 0
    target_probabilities: list[float] = []
    observed_actions = evaluation_frame[ACTION_COLUMN].astype(str).to_numpy()
    observed_rewards = (evaluation_frame[TARGET_COLUMN].astype(str) == "yes").astype(float).to_numpy()

    contexts = []
    for _, row in evaluation_frame.iterrows():
        context = _context(row)
        context["_default_segment"] = default_segment(context)
        contexts.append(context)
    behavior_matrix = behavior.to_numpy(dtype=float)
    q_matrix = q_values.to_numpy(dtype=float)
    action_positions = {action: position for position, action in enumerate(reward_model.actions)}

    for position, context in enumerate(contexts):
        probability_method = getattr(policy, "action_probabilities", None)
        if callable(probability_method):
            target_probabilities_by_action = probability_method(context)
        else:
            target_probabilities_by_action = {
                action: policy.action_probability(context, action) for action in reward_model.actions
            }
        selected_action = _select(policy, context, f"ope-{policy_name}-{position}", log=False)
        observed_action = observed_actions[position]
        raw_behavior_probability = float(behavior_matrix[position, action_positions[observed_action]])
        behavior_probability = max(raw_behavior_probability, propensity_clip)
        clipped += int(raw_behavior_probability < propensity_clip)
        target_probability = target_probabilities_by_action.get(observed_action, 0.0)
        weight = target_probability / behavior_probability
        q_policy = sum(
            target_probabilities_by_action[action] * q_matrix[position, action_positions[action]]
            for action in reward_model.actions
        )
        q_observed = float(q_matrix[position, action_positions[observed_action]])
        weights.append(weight)
        rewards.append(observed_rewards[position])
        dr_values.append(q_policy + weight * (observed_rewards[position] - q_observed))
        # Adaptive policies receive feedback only when the replay action matches
        # the observed action; otherwise the counterfactual outcome is unknown.
        if selected_action == observed_action:
            policy.update(context, observed_action, observed_rewards[position])
        target_probabilities.append(target_probability)

    weights_array = np.asarray(weights, dtype=float)
    rewards_array = np.asarray(rewards, dtype=float)
    dr_array = np.asarray(dr_values, dtype=float)
    estimates = _ope_estimates(weights_array, rewards_array, dr_array)
    block_ips: list[float] = []
    block_snips: list[float] = []
    block_dr: list[float] = []
    for bootstrap_seed in range(100):
        indices = np.arange(len(weights_array))
        sampled = _block_bootstrap(indices, block_size=block_size, seed=seed + bootstrap_seed, samples=1).astype(int)
        block = _ope_estimates(weights_array[sampled], rewards_array[sampled], dr_array[sampled])
        block_ips.append(block["ips"])
        block_snips.append(block["snips"])
        block_dr.append(block["doubly_robust"])
    estimates.update({
        "ips_ci_low": float(np.quantile(block_ips, 0.025)),
        "ips_ci_high": float(np.quantile(block_ips, 0.975)),
        "snips_ci_low": float(np.quantile(block_snips, 0.025)),
        "snips_ci_high": float(np.quantile(block_snips, 0.975)),
        "doubly_robust_ci_low": float(np.quantile(block_dr, 0.025)),
        "doubly_robust_ci_high": float(np.quantile(block_dr, 0.975)),
        "policy": policy_name,
        "rows": int(len(evaluation_frame)),
        "propensity_min": float(behavior.to_numpy().min()),
        "propensity_p05": float(np.quantile(behavior.to_numpy(), 0.05)),
        "clip_rate": float(clipped / len(evaluation_frame)),
        "target_probability_mean": float(np.mean(target_probabilities)),
    })
    return estimates


def evaluate_policy_suite(
    train_frame: pd.DataFrame,
    evaluation_frame: pd.DataFrame,
    reward_model: ChannelRewardModel,
    *,
    policy_names: Sequence[str] = (
        "fixed_best", "random", "segment_baseline", "epsilon_greedy", "thompson_sampling"
    ),
    seeds: Sequence[int] = tuple(range(30)),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the semisynthetic simulator and observational OPE for all policies."""

    simulation = simulate_policies(
        train_frame, evaluation_frame, reward_model,
        policy_names=policy_names, seeds=seeds, compact=True,
    )
    simulation_summary = summarize_simulation(simulation)
    behavior_model = BehaviorPropensityModel.fit(train_frame, reward_model.actions)
    # These matrices depend only on the evaluation set, not on the policy.
    # Reuse them across OPE runs instead of refitting/transformation five times.
    behavior = behavior_model.predict(evaluation_frame)
    q_values = reward_model.predict_all(evaluation_frame)
    ope_rows = [
        evaluate_ope(
            train_frame,
            evaluation_frame,
            policy,
            behavior_model,
            reward_model,
            behavior=behavior,
            q_values=q_values,
        )
        for policy in policy_names
    ]
    ope_summary = pd.DataFrame(ope_rows)
    baseline = ope_summary[ope_summary["policy"] == "fixed_best"]
    if not baseline.empty:
        for metric in ("ips", "snips", "doubly_robust"):
            baseline_value = float(baseline.iloc[0][metric])
            ope_summary[f"{metric}_uplift"] = ope_summary[metric] - baseline_value
    return simulation_summary, ope_summary
