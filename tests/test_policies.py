import pandas as pd

from adaptive_offers.baselines import action_statistics, best_action, default_segment
from adaptive_offers.config import COLUMNS
from adaptive_offers.policies import (
    EpsilonGreedyPolicy,
    FixedBestPolicy,
    Guardrails,
    ThompsonSamplingPolicy,
)


def frame() -> pd.DataFrame:
    row = {
        "age": 35, "job": "admin.", "marital": "married", "education": "university.degree",
        "default": "no", "housing": "yes", "loan": "no", "contact": "cellular",
        "month": "may", "day_of_week": "mon", "duration": 100, "campaign": 1,
        "pdays": -1, "previous": 0, "poutcome": "nonexistent", "emp.var.rate": 1.1,
        "cons.price.idx": 93.9, "cons.conf.idx": -36.4, "euribor3m": 4.8,
        "nr.employed": 5191.0, "y": "no",
    }
    rows = [row.copy() for _ in range(8)]
    for index, item in enumerate(rows):
        item["contact"] = "cellular" if index < 4 else "telephone"
        item["y"] = "yes" if index < 2 or index == 4 else "no"
    return pd.DataFrame(rows, columns=COLUMNS)


def test_action_statistics_and_fixed_baseline_use_train_only() -> None:
    data = frame()
    stats = action_statistics(data)
    assert stats["cellular"].observations == 4
    assert best_action(data) == "cellular"
    assert FixedBestPolicy().fit(data).best == "cellular"


def test_thompson_sampling_updates_and_logs_decision() -> None:
    data = frame()
    policy = ThompsonSamplingPolicy(
        ["cellular", "telephone"], min_segment_observations=0, seed=7,
        guardrails=Guardrails(min_action_share=0.10, max_action_share=0.90),
    ).fit(data)
    context = data.iloc[0].drop(labels=["y", "contact", "duration"])
    action = policy.select_action(context, request_id="golden-1")
    assert action in {"cellular", "telephone"}
    assert policy.decision_log[-1]["request_id"] == "golden-1"
    assert policy.decision_log[-1]["policy_version"] == "thompson-sampling-v1"
    probabilities = [policy.action_probability(context, item) for item in policy.actions]
    assert abs(sum(probabilities) - 1) < 1e-6
    before = policy.global_alpha[action]
    policy.update(context, action, 1)
    assert policy.global_alpha[action] == before + 1


def test_contextual_thompson_uses_bound_reward_scores() -> None:
    data = frame()
    context = data.iloc[0].drop(labels=["y", "contact", "duration"])
    scores = pd.DataFrame(
        {"cellular": [0.01], "telephone": [0.80]},
        index=[context.name],
    )

    class _Scores:
        actions = ("cellular", "telephone")

    policy = ThompsonSamplingPolicy(
        ["cellular", "telephone"],
        seed=1,
        reward_model=_Scores(),
        guardrails=Guardrails(min_action_share=0.0, max_action_share=1.0),
        model_strength=200,
    ).fit(data)
    policy.bind_expected_rewards(scores)
    choices = [policy.select_action(context, log=False) for _ in range(30)]
    assert choices.count("telephone") > choices.count("cellular")
    assert policy.policy_version == "thompson-sampling-contextual-v1"
    assert policy.decision_log == []
    assert policy.last_decision["context_source"] == "reward_model"


def test_thompson_probability_matches_selector_and_forced_guardrail() -> None:
    data = frame()
    context = data.iloc[0].drop(labels=["y", "contact", "duration"])
    policy = ThompsonSamplingPolicy(
        ["cellular", "telephone"], seed=3,
        guardrails=Guardrails(min_action_share=0.10, max_action_share=0.90),
    ).fit(data)

    probabilities = policy.action_probabilities(context)
    assert abs(sum(probabilities.values()) - 1.0) < 1e-9
    selected = policy.select_action(context)
    assert policy.last_decision["action_probability"] == probabilities[selected]
    policy.selection_counts[selected] = 100
    policy.selection_counts[next(action for action in policy.actions if action != selected)] = 0
    forced_probabilities = policy.action_probabilities(context)
    forced = next(action for action, probability in forced_probabilities.items() if probability == 1)
    assert forced != selected


def test_epsilon_greedy_probability_and_segment_key() -> None:
    data = frame()
    policy = EpsilonGreedyPolicy(["cellular", "telephone"], epsilon=0.2, seed=1).fit(data)
    context = data.iloc[0].drop(labels=["y", "contact", "duration"])
    action = policy.select_action(context, request_id="epsilon-1")
    assert action in policy.actions
    assert abs(sum(policy.action_probability(context, item) for item in policy.actions) - 1) < 1e-6
    assert "poutcome=nonexistent" in default_segment(context)
