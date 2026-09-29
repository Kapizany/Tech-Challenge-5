"""Deterministic five-case Golden Set for recommendation review."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import ACTION_COLUMN, TARGET_COLUMN
from .policies import ThompsonSamplingPolicy
from .reward_model import ChannelRewardModel


def _clean_value(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if pd.isna(value):
        return None
    return value


def _context(row: pd.Series) -> dict[str, Any]:
    return {
        key: _clean_value(value)
        for key, value in row.drop(labels=[ACTION_COLUMN, TARGET_COLUMN, "duration"], errors="ignore").items()
    }


def select_golden_rows(
    test_frame: pd.DataFrame,
    count: int = 5,
    reward_model: ChannelRewardModel | None = None,
) -> list[tuple[str, pd.Series]]:
    """Select five stable, reviewable cases using deterministic criteria."""

    if len(test_frame) < count:
        raise ValueError(f"Golden Set requer pelo menos {count} linhas")
    selected: list[tuple[str, pd.Series]] = []
    used: set[object] = set()

    def choose(case_id: str, candidates: pd.DataFrame) -> None:
        for index, row in candidates.iterrows():
            if index not in used:
                used.add(index)
                selected.append((case_id, row))
                return

    unknown_mask = test_frame[["job", "marital", "education", "default", "housing", "loan"]].eq("unknown").any(axis=1)
    choose("unknown_category", test_frame[unknown_mask])
    if reward_model is not None and len(reward_model.actions) >= 2:
        expected = reward_model.predict_all(test_frame)
        first_action, second_action = reward_model.actions[:2]
        difference = expected[first_action] - expected[second_action]
        uncertainty = (expected.max(axis=1) - expected.min(axis=1)).sort_values()
        choose(f"likely_{first_action}", test_frame.loc[difference.sort_values(ascending=False).index])
        choose(f"likely_{second_action}", test_frame.loc[difference.sort_values().index])
        choose("high_uncertainty", test_frame.loc[uncertainty.index])
    else:
        choose("previous_success", test_frame[test_frame["poutcome"].eq("success")])
    choose("edge_campaign", test_frame.sort_values("campaign", ascending=False))
    choose("stable_reference", test_frame)

    if len(selected) < count:
        for index, row in test_frame.iterrows():
            if index not in used:
                selected.append((f"fallback_{len(selected) + 1}", row))
                used.add(index)
            if len(selected) == count:
                break
    return selected[:count]


def build_golden_set(
    test_frame: pd.DataFrame,
    reward_model: ChannelRewardModel,
    policy: ThompsonSamplingPolicy,
) -> list[dict[str, Any]]:
    """Generate recommendations, probabilities and review-oriented explanations."""

    records: list[dict[str, Any]] = []
    for case_id, row in select_golden_rows(test_frame, reward_model=reward_model):
        context = _context(row)
        context_frame = pd.DataFrame([context])
        expected_rewards = {
            action: float(reward_model.predict_proba(context_frame, action)[0])
            for action in reward_model.actions
        }
        recommendation = policy.select_action(pd.Series(context), request_id=f"golden-{case_id}")
        decision = policy.last_decision or {}
        best_expected_action = max(expected_rewards, key=expected_rewards.get)
        guardrail = decision.get("reason") == "guardrail"
        if guardrail and recommendation != best_expected_action:
            reason = "guardrail de exposição escolheu um canal diferente da maior recompensa esperada"
        elif recommendation == best_expected_action and guardrail:
            reason = "maior recompensa esperada; o guardrail de exposição apontou o mesmo canal"
        elif recommendation == best_expected_action:
            reason = "maior recompensa esperada entre os canais"
        else:
            reason = "exploração devido à incerteza posterior"
        records.append({
            "case_id": case_id,
            "source_index": str(row.name),
            "context": context,
            "recommendation": {
                "action": recommendation,
                "action_probability": float(decision.get("action_probability", policy.action_probability(pd.Series(context), recommendation))),
                "expected_reward_by_action": expected_rewards,
                "reason": reason,
                "policy_version": decision.get("policy_version", "thompson-sampling-v1"),
            },
            "human_review": {
                "status": "required",
                "question": "A ação é coerente com o contexto e com a finalidade da campanha?",
                "observed_action_excluded_from_context": True,
            },
        })
    return records
