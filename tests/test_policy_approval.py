import json
from datetime import datetime, timezone

import pytest

from adaptive_offers.policy_state import read_current
from scripts.approve_policy import approve
from scripts.consolidate_policy import consolidate


def _approved_report():
    return {
        "git_sha": "abc123",
        "claim": {
            "simulation_uplift_ci_above_zero": True,
            "ope_clip_rate_below_10_percent": True,
            "model_ranking_matches_observed_test": True,
            "deployment_claim": True,
        },
        "test_observed_roc_auc": 0.75,
        "simulation": [{"policy": "thompson_sampling", "seeds": 30}],
        "ope": [{"policy": "thompson_sampling", "clip_rate": 0.02}],
    }


def test_weekly_candidate_requires_approval_before_serving(tmp_path):
    events = [
        {
            "request_id": "feedback-1",
            "action": "cellular",
            "reward": 1,
            "timestamp": "2026-09-01T12:00:00+00:00",
            "segment": "segment-a",
        }
    ]
    config = {
        "trigger": "interval",
        "interval_days": 7,
        "min_new_rewards": 1,
        "actions": ["cellular", "telephone"],
        "prior_alpha": 1.0,
        "prior_beta": 1.0,
        "model_strength": 40.0,
    }
    candidate = consolidate(
        events, config, tmp_path, now=datetime(2026, 9, 28, tzinfo=timezone.utc),
        git_sha="abc123",
    )
    assert candidate["status"] == "candidate_created"
    assert read_current(tmp_path) is None

    bad_report = _approved_report()
    bad_report["claim"]["deployment_claim"] = False
    with pytest.raises(ValueError, match="quality gate"):
        approve(candidate["version"], "reviewer", bad_report, tmp_path)
    assert read_current(tmp_path) is None

    result = approve(
        candidate["version"], "reviewer", _approved_report(), tmp_path,
        now=datetime(2026, 9, 29, tzinfo=timezone.utc),
    )
    assert result["promotion_status"] == "approved"
    version, state = read_current(tmp_path)
    assert version == candidate["version"]
    assert state["online_successes"]["segment-a"]["cellular"] == 1
    assert json.loads((tmp_path / "current.json").read_text())["approved_by"] == "reviewer"


def test_unapproved_legacy_current_is_ignored(tmp_path):
    (tmp_path / "current.json").write_text(
        json.dumps({"version": "legacy", "policy_state": {}})
    )
    assert read_current(tmp_path) is None


def test_api_feedback_flows_into_approved_posterior(tmp_path):
    import pandas as pd
    from fastapi.testclient import TestClient

    from adaptive_offers.api import create_app
    from adaptive_offers.decision_store import SQLiteDecisionStore

    class RewardModel:
        actions = ("cellular", "telephone")

        def predict_all(self, frame):
            return pd.DataFrame(
                {"cellular": [0.4] * len(frame), "telephone": [0.6] * len(frame)},
                index=frame.index,
            )

    store = SQLiteDecisionStore(tmp_path / "decisions.sqlite3")
    policy_dir = tmp_path / "policy_versions"
    app = create_app(
        RewardModel(), decision_store=store, load_posterior=True,
        policy_state_dir=policy_dir,
    )
    client = TestClient(app)
    context = {
        "age": 33, "job": "admin.", "marital": "married",
        "education": "university.degree", "default": "no", "housing": "yes",
        "loan": "no", "month": "may", "day_of_week": "mon", "campaign": 1,
        "pdays": 999, "previous": 0, "poutcome": "nonexistent",
        "emp.var.rate": 1.1, "cons.price.idx": 93.994,
        "cons.conf.idx": -36.4, "euribor3m": 4.857, "nr.employed": 5191.0,
    }
    recommendation = client.post(
        "/recommend", json=context, headers={"Idempotency-Key": "feedback-integration-1"}
    )
    assert recommendation.status_code == 200
    feedback = client.post(
        "/feedback", json={"request_id": "feedback-integration-1", "reward": 1}
    )
    assert feedback.status_code == 200
    assert len(store.feedback_events()) == 1

    candidate = consolidate(
        store.feedback_events(),
        {
            "trigger": "interval", "interval_days": 7, "min_new_rewards": 1,
            "actions": ["cellular", "telephone"], "prior_alpha": 1.0,
            "prior_beta": 1.0, "model_strength": 40.0,
        },
        policy_dir, now=datetime.now(timezone.utc),
        git_sha="abc123",
    )
    assert candidate["status"] == "candidate_created"
    assert client.get("/model-info").json()["feedback_posterior_active"] is False

    approve(candidate["version"], "reviewer", _approved_report(), policy_dir)
    app.state.posterior_cache["checked"] = False
    info = client.get("/model-info").json()
    assert info["policy"] == candidate["version"]
    assert info["feedback_posterior_active"] is True
