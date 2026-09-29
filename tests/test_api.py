import base64
import json

import pandas as pd
from fastapi.testclient import TestClient

from adaptive_offers.api import create_app
from adaptive_offers.config import COLUMNS
from adaptive_offers.preparation import temporal_split
from adaptive_offers.reward_model import ChannelRewardModel


def sample_frame(rows: int = 30) -> pd.DataFrame:
    row = {
        "age": 35, "job": "admin.", "marital": "married", "education": "university.degree",
        "default": "no", "housing": "yes", "loan": "no", "contact": "cellular",
        "month": "may", "day_of_week": "mon", "duration": 100, "campaign": 1,
        "pdays": 999, "previous": 0, "poutcome": "nonexistent", "emp.var.rate": 1.1,
        "cons.price.idx": 93.994, "cons.conf.idx": -36.4, "euribor3m": 4.857,
        "nr.employed": 5191.0, "y": "no",
    }
    records = [row.copy() for _ in range(rows)]
    for index, item in enumerate(records):
        item["age"] = 20 + index
        item["contact"] = "cellular" if index % 2 == 0 else "telephone"
        item["y"] = "yes" if index % 3 == 0 else "no"
    return pd.DataFrame(records, columns=COLUMNS)


def client_payload() -> dict:
    return {
        "age": 33, "job": "admin.", "marital": "married", "education": "university.degree",
        "default": "no", "housing": "yes", "loan": "no", "month": "may", "day_of_week": "mon",
        "campaign": 1, "pdays": 999, "previous": 0, "poutcome": "nonexistent",
        "emp.var.rate": 1.1, "cons.price.idx": 93.994, "cons.conf.idx": -36.4,
        "euribor3m": 4.857, "nr.employed": 5191.0,
    }


def test_recommend_returns_a_channel_and_asks_for_review() -> None:
    splits = temporal_split(sample_frame())
    model = ChannelRewardModel.fit(splits.train)
    client = TestClient(create_app(model))
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["model_loaded"] is True
    assert health.headers["x-request-id"]
    assert float(health.headers["x-process-time-ms"]) >= 0
    propagated = client.get("/health", headers={"X-Request-ID": "trace-123"})
    assert propagated.headers["x-request-id"] == "trace-123"
    response = client.post(
        "/recommend", json=client_payload(), headers={"Idempotency-Key": "decision-1"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["action"] in set(model.actions)
    assert set(body["expected_reward_by_action"]) == set(model.actions)
    assert body["human_review_required"] is True
    assert body["policy_version"] == "thompson-sampling-contextual-v1"
    assert 0 < body["action_probability"] <= 1
    repeated = client.post(
        "/recommend", json=client_payload(), headers={"Idempotency-Key": "decision-1"}
    )
    assert repeated.json()["action"] == body["action"]
    assert repeated.json()["request_id"] == body["request_id"]

    # Exposure state is persisted between recommendations. With only one
    # recorded selection, the other action is forced and its propensity is 1.
    second = client.post(
        "/recommend", json=client_payload(), headers={"Idempotency-Key": "decision-2"}
    )
    assert second.status_code == 200
    assert second.json()["action"] != body["action"]
    assert second.json()["action_probability"] == 1.0

    conflict = client.post(
        "/recommend",
        json={**client_payload(), "age": 34},
        headers={"Idempotency-Key": "decision-1"},
    )
    assert conflict.status_code == 409


def test_feedback_is_idempotent_and_requires_a_known_decision() -> None:
    splits = temporal_split(sample_frame())
    model = ChannelRewardModel.fit(splits.train)
    client = TestClient(create_app(model))
    feedback = {"request_id": "missing", "reward": 1, "timestamp": "2026-09-27T12:00:00Z"}
    assert client.post("/feedback", json=feedback).status_code == 404

    recommendation = client.post(
        "/recommend", json=client_payload(), headers={"Idempotency-Key": "feedback-1"}
    )
    assert recommendation.status_code == 200
    feedback["request_id"] = recommendation.json()["request_id"]
    first = client.post("/feedback", json=feedback)
    second = client.post("/feedback", json=feedback)
    assert first.status_code == 200 and first.json()["status"] == "accepted"
    assert second.status_code == 200 and second.json()["status"] == "duplicate"
    conflict = client.post("/feedback", json={**feedback, "reward": 0})
    assert conflict.status_code == 409


def test_reusing_idempotency_key_with_invalid_path_characters_is_rejected() -> None:
    client = TestClient(create_app(object()))
    response = client.post(
        "/recommend", json=client_payload(), headers={"Idempotency-Key": "bad/key"}
    )
    assert response.status_code == 422


def test_pubsub_feedback_event_is_decoded_and_written_to_sink() -> None:
    splits = temporal_split(sample_frame())
    model = ChannelRewardModel.fit(splits.train)

    class _Sink:
        def __init__(self):
            self.events = []

        def write(self, event):
            self.events.append(event)

    sink = _Sink()
    client = TestClient(create_app(model, feedback_sink=sink))
    event = {
        "request_id": "event-1",
        "reward": 1,
        "timestamp": "2026-09-27T12:00:00Z",
    }
    envelope = {"message": {"data": base64.b64encode(json.dumps(event).encode()).decode()}}
    response = client.post("/events/feedback", json=envelope)
    assert response.status_code == 204
    assert sink.events == [event]


def test_consolidated_feedback_posterior_is_used_for_the_next_offer(tmp_path) -> None:
    from adaptive_offers.baselines import default_segment

    splits = temporal_split(sample_frame())
    model = ChannelRewardModel.fit(splits.train)
    segment = default_segment(client_payload())
    (tmp_path / "current.json").write_text(
        json.dumps({
            "version": "policy-from-feedback",
            "generated_at": "2026-09-28T00:00:00+00:00",
            "promotion_status": "approved",
            "approved_by": "reviewer",
            "approved_at": "2026-09-28T01:00:00+00:00",
            "policy_state": {
                "online_counts": {segment: {"cellular": 1, "telephone": 30}},
                "online_successes": {segment: {"cellular": 0, "telephone": 30}},
            },
        }),
        encoding="utf-8",
    )
    client = TestClient(create_app(model, load_posterior=True, policy_state_dir=tmp_path))
    response = client.post(
        "/recommend", json=client_payload(), headers={"Idempotency-Key": "posterior-1"}
    )
    assert response.status_code == 200
    assert response.json()["policy_version"] == "policy-from-feedback"
    assert client.get("/model-info").json()["feedback_posterior_active"] is True


def test_invalid_category_is_rejected() -> None:
    client = TestClient(create_app(object()))
    payload = client_payload()
    payload["job"] = "astronaut"
    response = client.post("/recommend", json=payload)
    assert response.status_code == 422


def test_missing_model_returns_service_unavailable() -> None:
    client = TestClient(create_app(None))
    assert client.get("/health").json()["status"] == "model_missing"
    assert client.get("/").status_code == 503
    response = client.post("/recommend", json=client_payload())
    assert response.status_code == 503
