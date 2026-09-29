from datetime import datetime, timezone

import pytest

from adaptive_offers.decision_store import DecisionConflict, SQLiteDecisionStore


def test_sqlite_store_persists_decisions_feedback_and_exposure(tmp_path) -> None:
    path = tmp_path / "decisions.sqlite3"
    store = SQLiteDecisionStore(path)
    def builder(counts):
        return {"action": "telephone" if counts.get("cellular", 0) else "cellular"}

    first, duplicate = store.decide("one", "hash-a", builder)
    assert first["action"] == "cellular"
    assert duplicate is False
    assert store.decide("one", "hash-a", builder) == (first, True)
    with pytest.raises(DecisionConflict):
        store.decide("one", "hash-b", builder)

    second, _ = store.decide("two", "hash-c", builder)
    assert second["action"] == "telephone"
    assert store.record_feedback("one", 1, datetime.now(timezone.utc)) is False
    assert store.record_feedback("one", 1, datetime.now(timezone.utc)) is True
    with pytest.raises(DecisionConflict):
        store.record_feedback("one", 0, datetime.now(timezone.utc))

    reopened = SQLiteDecisionStore(path)
    third, _ = reopened.decide("three", "hash-d", builder)
    assert third["action"] == "telephone"
    assert reopened.health()
