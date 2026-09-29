from datetime import datetime, timezone

from adaptive_offers.decision_store import SQLiteDecisionStore


def test_sqlite_store_exports_feedback_with_only_its_minimized_segment(tmp_path):
    store = SQLiteDecisionStore(tmp_path / "decisions.sqlite3")
    segment = "poutcome=failure|contacts_before=0|housing=yes"
    response, duplicate = store.decide(
        "request-1",
        "context-hash",
        lambda _: {"action": "cellular"},
        segment=segment,
    )
    assert response["action"] == "cellular"
    assert duplicate is False

    store.record_feedback("request-1", 1, datetime(2026, 9, 28, tzinfo=timezone.utc))

    assert store.feedback_events() == [
        {
            "request_id": "request-1",
            "action": "cellular",
            "segment": segment,
            "reward": 1,
            "timestamp": "2026-09-28T00:00:00+00:00",
        }
    ]


def test_sqlite_store_migrates_old_decision_table(tmp_path):
    import sqlite3

    path = tmp_path / "old.sqlite3"
    connection = sqlite3.connect(path)
    connection.executescript(
        "CREATE TABLE decisions (request_id TEXT PRIMARY KEY, context_hash TEXT NOT NULL, "
        "action TEXT NOT NULL, response_json TEXT NOT NULL, created_at TEXT NOT NULL);"
    )
    connection.close()

    store = SQLiteDecisionStore(path)

    columns = {row["name"] for row in store._connection.execute("PRAGMA table_info(decisions)")}
    assert "segment" in columns
