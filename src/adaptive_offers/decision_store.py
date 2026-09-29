"""Idempotent decision and feedback storage for local and cloud serving."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol


class DecisionConflict(ValueError):
    """An idempotency key was reused with a different request or feedback."""


class UnknownDecision(KeyError):
    """Feedback references a decision that was never recorded."""


DecisionBuilder = Callable[[dict[str, int]], dict[str, Any]]


class DecisionStore(Protocol):
    def decide(
        self,
        request_id: str,
        context_hash: str,
        builder: DecisionBuilder,
        *,
        segment: str | None = None,
    ) -> tuple[dict[str, Any], bool]: ...

    def record_feedback(self, request_id: str, reward: int, timestamp: datetime) -> bool: ...

    def feedback_events(self) -> list[dict[str, Any]]: ...

    def health(self) -> bool: ...


class SQLiteDecisionStore:
    """Transactional single-host store for development and local demos."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            self.path, timeout=30, check_same_thread=False, isolation_level=None
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS decisions (
                request_id TEXT PRIMARY KEY,
                context_hash TEXT NOT NULL,
                action TEXT NOT NULL,
                response_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                segment TEXT
            );
            CREATE TABLE IF NOT EXISTS feedback (
                request_id TEXT PRIMARY KEY REFERENCES decisions(request_id),
                reward INTEGER NOT NULL CHECK (reward IN (0, 1)),
                timestamp TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )

        decision_columns = {
            str(row["name"])
            for row in self._connection.execute("PRAGMA table_info(decisions)").fetchall()
        }
        if "segment" not in decision_columns:
            self._connection.execute("ALTER TABLE decisions ADD COLUMN segment TEXT")

    def decide(
        self,
        request_id: str,
        context_hash: str,
        builder: DecisionBuilder,
        *,
        segment: str | None = None,
    ) -> tuple[dict[str, Any], bool]:
        with self._lock:
            connection = self._connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                existing = connection.execute(
                    "SELECT context_hash, response_json FROM decisions WHERE request_id = ?",
                    (request_id,),
                ).fetchone()
                if existing is not None:
                    if existing["context_hash"] != context_hash:
                        raise DecisionConflict("Idempotency-Key já usado com outro contexto")
                    connection.commit()
                    return json.loads(existing["response_json"]), True

                rows = connection.execute(
                    "SELECT action, COUNT(*) AS count FROM decisions GROUP BY action"
                ).fetchall()
                counts = {str(row["action"]): int(row["count"]) for row in rows}
                response = builder(counts)
                action = str(response["action"])
                connection.execute(
                    "INSERT INTO decisions(request_id, context_hash, action, response_json, created_at, segment) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        request_id,
                        context_hash,
                        action,
                        json.dumps(response, ensure_ascii=False, separators=(",", ":")),
                        datetime.now(timezone.utc).isoformat(),
                        segment,
                    ),
                )
                connection.commit()
                return response, False
            except Exception:
                connection.rollback()
                raise

    def record_feedback(self, request_id: str, reward: int, timestamp: datetime) -> bool:
        with self._lock:
            connection = self._connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                exists = connection.execute(
                    "SELECT 1 FROM decisions WHERE request_id = ?", (request_id,)
                ).fetchone()
                if exists is None:
                    raise UnknownDecision(request_id)
                old_feedback = connection.execute(
                    "SELECT reward, timestamp FROM feedback WHERE request_id = ?",
                    (request_id,),
                ).fetchone()
                normalized_timestamp = timestamp.astimezone(timezone.utc).isoformat()
                if old_feedback is not None:
                    if old_feedback["reward"] != reward:
                        raise DecisionConflict("Feedback conflitante para esta decisão")
                    connection.commit()
                    return True
                connection.execute(
                    "INSERT INTO feedback(request_id, reward, timestamp, created_at) VALUES (?, ?, ?, ?)",
                    (
                        request_id,
                        reward,
                        normalized_timestamp,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
                connection.commit()
                return False
            except Exception:
                connection.rollback()
                raise

    def health(self) -> bool:
        with self._lock:
            return self._connection.execute("SELECT 1").fetchone()[0] == 1

    def feedback_events(self) -> list[dict[str, Any]]:
        """Return joined feedback ordered by event time and minimized segment."""
        with self._lock:
            rows = self._connection.execute(
                "SELECT f.request_id, d.action, d.segment, f.reward, f.timestamp "
                "FROM feedback AS f JOIN decisions AS d USING (request_id) "
                "ORDER BY f.timestamp, f.request_id"
            ).fetchall()
            return [dict(row) for row in rows]


class MemoryDecisionStore:
    """Thread-safe ephemeral store used by tests and standalone helper calls."""

    def __init__(self):
        self._lock = threading.RLock()
        self._decisions: dict[str, tuple[str, dict[str, Any]]] = {}
        self._segments: dict[str, str | None] = {}
        self._feedback: dict[str, tuple[int, str]] = {}

    def decide(
        self,
        request_id: str,
        context_hash: str,
        builder: DecisionBuilder,
        *,
        segment: str | None = None,
    ) -> tuple[dict[str, Any], bool]:
        with self._lock:
            existing = self._decisions.get(request_id)
            if existing is not None:
                old_hash, response = existing
                if old_hash != context_hash:
                    raise DecisionConflict("Idempotency-Key já usado com outro contexto")
                return response.copy(), True
            counts: dict[str, int] = {}
            for _, response in self._decisions.values():
                action = str(response["action"])
                counts[action] = counts.get(action, 0) + 1
            response = builder(counts)
            self._decisions[request_id] = (context_hash, response.copy())
            self._segments[request_id] = segment
            return response, False

    def record_feedback(self, request_id: str, reward: int, timestamp: datetime) -> bool:
        with self._lock:
            if request_id not in self._decisions:
                raise UnknownDecision(request_id)
            old_feedback = self._feedback.get(request_id)
            if old_feedback is not None:
                if old_feedback[0] != reward:
                    raise DecisionConflict("Feedback conflitante para esta decisão")
                return True
            self._feedback[request_id] = (reward, timestamp.astimezone(timezone.utc).isoformat())
            return False

    def health(self) -> bool:
        return True

    def feedback_events(self) -> list[dict[str, Any]]:
        with self._lock:
            events = [
                {
                    "request_id": request_id,
                    "action": self._decisions[request_id][1]["action"],
                    "segment": self._segments.get(request_id),
                    "reward": reward,
                    "timestamp": timestamp,
                }
                for request_id, (reward, timestamp) in self._feedback.items()
            ]
        return sorted(events, key=lambda item: (item["timestamp"], item["request_id"]))


class FirestoreDecisionStore:
    """Firestore implementation with transactionally shared exposure counts."""

    def __init__(self, project: str | None = None):
        from google.cloud import firestore

        self._firestore = firestore
        self._client = firestore.Client(project=project)
        self._decisions = self._client.collection("adaptive_offer_decisions")
        self._feedback = self._client.collection("adaptive_offer_feedback")
        self._state = self._client.collection("adaptive_offer_state").document("exposure")

    def decide(
        self,
        request_id: str,
        context_hash: str,
        builder: DecisionBuilder,
        *,
        segment: str | None = None,
    ) -> tuple[dict[str, Any], bool]:
        decision_ref = self._decisions.document(request_id)
        transaction = self._client.transaction()

        @self._firestore.transactional
        def run(transaction):
            existing = decision_ref.get(transaction=transaction)
            if existing.exists:
                previous = existing.to_dict()
                if previous["context_hash"] != context_hash:
                    raise DecisionConflict("Idempotency-Key já usado com outro contexto")
                return previous["response"], True
            state_snapshot = self._state.get(transaction=transaction)
            counts = state_snapshot.to_dict().get("counts", {}) if state_snapshot.exists else {}
            response = builder({str(key): int(value) for key, value in counts.items()})
            action = str(response["action"])
            counts[action] = int(counts.get(action, 0)) + 1
            transaction.set(self._state, {"counts": counts})
            transaction.create(
                decision_ref,
                {
                    "request_id": request_id,
                    "context_hash": context_hash,
                    "action": action,
                    "segment": segment,
                    "response": response,
                    "created_at": datetime.now(timezone.utc),
                },
            )
            return response, False

        return run(transaction)

    def record_feedback(self, request_id: str, reward: int, timestamp: datetime) -> bool:
        decision_ref = self._decisions.document(request_id)
        feedback_ref = self._feedback.document(request_id)
        transaction = self._client.transaction()

        @self._firestore.transactional
        def run(transaction):
            decision = decision_ref.get(transaction=transaction)
            if not decision.exists:
                raise UnknownDecision(request_id)
            old_feedback = feedback_ref.get(transaction=transaction)
            if old_feedback.exists:
                if int(old_feedback.to_dict()["reward"]) != reward:
                    raise DecisionConflict("Feedback conflitante para esta decisão")
                return True
            transaction.create(
                feedback_ref,
                {
                    "request_id": request_id,
                    "reward": reward,
                    "timestamp": timestamp.astimezone(timezone.utc),
                    "created_at": datetime.now(timezone.utc),
                },
            )
            return False

        return run(transaction)

    def health(self) -> bool:
        self._state.get()
        return True

    def feedback_events(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for snapshot in self._feedback.stream():
            feedback = snapshot.to_dict()
            decision_snapshot = self._decisions.document(snapshot.id).get()
            if not decision_snapshot.exists:
                continue
            decision = decision_snapshot.to_dict()
            timestamp = feedback["timestamp"]
            if isinstance(timestamp, datetime):
                timestamp = timestamp.astimezone(timezone.utc).isoformat()
            events.append(
                {
                    "request_id": snapshot.id,
                    "action": decision.get("action"),
                    "segment": decision.get("segment"),
                    "reward": int(feedback["reward"]),
                    "timestamp": str(timestamp),
                }
            )
        return sorted(events, key=lambda item: (item["timestamp"], item["request_id"]))


def configured_store() -> DecisionStore:
    """Select local SQLite or cloud Firestore from explicit environment config."""

    backend = os.getenv("ADAPTIVE_STORE", "sqlite").lower()
    if backend == "firestore":
        return FirestoreDecisionStore(project=os.getenv("GOOGLE_CLOUD_PROJECT"))
    if backend != "sqlite":
        raise ValueError("ADAPTIVE_STORE deve ser 'sqlite' ou 'firestore'")
    return SQLiteDecisionStore(
        os.getenv("ADAPTIVE_STORE_PATH", "artifacts/adaptive_offers.sqlite3")
    )
