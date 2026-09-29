"""HTTP service that recommends a contact channel for one eligible client."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .baselines import default_segment
from .config import CATEGORICAL_VALUES, PROJECT_ROOT
from .decision_store import (
    DecisionConflict,
    DecisionStore,
    MemoryDecisionStore,
    UnknownDecision,
    configured_store,
)
from .policies import Guardrails, ThompsonSamplingPolicy
from .policy_state import apply_feedback_posterior, read_current, read_remote_current

MODEL_PATH = PROJECT_ROOT / "artifacts" / "channel_reward_model.joblib"
CATEGORICAL_FIELDS = (
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "month",
    "day_of_week",
    "poutcome",
)


class ClientContext(BaseModel):
    """Client attributes known before the contact. Duration and outcome are absent."""

    model_config = ConfigDict(populate_by_name=True)

    age: int = Field(ge=0, le=120)
    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    month: str
    day_of_week: str
    campaign: int = Field(
        ge=1, description="Contatos na campanha, incluindo o que está sendo decidido."
    )
    pdays: int = Field(description="Dias desde o contato anterior; 999 significa nunca contatado.")
    previous: int = Field(ge=0)
    poutcome: str
    emp_var_rate: float = Field(alias="emp.var.rate")
    cons_price_idx: float = Field(alias="cons.price.idx")
    cons_conf_idx: float = Field(alias="cons.conf.idx")
    euribor3m: float
    nr_employed: float = Field(alias="nr.employed")

    @field_validator(*CATEGORICAL_FIELDS)
    @classmethod
    def known_category(cls, value: str, info) -> str:
        allowed = CATEGORICAL_VALUES[info.field_name]
        if value not in allowed:
            raise ValueError(f"valor inválido para {info.field_name}")
        return value

    def as_frame(self) -> pd.DataFrame:
        payload = self.model_dump(by_alias=True)
        return pd.DataFrame([payload])


def recommend_client(
    reward_model,
    client: ClientContext,
    *,
    request_id: str | None = None,
    selection_counts: dict[str, int] | None = None,
    posterior_state: dict | None = None,
    posterior_version: str | None = None,
) -> dict:
    """Return one contextual Thompson Sampling decision for a client."""

    frame = client.as_frame()
    expected = reward_model.predict_all(frame).iloc[0]
    decision_id = request_id or str(uuid.uuid4())
    seed = int(hashlib.sha256(decision_id.encode("utf-8")).hexdigest()[:8], 16)
    policy = ThompsonSamplingPolicy(
        reward_model.actions,
        seed=seed,
        guardrails=Guardrails(),
        reward_model=reward_model,
    )
    if posterior_state is not None and posterior_version:
        apply_feedback_posterior(policy, posterior_state, posterior_version)
    policy.selection_counts.update(
        {action: int((selection_counts or {}).get(action, 0)) for action in policy.actions}
    )
    policy.bind_expected_rewards(pd.DataFrame([expected.to_dict()], index=frame.index))
    action = policy.select_action(frame.iloc[0], request_id=decision_id)
    decision = policy.last_decision or {}
    return {
        "request_id": decision_id,
        "action": action,
        "action_probability": float(decision.get("action_probability", 0.0)),
        "expected_reward_by_action": {name: float(expected[name]) for name in reward_model.actions},
        "policy_version": decision.get("policy_version", policy.policy_version),
        "reason": decision.get("reason", "posterior_sample"),
        "human_review_required": True,
        "note": (
            "A recomendação escolhe o canal de contato, não um produto. "
            "Decisão sensível exige revisão humana antes de qualquer acionamento."
        ),
    }


class FeedbackPayload(BaseModel):
    request_id: str = Field(pattern=r"^[A-Za-z0-9._:-]{1,128}$")
    reward: int = Field(ge=0, le=1, strict=True)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("timestamp")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp deve incluir timezone, por exemplo Z ou +00:00")
        return value.astimezone(timezone.utc)


class PubSubFeedbackPublisher:
    """Optional publisher used in cloud deployments after durable storage."""

    def __init__(self, project: str, topic: str):
        from google.cloud import pubsub_v1

        self._client = pubsub_v1.PublisherClient()
        self._topic_path = self._client.topic_path(project, topic)

    def publish(self, event: dict) -> None:
        future = self._client.publish(
            self._topic_path,
            json.dumps(event, separators=(",", ":")).encode("utf-8"),
            event_id=str(event["request_id"]),
        )
        future.result(timeout=10)


class BigQueryFeedbackSink:
    """Idempotent analytics sink for Pub/Sub push delivery."""

    def __init__(self, table_id: str):
        from google.cloud import bigquery

        self._client = bigquery.Client()
        self._table_id = table_id

    def write(self, event: dict) -> None:
        errors = self._client.insert_rows_json(
            self._table_id,
            [
                {
                    "request_id": event["request_id"],
                    "reward": int(event["reward"]),
                    "timestamp": event["timestamp"],
                    "received_at": datetime.now(timezone.utc).isoformat(),
                }
            ],
            row_ids=[str(event["request_id"])],
        )
        if errors:
            raise RuntimeError(f"BigQuery insert failed: {errors}")


def configured_feedback_publisher():
    topic = os.getenv("FEEDBACK_TOPIC")
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not topic or not project:
        return None
    return PubSubFeedbackPublisher(project, topic)


def configured_feedback_sink():
    table_id = os.getenv("FEEDBACK_BIGQUERY_TABLE")
    return BigQueryFeedbackSink(table_id) if table_id else None


def load_reward_model(path: Path = MODEL_PATH):
    model_uri = os.getenv("MODEL_URI")
    if model_uri and model_uri.startswith("gs://"):
        from google.cloud import storage

        bucket_name, object_name = model_uri[5:].split("/", 1)
        payload = storage.Client().bucket(bucket_name).blob(object_name).download_as_bytes()
        return joblib.load(io.BytesIO(payload))
    if model_uri:
        path = Path(model_uri)
    if not path.is_file():
        return None
    return joblib.load(path)


def _active_posterior(app: FastAPI) -> tuple[str, dict] | None:
    if not app.state.load_posterior:
        return None
    now = time.monotonic()
    cache = app.state.posterior_cache
    if cache["checked"] and now - cache["at"] < 30:
        return cache["value"]
    found = None
    uri = app.state.policy_versions_uri
    if uri:
        try:
            found = read_remote_current(uri)
        except Exception:
            found = None
    if found is None:
        found = read_current(app.state.policy_state_dir)
    cache["at"] = now
    cache["checked"] = True
    cache["value"] = found
    return found


def create_app(
    reward_model=None,
    decision_store: DecisionStore | None = None,
    feedback_publisher=None,
    feedback_sink=None,
    *,
    load_posterior: bool = False,
    policy_state_dir: Path | None = None,
    policy_versions_uri: str | None = None,
) -> FastAPI:
    app = FastAPI(title="Adaptive Offers", version="0.1.0")
    app.state.reward_model = reward_model
    app.state.decision_store = decision_store or MemoryDecisionStore()
    app.state.feedback_publisher = feedback_publisher
    app.state.feedback_sink = feedback_sink
    app.state.load_posterior = load_posterior
    app.state.policy_state_dir = policy_state_dir or (PROJECT_ROOT / "artifacts" / "policy_versions")
    app.state.policy_versions_uri = policy_versions_uri or ""
    app.state.posterior_cache = {"at": 0.0, "checked": False, "value": None}

    @app.middleware("http")
    async def add_request_observability(request: Request, call_next):
        supplied_id = request.headers.get("X-Request-ID", "")
        correlation_id = (
            supplied_id
            if re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", supplied_id)
            else str(uuid.uuid4())
        )
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = correlation_id
        response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - started) * 1000:.3f}"
        return response

    @app.get("/health")
    def health() -> dict:
        loaded = app.state.reward_model is not None
        try:
            store_ready = app.state.decision_store.health()
        except Exception:
            store_ready = False
        service_status = (
            "model_missing" if not loaded else "store_unavailable" if not store_ready else "ok"
        )
        return {
            "status": service_status,
            "model_loaded": loaded,
            "decision_store_ready": store_ready,
        }

    @app.get("/")
    def root() -> dict:
        current_health = health()
        if current_health["status"] != "ok":
            raise HTTPException(status_code=503, detail=current_health["status"])
        return current_health

    @app.get("/model-info")
    def model_info() -> dict:
        model = app.state.reward_model
        if model is None:
            raise HTTPException(
                status_code=503, detail="Modelo não carregado. Execute make train-policies."
            )
        posterior = _active_posterior(app)
        return {
            "actions": list(model.actions),
            "policy": posterior[0] if posterior else "thompson-sampling-contextual-v1",
            "feedback_posterior_active": posterior is not None,
            "context_enters_decision": True,
            "human_review_required": True,
        }

    @app.post("/recommend")
    def recommend(
        client: ClientContext,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> dict:
        model = app.state.reward_model
        if model is None:
            raise HTTPException(
                status_code=503, detail="Modelo não carregado. Execute make train-policies."
            )
        request_id = idempotency_key or str(uuid.uuid4())
        if not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", request_id):
            raise HTTPException(
                status_code=422,
                detail="Idempotency-Key deve conter 1 a 128 caracteres alfanuméricos ou . _ : -",
            )
        canonical = json.dumps(
            client.model_dump(by_alias=True), sort_keys=True, separators=(",", ":")
        )
        context_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        store = app.state.decision_store
        try:
            posterior = _active_posterior(app)
            response, _ = store.decide(
                request_id,
                context_hash,
                lambda counts: recommend_client(
                    model,
                    client,
                    request_id=request_id,
                    selection_counts=counts,
                    posterior_state=None if posterior is None else posterior[1],
                    posterior_version=None if posterior is None else posterior[0],
                ),
                segment=default_segment(client.as_frame().iloc[0].to_dict()),
            )
            return response
        except DecisionConflict as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/feedback")
    def feedback(payload: FeedbackPayload) -> dict:
        store = app.state.decision_store
        try:
            duplicate = store.record_feedback(payload.request_id, payload.reward, payload.timestamp)
        except UnknownDecision as error:
            raise HTTPException(status_code=404, detail="request_id não encontrado") from error
        except DecisionConflict as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        event = payload.model_dump(mode="json")
        publisher = app.state.feedback_publisher
        if publisher is not None:
            try:
                publisher.publish(event)
            except Exception as error:
                raise HTTPException(
                    status_code=503,
                    detail="Feedback persistido localmente, publicação assíncrona indisponível; tente novamente.",
                ) from error
        return {
            "request_id": payload.request_id,
            "status": "duplicate" if duplicate else "accepted",
        }

    @app.post("/events/feedback", status_code=status.HTTP_204_NO_CONTENT)
    def feedback_event(envelope: dict) -> None:
        """Pub/Sub push target. Cloud Run IAM protects this endpoint."""
        sink = app.state.feedback_sink
        if sink is None:
            raise HTTPException(status_code=503, detail="Feedback analytics sink not configured")
        try:
            encoded = envelope["message"]["data"]
            event = json.loads(base64.b64decode(encoded, validate=True))
            payload = FeedbackPayload.model_validate(event)
            sink.write(payload.model_dump(mode="json"))
        except HTTPException:
            raise
        except Exception as error:
            # Non-2xx causes Pub/Sub to retry and eventually dead-letter.
            raise HTTPException(
                status_code=400, detail="Malformed Pub/Sub feedback event"
            ) from error
        return None

    return app


app = create_app(
    load_reward_model(),
    decision_store=configured_store(),
    feedback_publisher=configured_feedback_publisher(),
    feedback_sink=configured_feedback_sink(),
    load_posterior=True,
    policy_versions_uri=os.getenv("POLICY_VERSIONS_URI") or None,
)
