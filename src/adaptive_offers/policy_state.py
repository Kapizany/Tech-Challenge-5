"""Load only an approved feedback posterior for recommendation serving."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .policies import ThompsonSamplingPolicy


def publish_current(
    output_dir: Path, version: str, state: dict[str, Any], generated_at: str,
    *, approved_by: str, approved_at: str,
) -> Path:
    path = output_dir / "current.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {"version": version, "generated_at": generated_at, "policy_state": state,
             "promotion_status": "approved", "approved_by": approved_by, "approved_at": approved_at},
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def read_current(directory: Path) -> tuple[str, dict[str, Any]] | None:
    path = directory / "current.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("promotion_status") != "approved" or not payload.get("approved_by"):
        return None
    return str(payload["version"]), payload["policy_state"]


def read_remote_current(uri: str) -> tuple[str, dict[str, Any]] | None:
    from google.cloud import storage

    bucket_name, _, prefix = uri.removeprefix("gs://").partition("/")
    blob_name = f"{prefix.strip('/')}/current.json" if prefix.strip("/") else "current.json"
    blob = storage.Client().bucket(bucket_name).blob(blob_name)
    if not blob.exists():
        return None
    payload = json.loads(blob.download_as_text())
    if payload.get("promotion_status") != "approved" or not payload.get("approved_by"):
        return None
    return str(payload["version"]), payload["policy_state"]


def apply_feedback_posterior(policy: ThompsonSamplingPolicy, state: dict[str, Any], version: str) -> None:
    """Add consolidated feedback counts to the Beta posterior used at decision time."""

    policy.online_counts = copy.deepcopy(state.get("online_counts") or {})
    policy.online_successes = copy.deepcopy(state.get("online_successes") or {})
    policy.policy_version = version
