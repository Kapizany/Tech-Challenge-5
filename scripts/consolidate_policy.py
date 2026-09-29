#!/usr/bin/env python3
"""Consolidate idempotent feedback into a review-only Thompson policy candidate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adaptive_offers.decision_store import configured_store  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "policy_update.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "policy_versions"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_time(value: str | datetime) -> datetime:
    parsed = (
        value
        if isinstance(value, datetime)
        else datetime.fromisoformat(value.replace("Z", "+00:00"))
    )
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp do feedback precisa incluir timezone")
    return parsed.astimezone(timezone.utc)


def _json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _git_sha() -> str | None:
    injected = os.getenv("GIT_SHA")
    if injected:
        return injected
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _state_totals(state: dict[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
    counts: dict[str, float] = {}
    successes: dict[str, float] = {}
    for per_action in state.get("online_counts", {}).values():
        for action, count in per_action.items():
            counts[str(action)] = counts.get(str(action), 0.0) + float(count)
    for per_action in state.get("online_successes", {}).values():
        for action, count in per_action.items():
            successes[str(action)] = successes.get(str(action), 0.0) + float(count)
    return counts, successes


def _drift_skip_reason(
    parent_state: dict[str, Any], eligible: list[dict[str, Any]], config: dict[str, Any]
) -> str | None:
    """Return a skip reason when the new window does not diverge from the parent."""

    counts, successes = _state_totals(parent_state)
    parent_trials = sum(counts.values())
    if parent_trials <= 0 or not eligible:
        return "sem histórico para medir deriva"
    parent_rate = sum(successes.values()) / parent_trials
    new_rate = sum(int(event["reward"]) for event in eligible) / len(eligible)
    if parent_rate - new_rate >= float(config.get("drift_reward_drop", 0.05)):
        return None
    share_limit = float(config.get("drift_action_share", 0.2))
    new_counts = {action: 0 for action in config["actions"]}
    for event in eligible:
        new_counts[str(event["action"])] = new_counts.get(str(event["action"]), 0) + 1
    for action in config["actions"]:
        parent_share = counts.get(action, 0.0) / parent_trials
        new_share = new_counts.get(action, 0) / len(eligible)
        if abs(parent_share - new_share) >= share_limit:
            return None
    return "deriva abaixo do limiar"


def _gs_bucket_and_prefix(uri: str) -> tuple[str, str]:
    bucket, _, prefix = uri.removeprefix("gs://").partition("/")
    return bucket, prefix.strip("/")


def restore_latest_candidate(output_dir: Path, uri: str) -> None:
    """Copy the newest persisted candidate from GCS so a fresh disk does not recount."""

    from google.cloud import storage

    bucket_name, prefix = _gs_bucket_and_prefix(uri)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    best_name = ""
    best_time = ""
    for blob in client.list_blobs(bucket, prefix=prefix):
        if not blob.name.endswith("/manifest.json"):
            continue
        try:
            generated_at = json.loads(blob.download_as_text()).get("generated_at", "")
        except (ValueError, OSError):
            continue
        state_blob = bucket.blob(f"{blob.name.rsplit('/', 1)[0]}/policy_state.json")
        if generated_at and state_blob.exists() and generated_at >= best_time:
            best_time = generated_at
            best_name = blob.name
    if not best_name:
        return
    version_prefix = best_name.rsplit("/", 1)[0]
    destination = output_dir / Path(version_prefix).name
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("manifest.json", "policy_state.json"):
        bucket.blob(f"{version_prefix}/{name}").download_to_filename(destination / name)


def upload_candidate(candidate_dir: Path, uri: str) -> None:
    from google.cloud import storage

    bucket_name, prefix = _gs_bucket_and_prefix(uri)
    bucket = storage.Client().bucket(bucket_name)
    # The manifest is the commit marker: never expose it before the state exists.
    for name in ("policy_state.json", "manifest.json"):
        path = candidate_dir / name
        blob_name = f"{prefix}/{candidate_dir.name}/{name}" if prefix else f"{candidate_dir.name}/{name}"
        bucket.blob(blob_name).upload_from_filename(path)


def _latest_candidate(output_dir: Path) -> tuple[dict[str, Any], dict[str, Any]] | None:
    candidates = []
    for manifest_path in output_dir.glob("policy-*/manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            state = json.loads(
                (manifest_path.parent / "policy_state.json").read_text(encoding="utf-8")
            )
            candidates.append((manifest.get("generated_at", ""), manifest, state))
        except (OSError, json.JSONDecodeError):
            continue
    if not candidates:
        return None
    _, manifest, state = max(candidates, key=lambda item: item[0])
    return manifest, state


def consolidate(
    events: list[dict[str, Any]],
    config: dict[str, Any],
    output_dir: Path,
    *,
    now: datetime | None = None,
    git_sha: str | None = None,
) -> dict[str, Any]:
    now = (now or _now()).astimezone(timezone.utc)
    trigger = config["trigger"]
    latest = _latest_candidate(output_dir)
    parent_manifest, parent_state = latest if latest else ({}, {})
    processed = set(parent_state.get("processed_request_ids", []))
    new_events = [event for event in events if event.get("request_id") not in processed]
    eligible = []
    ignored_missing_segment = 0
    for event in new_events:
        if not event.get("segment"):
            ignored_missing_segment += 1
            continue
        if event.get("action") not in config["actions"] or int(event.get("reward", -1)) not in (
            0,
            1,
        ):
            continue
        eligible.append(event)

    reason = None
    if trigger not in {"interval", "volume", "drift"}:
        raise ValueError("trigger deve ser interval, volume ou drift")
    if len(eligible) < int(config["min_new_rewards"]):
        reason = "abaixo do mínimo de novas recompensas"
    elif trigger == "interval" and latest:
        generated_at = _parse_time(parent_manifest["generated_at"])
        if now - generated_at < timedelta(days=int(config["interval_days"])):
            reason = "intervalo mínimo desde a última consolidação ainda não atingido"
    elif trigger == "drift":
        reason = _drift_skip_reason(parent_state, eligible, config)

    if reason:
        receipt = {
            "event": "policy_consolidation",
            "status": "skipped",
            "reason": reason,
            "trigger": trigger,
            "new_feedback": len(new_events),
            "eligible_feedback": len(eligible),
            "ignored_missing_segment": ignored_missing_segment,
            "generated_at": now.isoformat(),
        }
        _json(output_dir / "receipts" / f"skipped-{now.strftime('%Y%m%dT%H%M%S%fZ')}.json", receipt)
        return receipt

    state = {
        "schema_version": 1,
        "actions": list(config["actions"]),
        "prior_alpha": float(config["prior_alpha"]),
        "prior_beta": float(config["prior_beta"]),
        "model_strength": float(config["model_strength"]),
        "online_counts": copy.deepcopy(parent_state.get("online_counts", {})),
        "online_successes": copy.deepcopy(parent_state.get("online_successes", {})),
        "processed_request_ids": sorted(processed),
    }
    eligible.sort(key=lambda event: (_parse_time(event["timestamp"]), str(event["request_id"])))
    for event in eligible:
        segment = str(event["segment"])
        action = str(event["action"])
        reward = int(event["reward"])
        state["online_counts"].setdefault(segment, {item: 0 for item in state["actions"]})
        state["online_successes"].setdefault(segment, {item: 0 for item in state["actions"]})
        state["online_counts"][segment][action] += 1
        state["online_successes"][segment][action] += reward
        state["processed_request_ids"].append(str(event["request_id"]))
    state["processed_request_ids"] = sorted(set(state["processed_request_ids"]))

    version = f"policy-{now.strftime('%Y%m%dT%H%M%S%fZ')}-{uuid.uuid4().hex[:8]}"
    candidate_dir = output_dir / version
    candidate_dir.mkdir(parents=True, exist_ok=False)
    first_feedback = min(_parse_time(event["timestamp"]) for event in eligible)
    last_feedback = max(_parse_time(event["timestamp"]) for event in eligible)
    manifest = {
        "event": "policy_consolidation",
        "schema_version": 1,
        "status": "candidate_created",
        "version": version,
        "parent_version": parent_manifest.get("version", "thompson-sampling-contextual-v1"),
        "generated_at": now.isoformat(),
        "window": {"start": first_feedback.isoformat(), "end": last_feedback.isoformat()},
        "trigger": trigger,
        "new_feedback": len(new_events),
        "consolidated_feedback": len(eligible),
        "ignored_missing_segment": ignored_missing_segment,
        "git_sha": git_sha if git_sha is not None else _git_sha(),
        "config_sha256": _fingerprint(config),
        "feedback_sha256": _fingerprint(eligible),
        "promote": False,
        "promotion_status": "pending_review",
        "served": False,
        "posterior_active": False,
        "model_retrained": False,
    }
    _json(candidate_dir / "policy_state.json", state)
    _json(candidate_dir / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    for key in (
        "actions",
        "trigger",
        "interval_days",
        "min_new_rewards",
        "prior_alpha",
        "prior_beta",
        "model_strength",
    ):
        if key not in config:
            raise ValueError(f"config ausente: {key}")
    if (
        not config["actions"]
        or int(config["min_new_rewards"]) < 1
        or int(config["interval_days"]) < 1
    ):
        raise ValueError("ações, min_new_rewards e interval_days devem ser válidos")
    publish_uri = os.getenv("POLICY_VERSIONS_URI", "")
    if publish_uri:
        restore_latest_candidate(args.output_dir, publish_uri)
    store = configured_store()
    result = consolidate(store.feedback_events(), config, args.output_dir)
    if publish_uri and result["status"] == "candidate_created":
        upload_candidate(args.output_dir / result["version"], publish_uri)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
