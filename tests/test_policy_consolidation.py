import json
from datetime import datetime, timedelta, timezone

from scripts.consolidate_policy import consolidate


def _config(**overrides):
    return {
        "trigger": "interval",
        "interval_days": 7,
        "min_new_rewards": 2,
        "actions": ["cellular", "telephone"],
        "prior_alpha": 1.0,
        "prior_beta": 1.0,
        "model_strength": 40.0,
        "promote": False,
        **overrides,
    }


def _event(
    request_id,
    action,
    reward,
    timestamp="2026-09-01T12:00:00+00:00",
    segment="poutcome=failure|contacts_before=0|housing=yes",
):
    return {
        "request_id": request_id,
        "action": action,
        "reward": reward,
        "timestamp": timestamp,
        "segment": segment,
    }


def test_consolidation_writes_candidate_once_and_never_promotes(tmp_path):
    now = datetime(2026, 9, 28, tzinfo=timezone.utc)
    events = [
        _event("r1", "cellular", 1),
        _event("r2", "telephone", 0, timestamp="2026-09-02T12:00:00+00:00"),
    ]

    result = consolidate(events, _config(), tmp_path, now=now, git_sha="abc123")

    assert result["status"] == "candidate_created"
    assert result["promote"] is False
    assert result["promotion_status"] == "pending_review"
    candidate = tmp_path / result["version"]
    manifest = json.loads((candidate / "manifest.json").read_text())
    state = json.loads((candidate / "policy_state.json").read_text())
    segment = events[0]["segment"]
    assert manifest["git_sha"] == "abc123"
    assert state["online_counts"][segment] == {"cellular": 1, "telephone": 1}
    assert state["online_successes"][segment] == {"cellular": 1, "telephone": 0}
    assert state["processed_request_ids"] == ["r1", "r2"]
    assert not (tmp_path / "current.json").exists()
    assert manifest["posterior_active"] is False

    repeated = consolidate(events, _config(), tmp_path, now=now + timedelta(days=8))
    assert repeated["status"] == "skipped"
    assert repeated["new_feedback"] == 0


def test_consolidation_skips_small_sample_and_drift_without_baseline(tmp_path):
    now = datetime(2026, 9, 28, tzinfo=timezone.utc)
    small = consolidate([_event("r1", "cellular", 1)], _config(), tmp_path, now=now)
    assert small["status"] == "skipped"
    assert small["reason"] == "abaixo do mínimo de novas recompensas"

    drift = consolidate(
        [_event("r1", "cellular", 1), _event("r2", "telephone", 0)],
        _config(trigger="drift"),
        tmp_path,
        now=now,
    )
    assert drift["status"] == "skipped"
    assert drift["reason"] == "sem histórico para medir deriva"


def test_volume_ignores_the_weekly_interval(tmp_path):
    now = datetime(2026, 9, 28, tzinfo=timezone.utc)
    first = [
        _event("r1", "cellular", 1),
        _event("r2", "telephone", 0, timestamp="2026-09-02T12:00:00+00:00"),
    ]
    created = consolidate(first, _config(), tmp_path, now=now, git_sha="abc")
    assert created["status"] == "candidate_created"
    early = first + [
        _event("r3", "cellular", 1, timestamp="2026-09-03T12:00:00+00:00"),
        _event("r4", "telephone", 1, timestamp="2026-09-04T12:00:00+00:00"),
    ]
    result = consolidate(early, _config(trigger="volume"), tmp_path, now=now + timedelta(days=1), git_sha="def")
    assert result["status"] == "candidate_created"
    assert result["parent_version"] == created["version"]
    assert result["promote"] is False


def test_drift_fires_only_when_the_reward_rate_drops(tmp_path):
    now = datetime(2026, 9, 28, tzinfo=timezone.utc)
    baseline = [
        _event("r1", "cellular", 1),
        _event("r2", "telephone", 1, timestamp="2026-09-02T12:00:00+00:00"),
    ]
    consolidate(baseline, _config(), tmp_path, now=now, git_sha="abc")
    stable = baseline + [
        _event("r3", "cellular", 1, timestamp="2026-09-03T12:00:00+00:00"),
        _event("r4", "telephone", 1, timestamp="2026-09-04T12:00:00+00:00"),
    ]
    unchanged = consolidate(
        stable, _config(trigger="drift"), tmp_path, now=now + timedelta(days=1), git_sha="def"
    )
    assert unchanged["status"] == "skipped"
    assert unchanged["reason"] == "deriva abaixo do limiar"

    dropped = baseline + [
        _event("r5", "cellular", 0, timestamp="2026-09-05T12:00:00+00:00"),
        _event("r6", "telephone", 0, timestamp="2026-09-06T12:00:00+00:00"),
    ]
    shifted = consolidate(
        dropped, _config(trigger="drift"), tmp_path, now=now + timedelta(days=1), git_sha="ghi"
    )
    assert shifted["status"] == "candidate_created"
    assert shifted["promote"] is False
