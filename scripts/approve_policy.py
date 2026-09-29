#!/usr/bin/env python3
"""Approve a consolidated policy candidate after the evaluation gate passes."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adaptive_offers.policy_state import publish_current  # noqa: E402
from scripts.quality_gate import evaluate_gates  # noqa: E402

DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "policy_versions"
DEFAULT_REPORT = PROJECT_ROOT / "artifacts" / "phase5_evaluation_report.json"


def _remote_parts(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError("POLICY_VERSIONS_URI deve começar com gs://")
    bucket, _, prefix = uri[5:].partition("/")
    if not bucket:
        raise ValueError("bucket GCS ausente")
    return bucket, prefix.strip("/")


def _remote_name(prefix: str, suffix: str) -> str:
    return f"{prefix}/{suffix}" if prefix else suffix


def download_candidate(version: str, output_dir: Path, uri: str) -> None:
    from google.cloud import storage

    bucket_name, prefix = _remote_parts(uri)
    bucket = storage.Client().bucket(bucket_name)
    candidate_dir = output_dir / version
    candidate_dir.mkdir(parents=True, exist_ok=True)
    for name in ("manifest.json", "policy_state.json"):
        bucket.blob(_remote_name(prefix, f"{version}/{name}")).download_to_filename(
            candidate_dir / name
        )


def approve(
    version: str,
    approved_by: str,
    report: dict[str, Any],
    output_dir: Path,
    *,
    now: datetime | None = None,
    policy_uri: str | None = None,
) -> dict[str, Any]:
    if not re.fullmatch(r"policy-[0-9]{8}T[0-9]{12}Z-[0-9a-f]{8}", version):
        raise ValueError("versão da política inválida")
    if not approved_by.strip():
        raise ValueError("approved_by é obrigatório")
    failures = evaluate_gates(report)
    if failures:
        raise ValueError("quality gate bloqueou aprovação: " + "; ".join(failures))

    if policy_uri:
        download_candidate(version, output_dir, policy_uri)
    candidate_dir = output_dir / version
    manifest = json.loads((candidate_dir / "manifest.json").read_text(encoding="utf-8"))
    state = json.loads((candidate_dir / "policy_state.json").read_text(encoding="utf-8"))
    if manifest.get("version") != version or manifest.get("status") != "candidate_created":
        raise ValueError("manifesto não corresponde a um candidato válido")
    if manifest.get("promotion_status") != "pending_review" or manifest.get("served"):
        raise ValueError("candidato já foi servido ou não está pendente de revisão")
    if manifest.get("git_sha") and report.get("git_sha") != manifest["git_sha"]:
        raise ValueError("relatório de avaliação não corresponde ao código do candidato")
    actions = set(state.get("actions", []))
    if not actions or actions != {"cellular", "telephone"}:
        raise ValueError("ações do posterior inválidas")
    if not isinstance(state.get("processed_request_ids"), list):
        raise ValueError("lista de feedback processado ausente")
    for segment, counts in state.get("online_counts", {}).items():
        successes = state.get("online_successes", {}).get(segment, {})
        if set(counts) != actions or set(successes) != actions:
            raise ValueError("posterior incompleto")
        for action in actions:
            if not 0 <= int(successes[action]) <= int(counts[action]):
                raise ValueError("sucessos fora do intervalo de tentativas")

    timestamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="approval-", dir=output_dir) as temporary:
        staged = publish_current(
            Path(temporary), version, state, manifest["generated_at"],
            approved_by=approved_by.strip(), approved_at=timestamp,
        )
        if policy_uri:
            from google.cloud import storage

            bucket_name, prefix = _remote_parts(policy_uri)
            bucket = storage.Client().bucket(bucket_name)
            bucket.blob(_remote_name(prefix, "current.json")).upload_from_filename(staged)
        staged.replace(output_dir / "current.json")
    approval = {
        "version": version,
        "promotion_status": "approved",
        "approved_by": approved_by.strip(),
        "approved_at": timestamp,
        "report_git_sha": report.get("git_sha"),
    }
    approval_dir = output_dir / "approvals"
    approval_dir.mkdir(exist_ok=True)
    (approval_dir / f"{version}.json").write_text(
        json.dumps(approval, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return approval


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--approved-by", required=True)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--policy-uri", default=os.getenv("POLICY_VERSIONS_URI", ""))
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    result = approve(
        args.version, args.approved_by, report, args.output_dir,
        policy_uri=args.policy_uri or None,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
