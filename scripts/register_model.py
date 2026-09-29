#!/usr/bin/env python3
"""Record a model candidate. Vertex upload runs only after the offline gate passes."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.quality_gate import evaluate_gates  # noqa: E402

DEFAULT_REPORT = PROJECT_ROOT / "artifacts" / "phase5_evaluation_report.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "registry" / "candidate.json"


def registration_decision(report: dict) -> dict:
    failures = evaluate_gates(report)
    return {
        "event": "model_registration",
        "promotion_status": "blocked" if failures else "pending_review",
        "failures": failures,
        "registered": False,
    }


def upload_to_vertex(project: str, region: str, artifact_uri: str, image: str) -> str:
    from google.cloud import aiplatform

    aiplatform.init(project=project, location=region)
    model = aiplatform.Model.upload(
        display_name="adaptive-offers-channel",
        artifact_uri=artifact_uri,
        serving_container_image_uri=image,
        sync=True,
    )
    return model.resource_name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    decision = registration_decision(report)
    if not decision["failures"] and os.getenv("VERTEX_REGISTER") == "1":
        decision["registered"] = True
        decision["vertex_model"] = upload_to_vertex(
            os.environ["GOOGLE_CLOUD_PROJECT"],
            os.getenv("GCP_REGION", "southamerica-east1"),
            os.environ["MODEL_URI"],
            os.environ["TRAIN_IMAGE"],
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    return 1 if decision["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
