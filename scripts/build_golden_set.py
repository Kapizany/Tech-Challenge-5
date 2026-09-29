#!/usr/bin/env python3
"""Build the five-case Golden Set and recommendation explanations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adaptive_offers.config import DEFAULT_DATA_PATH
from adaptive_offers.data import load_bank_marketing, validate_schema
from adaptive_offers.golden_set import build_golden_set
from adaptive_offers.policies import Guardrails, ThompsonSamplingPolicy
from adaptive_offers.preparation import modeling_frame, temporal_split
from adaptive_offers.reward_model import ChannelRewardModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts" / "golden_set.json")
    args = parser.parse_args()

    raw = load_bank_marketing(args.input)
    validate_schema(raw)
    splits = temporal_split(modeling_frame(raw))
    model = ChannelRewardModel.fit(splits.train)
    policy = ThompsonSamplingPolicy(
        model.actions, seed=42, guardrails=Guardrails(), reward_model=model,
    ).fit(splits.train)
    records = build_golden_set(splits.test, model, policy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(f"Golden Set gerado com {len(records)} casos: {args.output}")


if __name__ == "__main__":
    main()
