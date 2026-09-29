#!/usr/bin/env python3
"""Recommend a contact channel for one client JSON payload."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adaptive_offers.api import ClientContext, load_reward_model, recommend_client

EXAMPLE = {
    "age": 33,
    "job": "admin.",
    "marital": "married",
    "education": "university.degree",
    "default": "no",
    "housing": "yes",
    "loan": "no",
    "month": "may",
    "day_of_week": "mon",
    "campaign": 1,
    "pdays": 999,
    "previous": 0,
    "poutcome": "nonexistent",
    "emp.var.rate": 1.1,
    "cons.price.idx": 93.994,
    "cons.conf.idx": -36.4,
    "euribor3m": 4.857,
    "nr.employed": 5191.0,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="JSON com um cliente. Sem o argumento, usa um exemplo.")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "artifacts" / "channel_reward_model.joblib")
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8")) if args.input else EXAMPLE
    model = load_reward_model(args.model)
    if model is None:
        raise SystemExit(f"Modelo não encontrado em {args.model}. Execute make train-policies.")
    print(json.dumps(recommend_client(model, ClientContext.model_validate(payload)), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
