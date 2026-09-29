#!/usr/bin/env python3
"""Write the EDA tables used by the README and the notebook."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adaptive_offers.config import DEFAULT_DATA_PATH
from adaptive_offers.data import load_bank_marketing, validate_schema
from adaptive_offers.eda import (
    campaign_fatigue,
    channel_by_euribor,
    channel_by_month,
    channel_by_period,
    conversion_by,
    duration_leakage_comparison,
    macro_correlation,
    unknown_rates,
)


def _records(frame):
    return json.loads(frame.to_json(orient="records", force_ascii=False))


def main() -> None:
    frame = load_bank_marketing(DEFAULT_DATA_PATH)
    validate_schema(frame)
    destination = PROJECT_ROOT / "artifacts"
    destination.mkdir(parents=True, exist_ok=True)
    summary = {
        "rows": int(len(frame)),
        "duplicate_rows": int(frame.duplicated().sum()),
        "conversion_rate": float((frame["y"] == "yes").mean()),
        "default_yes": int((frame["default"] == "yes").sum()),
        "pdays_sentinel_rows": int((frame["pdays"] >= 999).sum()),
        "by_contact": _records(conversion_by(frame, "contact")),
        "by_job": _records(conversion_by(frame, "job")),
        "by_month": _records(conversion_by(frame, "month")),
        "by_poutcome": _records(conversion_by(frame, "poutcome")),
        "campaign_fatigue": _records(campaign_fatigue(frame)),
        "channel_by_month": _records(channel_by_month(frame)),
        "channel_by_period": _records(channel_by_period(frame)),
        "channel_by_euribor": _records(channel_by_euribor(frame)),
        "unknown_rates": _records(unknown_rates(frame)),
        "macro_correlation": json.loads(macro_correlation(frame).to_json()),
        "duration_leakage": _records(duration_leakage_comparison(frame)),
    }
    path = destination / "eda_summary.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"EDA salva em {path}")


if __name__ == "__main__":
    main()
