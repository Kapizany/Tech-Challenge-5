#!/usr/bin/env python3
"""Validate the dataset and write provenance/quality artefacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow this script to run directly from a fresh clone before editable install.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adaptive_offers.config import DEFAULT_DATA_PATH, DEFAULT_HASH_PATH, DEFAULT_REPORT_PATH
from adaptive_offers.data import load_bank_marketing, sha256_file, validate_schema, write_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--hash-output", type=Path, default=DEFAULT_HASH_PATH)
    args = parser.parse_args()
    frame = load_bank_marketing(args.input)
    report = validate_schema(frame)
    digest = sha256_file(args.input)
    report["sha256"] = digest
    report["source_file"] = str(args.input)
    write_report(report, args.report)
    args.hash_output.parent.mkdir(parents=True, exist_ok=True)
    args.hash_output.write_text(f"{digest}  {args.input.name}\n", encoding="utf-8")
    print(f"Contrato válido: {len(frame):,} linhas; conversão={report['conversion_rate']:.4f}; sha256={digest}")


if __name__ == "__main__":
    main()
