#!/usr/bin/env python3
"""Create leakage-safe temporal splits and fitted context features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adaptive_offers.config import DEFAULT_DATA_PATH
from adaptive_offers.data import load_bank_marketing, validate_schema
from adaptive_offers.preparation import prepare_temporal_data, save_preparation_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--without-proxies", action="store_true")
    args = parser.parse_args()

    frame = load_bank_marketing(args.input)
    validate_schema(frame)
    prepared = prepare_temporal_data(frame, include_proxies=not args.without_proxies)
    manifest = save_preparation_artifacts(prepared, args.output_dir)
    print(
        "Preparação concluída: "
        f"train={manifest['split']['train']['rows']:,}, "
        f"validation={manifest['split']['validation']['rows']:,}, "
        f"test={manifest['split']['test']['rows']:,}, "
        f"features={manifest['feature_count']}"
    )
    print(f"Artefatos: {args.output_dir}")


if __name__ == "__main__":
    main()
