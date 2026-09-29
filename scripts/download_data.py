#!/usr/bin/env python3
"""Download the selected Kaggle dataset or its official UCI source."""

from __future__ import annotations

import argparse
import io
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

# Allow this script to run directly from a fresh clone before editable
# installation (``make install`` remains recommended for the full project).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adaptive_offers.config import DEFAULT_DATA_PATH, KAGGLE_DATASET, UCI_ARCHIVE_URL


def _find_csv(directory: Path) -> Path:
    candidates = list(directory.rglob("bank-additional-full.csv"))
    if not candidates:
        candidates = list(directory.rglob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"Nenhum CSV encontrado em {directory}")
    return candidates[0]


def download_kaggle(destination: Path, dataset: str) -> None:
    with tempfile.TemporaryDirectory(prefix="kaggle-bank-marketing-") as temp_dir:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset, "-p", temp_dir, "--unzip"],
            check=True,
        )
        shutil.copy2(_find_csv(Path(temp_dir)), destination)


def download_uci(destination: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="uci-bank-marketing-") as temp_dir:
        archive = Path(temp_dir) / "bank-marketing.zip"
        urllib.request.urlretrieve(UCI_ARCHIVE_URL, archive)
        with zipfile.ZipFile(archive) as outer_zip:
            # The UCI endpoint currently returns an archive containing two
            # archives (bank.zip and bank-additional.zip). Older mirrors may
            # expose the CSV directly, so support both layouts.
            direct_members = [
                name for name in outer_zip.namelist() if name.endswith("bank-additional-full.csv")
            ]
            if direct_members:
                with outer_zip.open(direct_members[0]) as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target)
                return

            nested_name = next(
                (name for name in outer_zip.namelist() if name.endswith("bank-additional.zip")),
                None,
            )
            if nested_name is None:
                raise FileNotFoundError(
                    "O ZIP da UCI não contém bank-additional-full.csv nem bank-additional.zip"
                )
            nested_bytes = outer_zip.read(nested_name)

        with zipfile.ZipFile(io.BytesIO(nested_bytes)) as nested_zip:
            member = next(
                (
                    name
                    for name in nested_zip.namelist()
                    if name.endswith("bank-additional-full.csv")
                ),
                None,
            )
            if member is None:
                raise FileNotFoundError(
                    "bank-additional.zip não contém bank-additional-full.csv"
                )
            with nested_zip.open(member) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["kaggle", "uci"], default="uci")
    parser.add_argument("--dataset", default=KAGGLE_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_DATA_PATH)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.source == "kaggle":
        download_kaggle(args.output, args.dataset)
    else:
        download_uci(args.output)
    print(f"Dados gravados em {args.output}")


if __name__ == "__main__":
    main()
