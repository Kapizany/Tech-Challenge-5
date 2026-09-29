"""Data loading, contract validation and provenance helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import (
    CATEGORICAL_VALUES,
    COLUMNS,
    EXPECTED_COLUMNS,
    EXPECTED_ROWS,
    TARGET_COLUMN,
)


class DataContractError(ValueError):
    """Raised when the downloaded dataset does not satisfy the project contract."""


def load_bank_marketing(path: str | Path) -> pd.DataFrame:
    """Load the UCI/Kaggle ``bank-additional-full.csv`` representation."""

    data_path = Path(path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Arquivo de dados não encontrado: {data_path}")
    frame = pd.read_csv(data_path, sep=";", dtype=str)
    for column in ["age", "duration", "campaign", "pdays", "previous"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in ["emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _missing(frame: pd.DataFrame) -> dict[str, int]:
    return {key: int(value) for key, value in frame.isna().sum().items() if value}


def _quality_warnings(frame: pd.DataFrame) -> list[str]:
    warnings = [
        "duration é mantida para EDA e deve ser removida antes do modelo: só é conhecida após o contato.",
        "contact representa a ação observada e não deve entrar como feature de contexto no treino do bandit.",
    ]
    if "pdays" in frame.columns:
        pdays = pd.to_numeric(frame["pdays"], errors="coerce")
        sentinel = int(((pdays >= 999) | (pdays < 0)).sum())
        if sentinel:
            warnings.append(
                "pdays usa 999 neste arquivo, ou valor negativo na outra extração UCI, "
                f"como sentinela de nunca contatado ({sentinel} linhas). "
                "O modelo usa never_contacted e pdays_contacted, não a distância bruta."
            )
    if "campaign" in frame.columns:
        warnings.append(
            "campaign inclui o contato atual. No instante da decisão o histórico conhecido é campaign - 1."
        )
    if "default" in frame.columns and len(frame):
        default_yes = int((frame["default"].astype(str) == "yes").sum())
        if default_yes <= 5:
            warnings.append(
                f"default='yes' aparece {default_yes} vezes; a coluna quase não varia e não deve sustentar uma regra."
            )
    if frame.duplicated().sum():
        warnings.append(
            f"há {int(frame.duplicated().sum())} linhas duplicadas; a preparação mantém a primeira ocorrência."
        )
    return warnings


def validate_schema(
    frame: pd.DataFrame,
    *,
    strict: bool = True,
    expected_rows: int | None = EXPECTED_ROWS,
) -> dict[str, Any]:
    """Validate shape, values and basic ranges, returning a serialisable report.

    ``strict=False`` is useful for unit tests and sampled development data. Production
    validation keeps ``strict=True`` and checks the expected 41,188 rows.
    """

    errors: list[str] = []
    if list(frame.columns) != COLUMNS:
        missing = [column for column in COLUMNS if column not in frame.columns]
        extra = [column for column in frame.columns if column not in COLUMNS]
        errors.append(f"colunas incompatíveis (faltantes={missing}, extras={extra})")
    if len(frame.columns) != EXPECTED_COLUMNS:
        errors.append(f"quantidade de colunas esperada={EXPECTED_COLUMNS}, observada={len(frame.columns)}")
    if strict and expected_rows is not None and len(frame) != expected_rows:
        errors.append(f"quantidade de linhas esperada={expected_rows}, observada={len(frame)}")

    if not frame.empty:
        missing = _missing(frame)
        if missing:
            errors.append(f"valores ausentes: {missing}")
        for column, allowed in CATEGORICAL_VALUES.items():
            if column in frame:
                observed = set(frame[column].dropna().astype(str).unique())
                invalid = sorted(observed - allowed)
                if invalid:
                    errors.append(f"valores inválidos em {column}: {invalid}")
        ranges = {
            "age": (0, 120),
            "duration": (0, None),
            "campaign": (1, None),
            "pdays": (-1, None),
            "previous": (0, None),
        }
        for column, (lower, upper) in ranges.items():
            if column not in frame:
                continue
            values = pd.to_numeric(frame[column], errors="coerce")
            if values.isna().any():
                errors.append(f"valores não numéricos em {column}")
            if lower is not None and (values < lower).any():
                errors.append(f"valor abaixo do limite em {column}: {lower}")
            if upper is not None and (values > upper).any():
                errors.append(f"valor acima do limite em {column}: {upper}")

    report: dict[str, Any] = {
        "valid": not errors,
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "column_names": list(frame.columns),
        "missing_values": _missing(frame),
        "duplicate_rows": int(frame.duplicated().sum()),
        "target_distribution": frame[TARGET_COLUMN].value_counts(dropna=False).to_dict()
        if TARGET_COLUMN in frame
        else {},
        "conversion_rate": float((frame[TARGET_COLUMN] == "yes").mean())
        if TARGET_COLUMN in frame and len(frame)
        else None,
        "channel_distribution": frame["contact"].value_counts(dropna=False).to_dict()
        if "contact" in frame
        else {},
        "errors": errors,
        "warnings": _quality_warnings(frame),
    }
    if errors:
        raise DataContractError("Contrato de dados inválido: " + " | ".join(errors))
    return report


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file for dataset provenance."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_report(report: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
