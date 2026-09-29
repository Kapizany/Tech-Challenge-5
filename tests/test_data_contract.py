import pandas as pd
import pytest

from adaptive_offers.config import COLUMNS
from adaptive_offers.data import DataContractError, validate_schema


def valid_frame(rows: int = 2) -> pd.DataFrame:
    values = {
        "age": 35,
        "job": "admin.",
        "marital": "married",
        "education": "university.degree",
        "default": "no",
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "month": "may",
        "day_of_week": "mon",
        "duration": 100,
        "campaign": 1,
        "pdays": -1,
        "previous": 0,
        "poutcome": "nonexistent",
        "emp.var.rate": 1.1,
        "cons.price.idx": 93.9,
        "cons.conf.idx": -36.4,
        "euribor3m": 4.8,
        "nr.employed": 5191.0,
        "y": "no",
    }
    return pd.DataFrame([values] * rows, columns=COLUMNS)


def test_contract_accepts_development_sample() -> None:
    report = validate_schema(valid_frame(), strict=False, expected_rows=None)
    assert report["valid"] is True
    assert report["columns"] == 21


def test_contract_rejects_unknown_action() -> None:
    frame = valid_frame()
    frame.loc[0, "contact"] = "sms"
    with pytest.raises(DataContractError, match="contact"):
        validate_schema(frame, strict=False, expected_rows=None)


def test_contract_rejects_missing_column() -> None:
    frame = valid_frame().drop(columns=["duration"])
    with pytest.raises(DataContractError, match="colunas incompatíveis"):
        validate_schema(frame, strict=False, expected_rows=None)
