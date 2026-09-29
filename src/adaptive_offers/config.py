"""Project constants shared by acquisition, validation and notebooks."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_DATA_PATH = RAW_DATA_DIR / "bank-additional-full.csv"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "artifacts" / "data_quality_report.json"
DEFAULT_HASH_PATH = PROJECT_ROOT / "artifacts" / "data.sha256"

KAGGLE_DATASET = "henriqueyamahata/bank-marketing"
UCI_ARCHIVE_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"

EXPECTED_ROWS = 41_188
EXPECTED_COLUMNS = 21
TARGET_COLUMN = "y"
ACTION_COLUMN = "contact"

COLUMNS = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
    "y",
]

CATEGORICAL_VALUES = {
    "job": {
        "admin.",
        "blue-collar",
        "entrepreneur",
        "housemaid",
        "management",
        "retired",
        "self-employed",
        "services",
        "student",
        "technician",
        "unemployed",
        "unknown",
    },
    "marital": {"divorced", "married", "single", "unknown"},
    "education": {
        "basic.4y",
        "basic.6y",
        "basic.9y",
        "high.school",
        "illiterate",
        "professional.course",
        "university.degree",
        "unknown",
    },
    "default": {"yes", "no", "unknown"},
    "housing": {"yes", "no", "unknown"},
    "loan": {"yes", "no", "unknown"},
    "contact": {"cellular", "telephone"},
    "month": {"jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"},
    "day_of_week": {"mon", "tue", "wed", "thu", "fri"},
    "poutcome": {"failure", "nonexistent", "success"},
    "y": {"yes", "no"},
}
