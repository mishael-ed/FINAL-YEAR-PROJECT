from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from .schema import COLUMN_ALIASES, OPTIONAL_COLUMNS, PREDICT_REQUIRED_COLUMNS, TRAIN_REQUIRED_COLUMNS


@dataclass
class ValidationResult:
    ok: bool
    missing: list[str]
    mode: str


def load_table(path):
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    if p.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(p)
    raise ValueError("File must be CSV or Excel")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapper = {}
    for c in df.columns:
        key = c.strip().lower().replace(" ", "_")
        mapper[c] = COLUMN_ALIASES.get(key, c)
    return df.rename(columns=mapper) 


def validate(df: pd.DataFrame, mode: str) -> ValidationResult:
    work = normalize_columns(df)
    required = TRAIN_REQUIRED_COLUMNS if mode == "train" else PREDICT_REQUIRED_COLUMNS
    missing = [c for c in required if c not in work.columns]
    return ValidationResult(ok=not missing, missing=missing, mode=mode)


def prepare(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    work = normalize_columns(df)
    vr = validate(work, mode)
    if not vr.ok:
        raise ValueError(f"Missing required columns ({mode}): {', '.join(vr.missing)}")
    required = TRAIN_REQUIRED_COLUMNS if mode == "train" else PREDICT_REQUIRED_COLUMNS
    optional_present = [c for c in OPTIONAL_COLUMNS if c in work.columns and c not in required]
    return work[required + optional_present].copy()
