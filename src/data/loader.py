"""Load and validate the daily liquidity time series."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

NUMERIC_COLUMNS = ["Income", "Outcome", "Balance", "tax_day", "IMICEX", "TransRUB1M"]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names."""
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def _coerce_numeric(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        return pd.to_numeric(
            series.astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        )
    return pd.to_numeric(series, errors="coerce")


def load_raw_data(path: str | Path, date_column: str = "Date") -> pd.DataFrame:
    """Read CSV, parse dates, and normalize numeric columns."""
    df = pd.read_csv(path)
    if df.columns[0].startswith("Unnamed"):
        df = df.drop(columns=df.columns[0])

    df = normalize_columns(df)
    df[date_column] = pd.to_datetime(df[date_column])
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = _coerce_numeric(df[col])

    for col in ["covid", "IsDayOff_Status_Workalendar_RU"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    return df


def validate_date_index(df: pd.DataFrame, date_column: str = "Date") -> pd.DataFrame:
    """Sort by date, drop duplicate dates, keep deterministic order."""
    out = df.sort_values(date_column).reset_index(drop=True)
    dup_count = out[date_column].duplicated().sum()
    if dup_count:
        out = out.drop_duplicates(subset=[date_column], keep="last").reset_index(drop=True)
    return out


def add_active_flag(
    df: pd.DataFrame,
    income_col: str = "Income",
    outcome_col: str = "Outcome",
) -> pd.DataFrame:
    """Mark business-active days (non-zero inflows or outflows)."""
    out = df.copy()
    out["is_active"] = (out[income_col].fillna(0) + out[outcome_col].fillna(0)) > 0
    return out


def build_data_quality_report(df: pd.DataFrame, date_column: str = "Date") -> dict[str, Any]:
    """Summarize dataset quality for pipeline logging."""
    dates = pd.to_datetime(df[date_column])
    missing_by_col = df.isna().sum().to_dict()
    duplicate_dates = int(dates.duplicated().sum())
    date_gaps = int((dates.diff().dt.days > 1).sum())

    return {
        "rows": len(df),
        "date_min": str(dates.min().date()),
        "date_max": str(dates.max().date()),
        "duplicate_dates": duplicate_dates,
        "calendar_gaps": date_gaps,
        "active_days": int(df.get("is_active", pd.Series(dtype=bool)).sum()),
        "missing_by_column": {k: int(v) for k, v in missing_by_col.items() if v > 0},
    }
