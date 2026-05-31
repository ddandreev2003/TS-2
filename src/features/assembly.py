"""Feature matrix assembly and supervised dataset preparation.

ПОЧЕМУ: единая точка сборки признаков гарантирует одинаковый набор колонок
        для FS, обучения, eval и serving.
КАК: последовательно мёрджим autoregressive, calendar, tax, macro блоки;
     prepare_supervised отфильтровывает неактивные дни и выставляет Date как index.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.features.autoregressive import build_autoregressive_features
from src.features.calendar import build_calendar_features, build_tax_features
from src.features.macro import build_macro_features

META_COLUMNS = {"Date", "Balance", "Income", "Outcome", "is_active"}


def assemble_feature_matrix(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Merge autoregressive, calendar, tax, and macro feature frames."""
    ar = build_autoregressive_features(df, config)
    cal = build_calendar_features(df, config)
    tax = build_tax_features(df, config)
    macro = build_macro_features(df, config)

    merged = df[["Date"]].copy()
    for part in (ar, cal, tax, macro):
        feature_cols = [c for c in part.columns if c != "Date"]
        merged = merged.merge(part[["Date", *feature_cols]], on="Date", how="left")

    if "is_active" in df.columns:
        merged["is_active"] = df["is_active"].values
    for col in ("Balance", "Income", "Outcome"):
        if col in df.columns:
            merged[col] = df[col].values

    return merged


def prepare_supervised(
    feature_df: pd.DataFrame,
    target_col: str,
    active_only: bool = True,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Filter active days, drop NA rows, and return X, y, feature names."""
    work = feature_df.copy()
    if active_only and "is_active" in work.columns:
        work = work[work["is_active"]]

    if "Date" in work.columns:
        work = work.copy()
        work.index = pd.to_datetime(work["Date"])

    y = work[target_col].astype(float)
    feature_names = [
        c
        for c in work.columns
        if c not in META_COLUMNS and c != target_col and not c.startswith("Unnamed")
    ]
    X = work[feature_names].copy()
    valid = X.notna().all(axis=1) & y.notna()
    return X.loc[valid], y.loc[valid], feature_names


def get_feature_groups(columns: list[str]) -> dict[str, list[str]]:
    """Classify features into autoregressive, calendar, tax, and macro groups."""
    groups: dict[str, list[str]] = {
        "autoregressive": [],
        "calendar": [],
        "tax": [],
        "macro": [],
        "other": [],
    }
    for col in columns:
        lower = col.lower()
        if any(token in lower for token in ("lag", "roll", "ewm", "diff", "io_", "volatility", "_sign", "_abs")):
            groups["autoregressive"].append(col)
        elif "tax" in lower:
            groups["tax"].append(col)
        elif any(token in lower for token in ("dow", "dom", "month", "quarter", "year", "holiday", "nonworking", "post_holiday")):
            groups["calendar"].append(col)
        elif any(token in lower for token in ("imicex", "transrub", "covid", "key_rate", "ruonia", "usd_rub")):
            groups["macro"].append(col)
        else:
            groups["other"].append(col)
    return groups
