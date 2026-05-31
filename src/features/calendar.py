"""Calendar, holiday, and tax-day features."""

from __future__ import annotations

from typing import Any

import holidays
import numpy as np
import pandas as pd


def _is_ru_holiday(dates: pd.Series) -> pd.Series:
    ru_holidays = holidays.RU(years=range(dates.dt.year.min(), dates.dt.year.max() + 1))
    return dates.dt.date.map(lambda d: d in ru_holidays).astype(int)


def _is_nonworking(dates: pd.Series, is_day_off: pd.Series | None) -> pd.Series:
    if is_day_off is not None:
        return is_day_off.fillna(0).astype(int)
    return ((dates.dt.dayofweek >= 5) | _is_ru_holiday(dates).astype(bool)).astype(int)


def build_calendar_features(df: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Build calendar and holiday features."""
    _ = config
    dates = pd.to_datetime(df["Date"])
    is_day_off = df.get("IsDayOff_Status_Workalendar_RU")

    out = pd.DataFrame({"Date": dates})
    out["dow"] = dates.dt.dayofweek
    out["dom"] = dates.dt.day
    out["month"] = dates.dt.month
    out["quarter"] = dates.dt.quarter
    out["year"] = dates.dt.year
    out["week_of_year"] = dates.dt.isocalendar().week.astype(int)
    out["is_month_start"] = dates.dt.is_month_start.astype(int)
    out["is_month_end"] = dates.dt.is_month_end.astype(int)
    out["is_quarter_start"] = dates.dt.is_quarter_start.astype(int)
    out["is_quarter_end"] = dates.dt.is_quarter_end.astype(int)
    out["is_year_start"] = dates.dt.is_year_start.astype(int)
    out["is_year_end"] = dates.dt.is_year_end.astype(int)
    out["is_holiday"] = _is_ru_holiday(dates)
    out["is_nonworking"] = _is_nonworking(dates, is_day_off)

    nonworking = out["is_nonworking"].astype(bool)
    post_holiday = (~nonworking) & nonworking.shift(1, fill_value=False)
    out["post_holiday"] = post_holiday.astype(int)

    run = 0
    post_lengths = []
    for flag in post_holiday:
        run = run + 1 if flag else 0
        post_lengths.append(run)
    out["post_holiday_run"] = post_lengths

    return out


def build_tax_features(df: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Build tax-day proximity features and include CSV tax_day column when present."""
    _ = config
    dates = pd.to_datetime(df["Date"])
    out = pd.DataFrame({"Date": dates})

    tax_days = [15, 25, 28]
    for day in tax_days:
        out[f"is_tax_day_{day}"] = (dates.dt.day == day).astype(int)
        for window in [1, 3, 5]:
            near = dates.dt.day.apply(lambda d: min(abs(d - day), abs(d - day + 31), abs(d - day - 31)) <= window)
            out[f"near_tax_{day}_w{window}"] = near.astype(int)

    out["is_quarter_tax_month"] = dates.dt.month.isin([3, 6, 9, 12]).astype(int)
    out["is_month_end_tax_window"] = ((dates.dt.day >= 25) | (dates.dt.day <= 3)).astype(int)

    if "tax_day" in df.columns:
        out["tax_day_csv"] = pd.to_numeric(df["tax_day"], errors="coerce").fillna(0).astype(int)

    return out
