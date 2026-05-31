"""ARIMA, SARIMA, and ARIMAX model family."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.tuning.hyperopt import get_arima_search_space, get_arimax_search_space, get_sarima_search_space


def _as_ts_endog(y: pd.Series) -> pd.Series:
    """Return endog with a regular business-day DatetimeIndex (required by statsmodels)."""
    values = y.astype(float).to_numpy()
    if isinstance(y.index, pd.DatetimeIndex):
        start = y.index[0]
    else:
        start = pd.Timestamp("2000-01-01")
    index = pd.bdate_range(start=start, periods=len(values), freq="B")
    return pd.Series(values, index=index, name=getattr(y, "name", None))


def _as_ts_exog(exog: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(exog.to_numpy(), index=index, columns=exog.columns)


class SarimaxFitError(Exception):
    """Raised when SARIMAX MLE optimization does not converge after retries."""


def _fit_sarimax(model: SARIMAX, *, require_convergence: bool = True, for_cv: bool = False) -> Any:
    """Fit SARIMAX with retries and without convergence warnings."""
    if for_cv:
        fit_attempts: tuple[dict[str, Any], ...] = (
            {"maxiter": 150},
            {"method": "powell", "maxiter": 300},
        )
    else:
        fit_attempts = (
            {"maxiter": 200},
            {"method": "powell", "maxiter": 500},
            {"method": "nm", "maxiter": 1000},
        )
    fitted: Any | None = None

    for attempt_kwargs in fit_attempts:
        fitted = model.fit(disp=False, warn_convergence=False, **attempt_kwargs)
        if fitted.mle_retvals.get("converged", True):
            return fitted

    if require_convergence:
        raise SarimaxFitError("SARIMAX MLE did not converge")
    return fitted


@dataclass
class FittedTSModel:
    name: str
    model: Any
    order: tuple[int, ...]
    seasonal_order: tuple[int, int, int, int] | None = None
    exog_columns: list[str] | None = None

    def forecast(self, steps: int = 1, exog: pd.DataFrame | None = None) -> np.ndarray:
        if self.exog_columns and exog is not None:
            forecast = self.model.get_forecast(steps=steps, exog=exog[self.exog_columns])
        else:
            forecast = self.model.get_forecast(steps=steps)
        return np.asarray(forecast.predicted_mean, dtype=float)


def cv_mae(
    y: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int] | None,
    exog: pd.DataFrame | None,
    n_splits: int,
) -> float:
    y_ts = _as_ts_endog(y)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []

    for train_idx, test_idx in tscv.split(y_ts):
        y_train = y_ts.iloc[train_idx]
        y_test = y_ts.iloc[test_idx]
        train_exog = _as_ts_exog(exog.iloc[train_idx], y_train.index) if exog is not None else None
        test_exog = _as_ts_exog(exog.iloc[test_idx], y_ts.iloc[test_idx].index) if exog is not None else None

        try:
            model = SARIMAX(
                y_train,
                exog=train_exog,
                order=order,
                seasonal_order=seasonal_order or (0, 0, 0, 0),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fitted = _fit_sarimax(model, for_cv=True)
            preds = fitted.forecast(steps=len(test_idx), exog=test_exog)
            scores.append(mean_absolute_error(y_test, preds))
        except Exception:
            continue

    return float(np.mean(scores)) if scores else float("inf")


def _grid_search(
    y_train: pd.Series,
    config: dict[str, Any],
    space: dict[str, Any],
    exog: pd.DataFrame | None,
    seasonal: bool,
    reduced: bool = False,
) -> tuple[tuple[int, ...], tuple[int, int, int, int] | None, float]:
    n_splits = config.get("split", {}).get("arimax_cv_folds" if reduced else "time_series_cv_folds", 5)
    best_mae = float("inf")
    best_order = (1, 0, 0)
    best_seasonal = (0, 0, 0, 0) if seasonal else None

    p_vals = space["p"]
    d_vals = space["d"]
    q_vals = space["q"]

    if seasonal:
        seasonal_combos = product(space["P"], space["D"], space["Q"])
    else:
        seasonal_combos = [(0, 0, 0)]

    for p, d, q in product(p_vals, d_vals, q_vals):
        for P, D, Q in seasonal_combos:
            order = (p, d, q)
            seasonal_order = (P, D, Q, space["s"]) if seasonal else None
            score = cv_mae(
                y_train,
                order,
                seasonal_order,
                exog,
                n_splits=n_splits,
            )
            if score < best_mae:
                best_mae = score
                best_order = order
                best_seasonal = seasonal_order

    return best_order, best_seasonal, best_mae


def grid_search_arima(y_train: pd.Series, config: dict[str, Any]) -> tuple[tuple[int, ...], float]:
    space = get_arima_search_space(config)
    space["s"] = config.get("models", {}).get("seasonal_period", 5)
    order, _, mae = _grid_search(y_train, config, space, exog=None, seasonal=False)
    return order, mae


def grid_search_sarima(y_train: pd.Series, config: dict[str, Any]) -> tuple[tuple[int, ...], tuple[int, int, int, int], float]:
    space = get_sarima_search_space(config)
    order, seasonal_order, mae = _grid_search(y_train, config, space, exog=None, seasonal=True)
    return order, seasonal_order or (0, 0, 0, 0), mae


def grid_search_arimax(
    y_train: pd.Series,
    exog: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[tuple[int, ...], float]:
    space = get_arimax_search_space(config)
    space["s"] = config.get("models", {}).get("seasonal_period", 5)
    order, _, mae = _grid_search(y_train, config, space, exog=exog, seasonal=False, reduced=True)
    return order, mae


def fit_arima_model(y_train: pd.Series, order: tuple[int, int, int]) -> FittedTSModel:
    y_ts = _as_ts_endog(y_train)
    model = SARIMAX(
        y_ts,
        order=order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = _fit_sarimax(model)
    return FittedTSModel(name="arima", model=fitted, order=order)


def fit_sarima_model(
    y_train: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
) -> FittedTSModel:
    y_ts = _as_ts_endog(y_train)
    model = SARIMAX(
        y_ts,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = _fit_sarimax(model)
    return FittedTSModel(name="sarima", model=fitted, order=order, seasonal_order=seasonal_order)


def fit_arimax_model(
    y_train: pd.Series,
    exog: pd.DataFrame,
    order: tuple[int, int, int],
    exog_columns: list[str],
) -> FittedTSModel:
    y_ts = _as_ts_endog(y_train)
    exog_ts = _as_ts_exog(exog[exog_columns], y_ts.index)
    model = SARIMAX(
        y_ts,
        exog=exog_ts,
        order=order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = _fit_sarimax(model)
    return FittedTSModel(
        name="arimax",
        model=fitted,
        order=order,
        exog_columns=exog_columns,
    )


def fit_and_forecast(model: FittedTSModel, steps: int = 1, exog: pd.DataFrame | None = None) -> np.ndarray:
    return model.forecast(steps=steps, exog=exog)
