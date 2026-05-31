"""Tabular regression models with sklearn pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.tuning.hyperopt import get_linear_search_space, get_nonlinear_search_space

LINEAR_MODELS = {
    "lasso": Lasso(max_iter=10000),
    "ridge": Ridge(),
    "elasticnet": ElasticNet(max_iter=10000),
}

NONLINEAR_MODELS = {
    "random_forest": RandomForestRegressor(random_state=42),
    "extra_trees": ExtraTreesRegressor(random_state=42),
    "gradient_boosting": GradientBoostingRegressor(random_state=42),
}


@dataclass
class FittedTabularModel:
    name: str
    pipeline: Any
    features: list[str]
    cv_best_mae: float | None = None


def _build_linear_pipeline(model_name: str) -> Pipeline:
    model = LINEAR_MODELS[model_name]
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def _build_nonlinear_pipeline(model_name: str, random_state: int) -> Pipeline:
    model = NONLINEAR_MODELS[model_name]
    model.set_params(random_state=random_state)
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])


def train_linear_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
    config: dict[str, Any],
    features: list[str] | None = None,
) -> FittedTabularModel:
    random_state = config.get("random_state", 42)
    features = features or X_train.columns.tolist()
    X = X_train[features]
    pipeline = _build_linear_pipeline(model_name)
    param_grid = get_linear_search_space(model_name, config)
    n_splits = config.get("split", {}).get("time_series_cv_folds", 5)
    search = GridSearchCV(
        pipeline,
        param_grid,
        cv=TimeSeriesSplit(n_splits=n_splits),
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    search.fit(X, y_train)
    return FittedTabularModel(
        name=model_name,
        pipeline=search.best_estimator_,
        features=features,
        cv_best_mae=float(-search.best_score_),
    )


def train_nonlinear_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
    config: dict[str, Any],
    features: list[str] | None = None,
) -> FittedTabularModel:
    random_state = config.get("random_state", 42)
    features = features or X_train.columns.tolist()
    X = X_train[features]
    pipeline = _build_nonlinear_pipeline(model_name, random_state)
    param_grid = get_nonlinear_search_space(model_name, config)
    n_splits = config.get("split", {}).get("time_series_cv_folds", 5)
    search = GridSearchCV(
        pipeline,
        param_grid,
        cv=TimeSeriesSplit(n_splits=n_splits),
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    search.fit(X, y_train)
    return FittedTabularModel(
        name=model_name,
        pipeline=search.best_estimator_,
        features=features,
        cv_best_mae=float(-search.best_score_),
    )


def predict_tabular(model: FittedTabularModel, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.pipeline.predict(X[model.features]), dtype=float)
