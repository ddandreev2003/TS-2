"""Feature selection with stability analysis across time-series CV folds.

ПОЧЕМУ: заказчик требует сравнить filter/wrapper/embedded методы и выбрать
        наиболее стабильный (Kuncheva + probe MAE через combined_loss).
КАК: TimeSeriesSplit по каждому методу → consensus features → stability metrics;
     select_best_fs_method выбирает победителя по loss_gamma из config.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, mutual_info_regression
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import spearmanr

from src.selection.stability import combined_loss, stability


@dataclass
class FeatureSelectionResult:
    fold_sets: dict[str, list[list[str]]] = field(default_factory=dict)
    stability_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    fold_mae: dict[str, list[float]] = field(default_factory=dict)
    consensus_sets: dict[str, list[str]] = field(default_factory=dict)
    method_ranking: pd.DataFrame | None = None


def fs_spearman(X: pd.DataFrame, y: pd.Series, k: int) -> list[str]:
    scores = []
    for col in X.columns:
        corr, _ = spearmanr(X[col], y, nan_policy="omit")
        scores.append((col, abs(corr) if corr is not None and not np.isnan(corr) else 0.0))
    scores.sort(key=lambda item: item[1], reverse=True)
    return [name for name, _ in scores[:k]]


def fs_mutual_info(X: pd.DataFrame, y: pd.Series, k: int, random_state: int = 42) -> list[str]:
    filled = X.fillna(X.median(numeric_only=True))
    mi = mutual_info_regression(filled, y, random_state=random_state)
    ranked = sorted(zip(X.columns, mi), key=lambda item: item[1], reverse=True)
    return [name for name, _ in ranked[:k]]


def fs_lasso(X: pd.DataFrame, y: pd.Series, k: int, random_state: int = 42) -> list[str]:
    filled = X.fillna(X.median(numeric_only=True))
    model = LassoCV(cv=5, random_state=random_state, max_iter=5000)
    model.fit(filled, y)
    coef = pd.Series(np.abs(model.coef_), index=X.columns)
    return coef.sort_values(ascending=False).head(k).index.tolist()


def fs_rfe_rf(X: pd.DataFrame, y: pd.Series, k: int, random_state: int = 42) -> list[str]:
    filled = X.fillna(X.median(numeric_only=True))
    estimator = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
    selector = RFE(estimator, n_features_to_select=min(k, X.shape[1]), step=0.1)
    selector.fit(filled, y)
    mask = selector.support_
    selected = X.columns[mask].tolist()
    if len(selected) > k:
        importances = selector.estimator_.feature_importances_
        ranked = sorted(zip(selected, importances), key=lambda item: item[1], reverse=True)
        selected = [name for name, _ in ranked[:k]]
    return selected


METHODS: dict[str, Callable[..., list[str]]] = {
    "spearman": fs_spearman,
    "mutual_info": fs_mutual_info,
    "lasso": fs_lasso,
    "rfe_rf": fs_rfe_rf,
}


def _probe_mae(
    X: pd.DataFrame,
    y: pd.Series,
    features: list[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    random_state: int,
    probe_cfg: dict[str, Any],
) -> float:
    if not features:
        return float("inf")
    model = RandomForestRegressor(
        n_estimators=probe_cfg.get("n_estimators", 200),
        max_depth=probe_cfg.get("max_depth", 8),
        random_state=random_state,
        n_jobs=-1,
    )
    X_train = X.iloc[train_idx][features].fillna(X[features].median(numeric_only=True))
    X_test = X.iloc[test_idx][features].fillna(X[features].median(numeric_only=True))
    model.fit(X_train, y.iloc[train_idx])
    preds = model.predict(X_test)
    return float(mean_absolute_error(y.iloc[test_idx], preds))


def _consensus_features(fold_sets: list[list[str]], min_folds: int) -> list[str]:
    counts: dict[str, int] = {}
    for fold in fold_sets:
        for feat in fold:
            counts[feat] = counts.get(feat, 0) + 1
    return sorted([feat for feat, count in counts.items() if count >= min_folds])


def run_feature_selection_cv(
    X: pd.DataFrame,
    y: pd.Series,
    config: dict[str, Any],
    methods: dict[str, Callable[..., list[str]]] | None = None,
) -> FeatureSelectionResult:
    """Run feature selection methods across time-series CV folds."""
    sel_cfg = config.get("selection", config)
    top_k = sel_cfg.get("top_k_features", 20)
    n_splits = sel_cfg.get("fs_cv_folds", config.get("split", {}).get("time_series_cv_folds", 5))
    min_folds = sel_cfg.get("consensus_min_folds", 3)
    random_state = config.get("random_state", 42)
    gamma_values = sel_cfg.get("gamma_values", [0.3, 0.5, 0.7])
    probe_cfg = sel_cfg.get("probe_rf", {})

    methods = methods or METHODS
    tscv = TimeSeriesSplit(n_splits=n_splits)
    result = FeatureSelectionResult()

    for method_name, method_fn in methods.items():
        fold_features: list[list[str]] = []
        fold_mae: list[float] = []

        for train_idx, test_idx in tscv.split(X):
            X_fold = X.iloc[train_idx]
            y_fold = y.iloc[train_idx]
            if method_name == "spearman":
                selected = method_fn(X_fold, y_fold, top_k)
            else:
                selected = method_fn(X_fold, y_fold, top_k, random_state=random_state)
            fold_features.append(selected)
            fold_mae.append(
                _probe_mae(X, y, selected, train_idx, test_idx, random_state, probe_cfg)
            )

        result.fold_sets[method_name] = fold_features
        result.fold_mae[method_name] = fold_mae
        result.stability_metrics[method_name] = stability((set(f) for f in fold_features), p=X.shape[1])
        result.consensus_sets[method_name] = _consensus_features(fold_features, min_folds)

    rows = []
    mae_means = {m: float(np.mean(scores)) for m, scores in result.fold_mae.items()}
    max_mae = max(mae_means.values()) if mae_means else 1.0
    min_mae = min(mae_means.values()) if mae_means else 0.0
    mae_range = max(max_mae - min_mae, 1e-9)

    for method_name in methods:
        stab = result.stability_metrics[method_name]
        mae_norm = (mae_means[method_name] - min_mae) / mae_range
        row = {
            "method": method_name,
            "mae_mean": mae_means[method_name],
            "mae_std": float(np.std(result.fold_mae[method_name])),
            "jaccard": stab["jaccard"],
            "dice": stab["dice"],
            "kuncheva": stab["kuncheva"],
            "n_consensus_features": len(result.consensus_sets[method_name]),
        }
        for gamma in gamma_values:
            row[f"loss_gamma_{gamma}"] = combined_loss(stab["kuncheva"], mae_norm, gamma)
        rows.append(row)

    result.method_ranking = pd.DataFrame(rows).sort_values("loss_gamma_0.5")
    return result


def select_features(X: pd.DataFrame, y: pd.Series, config: dict[str, Any]) -> dict[str, list[str]]:
    """Run CV feature selection and return consensus feature sets."""
    result = run_feature_selection_cv(X, y, config)
    return result.consensus_sets


def rank_methods(results: FeatureSelectionResult, gamma: float = 0.5) -> pd.DataFrame:
    """Return method ranking sorted by combined loss for a given gamma."""
    if results.method_ranking is None:
        return pd.DataFrame()
    col = f"loss_gamma_{gamma}"
    if col not in results.method_ranking.columns:
        return results.method_ranking.sort_values("mae_mean")
    return results.method_ranking.sort_values(col)


def select_best_fs_method(
    fs_result: FeatureSelectionResult,
    config: dict[str, Any],
    fallback: str = "spearman",
) -> str:
    """ПОЧЕМУ: победитель по stability+MAE должен автоматически использоваться в обучении."""
    gamma = config.get("selection", {}).get("ranking_gamma", 0.5)
    ranking = rank_methods(fs_result, gamma=gamma)
    if ranking.empty:
        return fallback
    return str(ranking.iloc[0]["method"])


def resolve_fs_features(
    fs_result: FeatureSelectionResult,
    config: dict[str, Any],
    all_features: list[str],
) -> tuple[str, list[str]]:
    """КАК: выбираем метод с минимальным combined_loss и берём его consensus set."""
    method = select_best_fs_method(fs_result, config)
    features = fs_result.consensus_sets.get(method, all_features)
    return method, features or all_features


def save_feature_sets(sets: dict[str, list[str]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(sets, f, indent=2, ensure_ascii=False)


def load_feature_sets(path: str | Path) -> dict[str, list[str]]:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)
