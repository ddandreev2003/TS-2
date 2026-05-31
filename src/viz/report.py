"""Report orchestration: generate all stage plots and manifest."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.data.loader import add_active_flag, load_raw_data, validate_date_index
from src.drift.history import load_residual_history
from src.features.assembly import assemble_feature_matrix, prepare_supervised
from src.utils.config import load_config, load_json, save_json
from src.viz.drift import generate_drift_plots
from src.viz.eda import generate_eda_plots
from src.viz.features import generate_feature_plots
from src.viz.models import generate_model_plots


def generate_all_plots(
    config: dict[str, Any],
    *,
    holdout_predictions: dict[str, Any] | None = None,
    eval_summary: dict[str, Any] | None = None,
    metrics_summary: dict[str, Any] | None = None,
    monitoring_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate EDA, FS, model, and drift plots into artifacts/plots/."""
    artifacts = config.get("artifacts", {})
    plots_dir = Path(artifacts.get("dir", "artifacts")) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = config.get("data", {})
    df = load_raw_data(data_cfg.get("path", "data.csv"), data_cfg.get("date_column", "Date"))
    df = validate_date_index(df, data_cfg.get("date_column", "Date"))
    df = add_active_flag(df)

    feature_df = assemble_feature_matrix(df, config)
    X, y, _ = prepare_supervised(feature_df, target_col=data_cfg.get("target_column", "Balance"), active_only=True)

    metrics_path = Path(artifacts.get("metrics", "artifacts/metrics.json"))
    eval_path = Path(artifacts.get("eval_metrics", "artifacts/eval_metrics.json"))
    metrics_summary = metrics_summary or (load_json(metrics_path) if metrics_path.exists() else {})
    eval_summary = eval_summary or (load_json(eval_path) if eval_path.exists() else {})

    holdout_path = Path(artifacts.get("holdout_predictions", "artifacts/holdout_predictions.json"))
    if holdout_predictions is None and holdout_path.exists():
        holdout_predictions = load_json(holdout_path)
    holdout_predictions = holdout_predictions or {}

    monitoring_config = monitoring_config or load_config("config/monitoring_config.yaml")

    plot_paths: dict[str, list[str]] = {
        "eda": generate_eda_plots(df, plots_dir, config),
        "features": generate_feature_plots(metrics_summary, X, y, plots_dir),
        "models": generate_model_plots(eval_summary, holdout_predictions, plots_dir),
    }

    residual_path = Path(
        monitoring_config.get("artifacts", {}).get("residual_history", "artifacts/residual_history.json")
    )
    baseline_path = Path(
        monitoring_config.get("artifacts", {}).get("drift_baseline", "artifacts/drift_baseline.json")
    )
    residual_history = load_residual_history(residual_path) if residual_path.exists() else []
    baseline = load_json(baseline_path) if baseline_path.exists() else {}
    plot_paths["drift"] = generate_drift_plots(residual_history, baseline, monitoring_config, plots_dir)

    all_files = [p for group in plot_paths.values() for p in group]
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "plots_dir": str(plots_dir),
        "plot_groups": plot_paths,
        "plot_files": all_files,
        "feature_selection_method": metrics_summary.get("feature_selection_method"),
        "best_model": _best_model(eval_summary),
        "n_plots": len(all_files),
    }
    manifest_path = Path(artifacts.get("report_manifest", "artifacts/report_manifest.json"))
    save_json(manifest, manifest_path)
    return manifest


def _best_model(eval_summary: dict[str, Any]) -> str | None:
    ranking = eval_summary.get("business_ranking") or []
    if ranking:
        return ranking[0].get("model")
    model_metrics = eval_summary.get("model_metrics", {})
    best_name = None
    best_mae = float("inf")
    for name, metrics in model_metrics.items():
        if isinstance(metrics, dict) and "Balance" in metrics:
            mae = metrics["Balance"].get("mae", float("inf"))
        elif isinstance(metrics, dict):
            mae = metrics.get("mae", float("inf"))
        else:
            continue
        if mae < best_mae:
            best_mae = mae
            best_name = name
    return best_name
