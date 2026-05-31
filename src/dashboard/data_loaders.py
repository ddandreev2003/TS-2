"""Artifact loading helpers for Streamlit dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
PLOTS = ARTIFACTS / "plots"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_manifest() -> dict[str, Any]:
    return load_json(ARTIFACTS / "report_manifest.json")


def load_metrics() -> dict[str, Any]:
    return load_json(ARTIFACTS / "metrics.json")


def load_eval() -> dict[str, Any]:
    return load_json(ARTIFACTS / "eval_metrics.json")


def load_holdout() -> dict[str, Any]:
    return load_json(ARTIFACTS / "holdout_predictions.json")


def load_drift_status() -> dict[str, Any]:
    return load_json(ARTIFACTS / "drift_status.json")


def load_calibration() -> dict[str, Any]:
    return load_json(ARTIFACTS / "calibration_policy.json")


def list_plot_files(group: str | None = None) -> list[Path]:
    manifest = load_manifest()
    if group and group in manifest.get("plot_groups", {}):
        return [Path(p) for p in manifest["plot_groups"][group] if Path(p).exists()]
    plot_files = manifest.get("plot_files", [])
    if plot_files:
        return [Path(p) for p in plot_files if Path(p).exists()]
    if PLOTS.exists():
        return sorted(PLOTS.glob("*.png"))
    return []


def model_ranking_df() -> pd.DataFrame:
    eval_data = load_eval()
    rows = []
    for name, metrics in eval_data.get("model_metrics", {}).items():
        if isinstance(metrics, dict) and "Balance" in metrics:
            bal = metrics["Balance"]
            rows.append(
                {
                    "model": name,
                    "mae": bal.get("mae"),
                    "asymmetric_cost": bal.get("asymmetric_cost"),
                    "within_threshold_share": bal.get("within_threshold_share"),
                    "quality_gate": eval_data.get("quality_gate_passed", {}).get(name),
                }
            )
        elif isinstance(metrics, dict) and "mae" in metrics:
            rows.append(
                {
                    "model": name,
                    "mae": metrics.get("mae"),
                    "asymmetric_cost": metrics.get("asymmetric_cost"),
                    "within_threshold_share": metrics.get("within_threshold_share"),
                    "quality_gate": eval_data.get("quality_gate_passed", {}).get(name),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("mae")


def retrain_events() -> list[dict[str, Any]]:
    retrain_dir = ROOT / "retraining"
    events = []
    if not retrain_dir.exists():
        return events
    for path in sorted(retrain_dir.glob("retrain_*.json")):
        events.append(load_json(path))
    return events
