"""Shared configuration and artifact helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_artifact_dirs(config: dict[str, Any]) -> dict[str, Path]:
    artifacts = config.get("artifacts", {})
    root = Path(artifacts.get("dir", "artifacts"))
    models_dir = Path(artifacts.get("models_dir", root / "models"))
    plots_dir = root / "plots"
    for path in (root, models_dir, plots_dir):
        path.mkdir(parents=True, exist_ok=True)
    return {"root": root, "models_dir": models_dir, "plots_dir": plots_dir}


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def save_model(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_model(path: str | Path) -> Any:
    return joblib.load(path)
