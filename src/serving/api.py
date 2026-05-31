"""FastAPI serving application."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.serving.model_loader import load_best_model
from src.serving.predictor import predict_next_day
from src.serving.prometheus_metrics import update_all_from_artifacts
from src.utils.config import load_config

app = FastAPI(title="TS-2 Liquidity Forecast", version="1.0.0")

CONFIG_PATH = "config/model_config.yaml"


@app.on_event("startup")
def _load_eval_metrics() -> None:
    config = load_config(CONFIG_PATH)
    update_all_from_artifacts(config)


@app.get("/health")
def health() -> dict[str, Any]:
    config = load_config(CONFIG_PATH)
    try:
        model_name, _ = load_best_model(CONFIG_PATH)
        status = "ok"
    except FileNotFoundError:
        model_name = None
        status = "no_model"
    return {
        "status": status,
        "model": model_name,
        "training_mode": config.get("training", {}).get("mode", "dual_target"),
    }


@app.post("/predict")
def predict() -> dict[str, Any]:
    try:
        model_name, model = load_best_model(CONFIG_PATH)
        return predict_next_day(CONFIG_PATH, model_name=model_name, model=model)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/metrics")
def metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
