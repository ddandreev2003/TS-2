"""Naive persistence baseline for time series."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NaiveModel:
    last_value: float

    def predict(self, n_steps: int = 1) -> np.ndarray:
        return np.full(n_steps, self.last_value, dtype=float)


def train_naive_baseline(y_train) -> NaiveModel:
    series = np.asarray(y_train, dtype=float)
    return NaiveModel(last_value=float(series[-1]))


def predict_naive(model: NaiveModel, n_steps: int = 1) -> np.ndarray:
    return model.predict(n_steps)
