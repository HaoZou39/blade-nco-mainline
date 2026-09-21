"""Canonical passive optimality-gap calculation."""

from __future__ import annotations

import numpy as np


def optimality_gap_percent(predicted_cost: np.ndarray, optimal_cost: np.ndarray) -> np.ndarray:
    predicted = np.asarray(predicted_cost, dtype=np.float64)
    optimal = np.asarray(optimal_cost, dtype=np.float64)
    if predicted.shape != optimal.shape or np.any(optimal <= 0):
        raise ValueError("predicted and optimal costs must have equal shape and positive optima")
    return 100.0 * (predicted - optimal) / optimal
