"""Frozen S2-A model wrappers. RF4 is not run (config never frozen)."""
from __future__ import annotations

import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

RF_NOT_RUN = "NOT_RUN_RF_CONFIG_NOT_FROZEN"
SEEDS = [2026, 2027, 2028, 2029, 2030]

HGB_CFG_CANONICAL = dict(
    learning_rate=0.05,
    max_iter=500,
    max_leaf_nodes=31,
    max_depth=None,
    min_samples_leaf=20,
    l2_regularization=1e-4,
    early_stopping=False,
)

HGB_CFG_SOURCE = {
    "file": "src/final_run/science.py",
    "symbol": "HGB_CFG",
    "usage": "train_hgb_controls: HistGradientBoostingRegressor on QD4, y=log1p(L_BQ)",
    "also_used_in": "scripts/run_cycle0_models_and_evaluation.py",
}

RF_CANDIDATES = {
    "A_sklearn_regressor_defaults": {
        "n_estimators": 100,
        "max_depth": None,
        "max_features": 1.0,
        "min_samples_leaf": 1,
        "max_leaf_nodes": None,
        "bootstrap": True,
        "n_jobs": 1,
        "note": "sklearn RandomForestRegressor defaults except random_state=SEED",
    },
    "B_hgb_capacity_analog": {
        "n_estimators": 500,
        "max_depth": None,
        "max_features": 1.0,
        "min_samples_leaf": 20,
        "max_leaf_nodes": 31,
        "bootstrap": True,
        "n_jobs": 1,
        "note": "capacity analog of canonical HGB_CFG (not itself a historical RF config)",
    },
    "C_moderate_regularization": {
        "n_estimators": 300,
        "max_depth": None,
        "max_features": "sqrt",
        "min_samples_leaf": 5,
        "max_leaf_nodes": None,
        "bootstrap": True,
        "n_jobs": 1,
        "note": "smaller/regularized forest; also not a historical hardness config",
    },
}


def hgb_params(seed: int) -> dict[str, Any]:
    params = deepcopy(HGB_CFG_CANONICAL)
    params["random_state"] = int(seed)
    return params


def _rss_bytes() -> int | None:
    try:
        import psutil
        return int(psutil.Process().memory_info().rss)
    except Exception:
        return None


def fit_hgb4_seed(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[Any, dict[str, Any]]:
    params = hgb_params(seed)
    rss_before = _rss_bytes()
    started = time.perf_counter()
    estimator = HistGradientBoostingRegressor(**params)
    estimator.fit(x, y)
    fit_s = time.perf_counter() - started
    rss_after = _rss_bytes()
    peak = None
    if rss_before is not None and rss_after is not None:
        peak = max(rss_before, rss_after)
    return estimator, {
        "seed": int(seed),
        "params": params,
        "fit_s": float(fit_s),
        "peak_rss_bytes": peak,
        "n_train": int(len(y)),
        "target": "log1p_L_BQ",
        "features": 4,
    }


def save_hgb(estimator, meta: dict[str, Any], path: Path, features: list[str]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": estimator,
        "features": list(features),
        "target": "log1p_L_BQ",
        "meta": meta,
        "source": HGB_CFG_SOURCE,
    }
    joblib.dump(payload, path, compress=3)
    meta = dict(meta)
    meta["path"] = str(path)
    meta["serialized_bytes"] = int(path.stat().st_size)
    meta["n_features"] = len(features)
    return meta


def save_hgb4(estimator, meta: dict[str, Any], path: Path) -> dict[str, Any]:
    return save_hgb(
        estimator,
        meta,
        path,
        [
            "spn_sum_of_lowest_edge_values",
            "spn_distance_varcoeff",
            "spn_mst_depth_mean",
            "spn_mst_dists_median",
        ],
    )


def load_hgb4(path: Path):
    payload = joblib.load(path)
    return payload["model"], payload.get("meta") or {}


def predict_hgb(estimator, x: np.ndarray) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    pred = np.asarray(estimator.predict(x), dtype=float)
    elapsed = time.perf_counter() - started
    if not np.isfinite(pred).all():
        raise RuntimeError("HGB4 produced non-finite predictions")
    return pred, elapsed


def median_stack(seed_preds: np.ndarray) -> np.ndarray:
    stacked = np.asarray(seed_preds, dtype=float)
    if stacked.ndim != 2 or stacked.shape[0] != 5:
        raise RuntimeError(f"expected (5, n) seed predictions, got {stacked.shape}")
    return np.median(stacked, axis=0)
