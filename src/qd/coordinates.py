"""Normalization A and fixed CVT coordinate helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
NORMALIZATION_ASSET = ROOT / "assets" / "qd" / "normalization" / "normalization_A_P1P99.json"
CENTROID_ASSET = ROOT / "assets" / "qd" / "cvt" / "centroids.csv"
QD4_FEATURE_NAMES = (
    "spn_sum_of_lowest_edge_values",
    "spn_distance_varcoeff",
    "spn_mst_depth_mean",
    "spn_mst_dists_median",
)


def load_normalization(path: str | Path = NORMALIZATION_ASSET) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    features = payload.get("features", {})
    if set(features) != set(QD4_FEATURE_NAMES):
        raise RuntimeError("Normalization A feature names do not match frozen QD4")
    lower = np.asarray([features[name]["P1"] for name in QD4_FEATURE_NAMES], dtype=np.float64)
    upper = np.asarray([features[name]["P99"] for name in QD4_FEATURE_NAMES], dtype=np.float64)
    if lower.shape != (4,) or not np.all(upper > lower):
        raise RuntimeError("Normalization A contains invalid P1/P99 bounds")
    return lower, upper


def normalize_qd4(values: np.ndarray, path: str | Path = NORMALIZATION_ASSET) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape[-1:] != (4,) or not np.isfinite(array).all():
        raise ValueError(f"expected finite QD4 values with final dimension 4, got {array.shape}")
    lower, upper = load_normalization(path)
    return np.clip((array - lower) / (upper - lower), 0.0, 1.0)


def load_centroids(path: str | Path = CENTROID_ASSET) -> np.ndarray:
    table = np.loadtxt(Path(path), delimiter=",", skiprows=1, dtype=np.float64)
    if table.shape != (1000, 5) or not np.array_equal(table[:, 0], np.arange(1000)):
        raise RuntimeError(f"fixed CVT table must contain niche_id 0..999 and four axes, got {table.shape}")
    centroids = table[:, 1:]
    if centroids.shape != (1000, 4) or not np.isfinite(centroids).all():
        raise RuntimeError(f"fixed CVT centroid shape must be (1000,4), got {centroids.shape}")
    return np.ascontiguousarray(centroids)


def assign_nearest(normalized_qd4: np.ndarray, centroids: np.ndarray | None = None) -> np.ndarray:
    points = np.asarray(normalized_qd4, dtype=np.float64)
    centers = load_centroids() if centroids is None else np.asarray(centroids, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 4 or centers.shape != (1000, 4):
        raise ValueError("nearest-centroid assignment requires points (B,4) and centroids (1000,4)")
    # Chunking bounds memory without changing exact Euclidean argmin semantics.
    result = np.empty(len(points), dtype=np.int64)
    for start in range(0, len(points), 4096):
        chunk = points[start : start + 4096]
        squared = np.square(chunk[:, None, :] - centers[None, :, :]).sum(axis=2)
        result[start : start + len(chunk)] = np.argmin(squared, axis=1)
    return result
