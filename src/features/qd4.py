"""Exact semantically defined QD4 descriptor used by the frozen protocol."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Sequence

import numpy as np

try:
    from numba import njit
except ModuleNotFoundError:  # pragma: no cover - environment guard
    def njit(function):
        return function


QD4_FEATURE_NAMES = (
    "spn_sum_of_lowest_edge_values",
    "spn_distance_varcoeff",
    "spn_mst_depth_mean",
    "spn_mst_dists_median",
)
ROOT = Path(__file__).resolve().parents[2]
SCALER_ASSET = ROOT / "assets" / "qd" / "reference_cloud" / "formal_raw_scaler.json"
R_SCRIPT = Path(__file__).with_name("extract_salesperson_raw_features.R")


def _rscript() -> str:
    configured = os.environ.get("BLADE_RSCRIPT")
    if configured:
        return configured
    found = shutil.which("Rscript")
    if found:
        return found
    standard = Path("C:/Program Files/R/R-4.6.0/bin/x64/Rscript.exe")
    if standard.is_file():
        return str(standard)
    raise FileNotFoundError("Rscript was not found; set BLADE_RSCRIPT")


def _transform_parameters() -> dict[str, tuple[float, float]]:
    payload = json.loads(SCALER_ASSET.read_text(encoding="utf-8"))
    params = {
        item["feature_name"]: (float(item["center_median"]), float(item["scale"]))
        for item in payload["parameters"]
    }
    missing = set(QD4_FEATURE_NAMES) - set(params)
    if missing or any(params[name][1] <= 0 for name in QD4_FEATURE_NAMES):
        raise RuntimeError(f"frozen QD4 transform is invalid; missing={sorted(missing)}")
    return params


def extract_qd4(coordinates: np.ndarray, state_ids: Sequence[str] | None = None) -> np.ndarray:
    """Extract transformed QD4 values for an array shaped ``(B,N,2)``.

    Raw values are produced by salesperson 1.0.0 with its full feature set,
    VRP blacklisted, costs excluded, normalization disabled, and duplicates
    retained. The frozen robust transform is then applied axis by axis.
    """

    xy = np.asarray(coordinates, dtype=np.float64)
    if xy.ndim != 3 or xy.shape[-1] != 2 or not np.isfinite(xy).all():
        raise ValueError(f"expected finite coordinates (B,N,2), got {xy.shape}")
    ids = list(state_ids) if state_ids is not None else [f"state_{i:06d}" for i in range(len(xy))]
    if len(ids) != len(xy) or len(set(ids)) != len(ids):
        raise ValueError("state_ids must be unique and match the batch length")

    with tempfile.TemporaryDirectory(prefix="blade_qd4_") as temporary:
        work = Path(temporary)
        input_csv = work / "coordinates_long.csv"
        output_csv = work / "features.csv"
        schema_csv = work / "schema.csv"
        with input_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["state_id", "node_idx", "x", "y"])
            for state_id, points in zip(ids, xy, strict=True):
                for node_idx, (x, y) in enumerate(points):
                    writer.writerow([state_id, node_idx, repr(float(x)), repr(float(y))])
        process = subprocess.run(
            [_rscript(), str(R_SCRIPT), str(input_csv), str(output_csv), str(schema_csv)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if process.returncode != 0 or not output_csv.is_file():
            raise RuntimeError(f"salesperson QD4 extraction failed: {process.stderr}")
        with output_csv.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
    if [row["state_id"] for row in rows] != ids:
        raise RuntimeError("salesperson output order differs from input order")
    params = _transform_parameters()
    result = np.empty((len(rows), 4), dtype=np.float64)
    for column, name in enumerate(QD4_FEATURE_NAMES):
        center, scale = params[name]
        raw = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        result[:, column] = np.arcsinh((raw - center) / scale)
    if not np.isfinite(result).all():
        raise RuntimeError("QD4 extraction produced non-finite values")
    return result


@njit
def _raw_qd4_exact(xy: np.ndarray) -> np.ndarray:
    """Specialized numeric reproduction of the four salesperson 1.0.0 axes."""

    batch, n, _ = xy.shape
    output = np.empty((batch, 4), dtype=np.float64)
    pair_count = n * (n - 1) // 2
    for b in range(batch):
        matrix = np.zeros((n, n), dtype=np.float64)
        distances = np.empty(pair_count, dtype=np.float64)
        cursor = 0
        for i in range(n):
            for j in range(i + 1, n):
                dx = xy[b, i, 0] - xy[b, j, 0]
                dy = xy[b, i, 1] - xy[b, j, 1]
                value = np.sqrt(dx * dx + dy * dy)
                matrix[i, j] = value
                matrix[j, i] = value
                distances[cursor] = value
                cursor += 1

        ordered = np.sort(distances)
        lowest_count = min(n, pair_count)
        lowest_sum = 0.0
        total = 0.0
        for i in range(pair_count):
            total += distances[i]
        for i in range(lowest_count):
            lowest_sum += ordered[i]
        mean_distance = total / pair_count
        variance_sum = 0.0
        for i in range(pair_count):
            delta = distances[i] - mean_distance
            variance_sum += delta * delta
        distance_sd = np.sqrt(variance_sum / (pair_count - 1))

        # vegan::spantree is Prim's algorithm rooted at the first point. Its
        # strict update and lowest-index tie behavior are reproduced here.
        in_tree = np.zeros(n, dtype=np.bool_)
        minimum = np.empty(n, dtype=np.float64)
        parent = np.empty(n, dtype=np.int64)
        in_tree[0] = True
        minimum[0] = 0.0
        parent[0] = -1
        for j in range(1, n):
            minimum[j] = matrix[0, j]
            parent[j] = 0
        mst_weights = np.empty(n - 1, dtype=np.float64)
        for step in range(n - 1):
            chosen = -1
            best = np.inf
            for j in range(n):
                if not in_tree[j] and minimum[j] < best:
                    best = minimum[j]
                    chosen = j
            if chosen < 0:
                raise ValueError("disconnected Euclidean graph")
            in_tree[chosen] = True
            mst_weights[step] = best
            for j in range(n):
                if not in_tree[j] and matrix[chosen, j] < minimum[j]:
                    minimum[j] = matrix[chosen, j]
                    parent[j] = chosen

        # Exact translation of vegan:::spandepth for a rooted spantree.
        active = np.ones(n, dtype=np.bool_)
        depth = np.ones(n, dtype=np.float64)
        while True:
            child_count = np.zeros(n, dtype=np.int64)
            for node in range(1, n):
                if active[node] and parent[node] >= 0:
                    child_count[parent[node]] += 1
            following = np.zeros(n, dtype=np.bool_)
            any_following = False
            for node in range(n):
                parent_active = parent[node] >= 0 and active[node] and active[parent[node]]
                keep = (child_count[node] >= 1 and parent_active) or child_count[node] >= 2
                following[node] = keep
                if keep:
                    depth[node] += 1.0
                    any_following = True
            active = following
            if not any_following:
                break

        ordered_mst = np.sort(mst_weights)
        midpoint = (n - 1) // 2
        if (n - 1) % 2 == 1:
            mst_median = ordered_mst[midpoint]
        else:
            mst_median = 0.5 * (ordered_mst[midpoint - 1] + ordered_mst[midpoint])
        output[b, 0] = lowest_sum
        output[b, 1] = distance_sd / mean_distance
        output[b, 2] = np.mean(depth)
        output[b, 3] = mst_median
    return output


def extract_qd4_exact(coordinates: np.ndarray) -> np.ndarray:
    """Extract exact frozen QD4 without invoking unrelated feature families.

    The implementation reproduces the four raw outputs of salesperson 1.0.0,
    then applies the byte-frozen robust transform. It is validated against the
    frozen deep-reference QD4 table by the bootstrap verifier and tests.
    """

    xy = np.ascontiguousarray(coordinates, dtype=np.float64)
    if xy.ndim != 3 or xy.shape[1:] != (100, 2) or not np.isfinite(xy).all():
        raise ValueError(f"expected finite N=100 coordinates (B,100,2), got {xy.shape}")
    raw = _raw_qd4_exact(xy)
    params = _transform_parameters()
    center = np.asarray([params[name][0] for name in QD4_FEATURE_NAMES], dtype=np.float64)
    scale = np.asarray([params[name][1] for name in QD4_FEATURE_NAMES], dtype=np.float64)
    transformed = np.arcsinh((raw - center) / scale)
    if not np.isfinite(transformed).all():
        raise RuntimeError("exact QD4 extraction produced non-finite values")
    return transformed
