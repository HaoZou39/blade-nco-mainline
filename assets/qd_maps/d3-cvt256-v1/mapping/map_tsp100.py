"""Assign TSP100 coordinates to D3-CVT-256 working map v1.

Feature extraction is the packaged copy of extract_salesperson_raw_features.R.
Scaling and nearest-center assignment follow the Stage 2B-D mapper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

PACKAGE = Path(__file__).resolve().parents[1]
RSCRIPT = os.environ.get("BLADE_RSCRIPT", "/home/haozou/micromamba/envs/blade-r/bin/Rscript")
FEATURES = ["spn_mst_dists_varcoeff", "spn_distance_median", "spn_angles_mean"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def resolve(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    if manifest.get("version_id") != "d3-cvt256-v1":
        raise SystemExit("map version check failed: version_id")
    scaler_path = resolve(path, manifest["scaler_path"])
    center_path = resolve(path, manifest["centroid_path"])
    extractor_path = resolve(path, manifest["feature_extractor_path"])
    schema_path = resolve(path, manifest["schema_path"])
    if sha256_file(scaler_path) != manifest["scaler_hash"]:
        raise SystemExit("map version check failed: scaler hash")
    if sha256_file(extractor_path) != manifest["feature_extractor_hash"]:
        raise SystemExit("map version check failed: extractor hash")
    if sha256_file(schema_path) != manifest["schema_hash"]:
        raise SystemExit("map version check failed: schema hash")
    centers = np.load(center_path)
    if sha256_array(centers) != manifest["centroid_array_hash"]:
        raise SystemExit("map version check failed: centroid array hash")
    if sha256_file(center_path) != manifest["centroid_file_hash"]:
        raise SystemExit("map version check failed: centroid file hash")
    if tuple(manifest["feature_names"]) != tuple(FEATURES) or int(manifest["K"]) != 256:
        raise SystemExit("map version check failed: descriptor or K")
    if centers.shape != (256, 3):
        raise SystemExit("map version check failed: centroid shape")
    manifest["centers"] = centers
    manifest["_scaler_path"] = scaler_path
    manifest["_extractor_path"] = extractor_path
    manifest["_schema_path"] = schema_path
    manifest["_manifest_hash"] = sha256_file(path)
    return manifest


def scaler_arrays(manifest: dict) -> tuple[np.ndarray, np.ndarray]:
    scaler = json.loads(Path(manifest["_scaler_path"]).read_text())
    medians = np.array([scaler["median"][name] for name in FEATURES], dtype=float)
    scales = np.array([scaler["iqr"][name] for name in FEATURES], dtype=float)
    if np.any(scales <= 0) or not np.isfinite(medians).all():
        raise SystemExit("map version check failed: scaler values")
    return medians, scales


def read_coordinates(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        try:
            values = np.load(path)
        except ValueError as exc:
            if "pickle" not in str(exc):
                raise
            values = np.load(path, allow_pickle=True)
        values = np.asarray(values, dtype=float)
    elif path.suffix == ".csv":
        frame = pd.read_csv(path)
        if {"state_id", "node_idx", "x", "y"} <= set(frame.columns):
            groups = []
            for _, block in frame.groupby("state_id", sort=False):
                ordered = block.sort_values("node_idx")
                groups.append(ordered[["x", "y"]].to_numpy(float))
            values = np.stack(groups)
        elif {"x", "y"} <= set(frame.columns):
            values = frame[["x", "y"]].to_numpy(float)
        else:
            raise SystemExit("input csv needs x,y or state_id,node_idx,x,y")
    elif path.suffix == ".json":
        payload = json.loads(path.read_text())
        if isinstance(payload, dict):
            payload = payload["coordinates"]
        values = np.asarray(payload, dtype=float)
    else:
        raise SystemExit("input must be .npy, .csv, or .json")
    if values.ndim == 2:
        values = values[None, :, :]
    if values.ndim != 3 or values.shape[1:] != (100, 2):
        raise SystemExit(f"expected shape (100, 2) or (N, 100, 2), got {tuple(values.shape)}")
    if not np.isfinite(values).all():
        raise SystemExit("input coordinates are not finite")
    return np.ascontiguousarray(values, dtype=np.float64)


def extract_raw(coordinates: np.ndarray, manifest: dict) -> np.ndarray:
    if len(coordinates) == 0:
        raise SystemExit("input is empty")
    with tempfile.TemporaryDirectory(prefix="blade-d3-") as folder:
        work = Path(folder)
        frames = []
        for start in range(0, len(coordinates), 250):
            end = min(start + 250, len(coordinates))
            ids = [f"s{index}" for index in range(start, end)]
            long_csv = work / f"long_{start}.csv"
            raw_csv = work / f"raw_{start}.csv"
            with long_csv.open("w") as handle:
                handle.write("state_id,node_idx,x,y\n")
                for state_id, points in zip(ids, coordinates[start:end]):
                    for node, (x_coord, y_coord) in enumerate(points):
                        handle.write(f"{state_id},{node},{x_coord:.16g},{y_coord:.16g}\n")
            completed = subprocess.run(
                [RSCRIPT, str(manifest["_extractor_path"]), str(long_csv), str(raw_csv), str(manifest["_schema_path"])],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            if completed.returncode != 0 or not raw_csv.exists():
                raise SystemExit(f"feature extraction failed: {completed.stdout[-400:]}")
            frames.append(pd.read_csv(raw_csv))
        table = pd.concat(frames, ignore_index=True).drop_duplicates("state_id").set_index("state_id")
        ordered = [f"s{index}" for index in range(len(coordinates))]
        return table.loc[ordered, FEATURES].to_numpy(float)


def scale_raw(raw: np.ndarray, medians: np.ndarray, scales: np.ndarray) -> np.ndarray:
    scaled = np.full_like(raw, np.nan, dtype=float)
    finite = np.isfinite(raw).all(axis=1)
    scaled[finite] = np.arcsinh((raw[finite] - medians) / scales)
    return scaled


def assign_scaled(scaled: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    niches = np.full(len(scaled), -1, dtype=int)
    nearest = np.full(len(scaled), np.nan)
    second = np.full(len(scaled), np.nan)
    finite = np.isfinite(scaled).all(axis=1)
    if not finite.any():
        return niches, nearest, second
    distance = cdist(scaled[finite], centers)
    rows = np.arange(distance.shape[0])
    first = distance.argmin(axis=1)
    nearest_values = distance[rows, first].copy()
    distance[rows, first] = np.inf
    second_index = distance.argmin(axis=1)
    niches[finite] = first
    nearest[finite] = nearest_values
    second[finite] = distance[rows, second_index]
    return niches, nearest, second


def support_label(distance: float, thresholds: dict) -> str:
    if not np.isfinite(distance):
        return "nonfinite_descriptor"
    if distance <= float(thresholds["p95"]):
        return "normal"
    if distance <= float(thresholds["p99"]):
        return "warning"
    return "severe"


def map_coordinates(coordinates: np.ndarray, manifest: dict) -> list[dict]:
    medians, scales = scaler_arrays(manifest)
    raw = extract_raw(coordinates, manifest)
    scaled = scale_raw(raw, medians, scales)
    niches, nearest, second = assign_scaled(scaled, manifest["centers"])
    thresholds = manifest["support_thresholds"]
    rows = []
    for index in range(len(coordinates)):
        margin = float(second[index] - nearest[index]) if np.isfinite(nearest[index]) else None
        rows.append({
            "raw_d3_features": raw[index].tolist(),
            "scaled_d3_features": scaled[index].tolist(),
            "niche_id": int(niches[index]),
            "nearest_centroid_distance": None if not np.isfinite(nearest[index]) else float(nearest[index]),
            "second_nearest_centroid_distance": None if not np.isfinite(second[index]) else float(second[index]),
            "assignment_margin": margin,
            "reference_support_status": support_label(nearest[index], thresholds),
            "map_version": manifest["version_id"],
            "map_manifest_hash": manifest["_manifest_hash"],
        })
    return rows


def write_output(path: Path, rows: list[dict]) -> None:
    if path.suffix == ".csv":
        pd.DataFrame(rows).to_csv(path, index=False)
        return
    path.write_text(json.dumps({"assignments": rows}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Assign D3-CVT-256 working map v1 niches to TSP100 coordinates.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--map", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        manifest = load_manifest(args.map)
        coordinates = read_coordinates(args.input)
        rows = map_coordinates(coordinates, manifest)
    except SystemExit as exc:
        message = str(exc) or "mapping failed"
        args.output.write_text(json.dumps({"error": message}))
        print(message, file=sys.stderr)
        raise
    write_output(args.output, rows)


if __name__ == "__main__":
    main()
