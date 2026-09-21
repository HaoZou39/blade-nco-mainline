"""Local loaders for the frozen passive evaluation families."""

from __future__ import annotations

import csv
import gzip
from pathlib import Path
from typing import Iterator

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EVALUATION_ROOT = ROOT / "assets" / "evaluation"
FIXED_DATASETS = {
    "Uniform100": ("uniform100", "coordinates"),
    "GaussianMixture100": ("gaussian_mixture100", "coordinates"),
    "TSPLIB100": ("tsplib100", "model_input_coordinates"),
}


def load_fixed_n100(name: str) -> np.ndarray:
    if name not in FIXED_DATASETS:
        raise KeyError(f"unknown fixed N=100 dataset: {name}")
    directory, key = FIXED_DATASETS[name]
    path = EVALUATION_ROOT / directory / "instances" / "coordinates.npz"
    with np.load(path, allow_pickle=False) as payload:
        coordinates = np.asarray(payload[key], dtype=np.float64)
    if coordinates.shape != (10_000, 100, 2) or not np.isfinite(coordinates).all():
        raise RuntimeError(f"{name} coordinates have invalid shape or values: {coordinates.shape}")
    return coordinates


def load_lehd_cross_size(size: int) -> tuple[np.ndarray, np.ndarray]:
    if size not in (200, 500, 1000):
        raise ValueError("frozen LEHD cross-size evaluation supports 200, 500, and 1000")
    path = EVALUATION_ROOT / "lehd_cross_size" / f"test_TSP{size}_n128.txt"
    coordinates: list[np.ndarray] = []
    tours: list[np.ndarray] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            tokens = line.split()
            if not tokens:
                continue
            try:
                marker = tokens.index("output")
            except ValueError as exc:
                raise ValueError(f"{path.name}:{line_number}: missing output marker") from exc
            if marker != 2 * size or len(tokens) - marker - 1 != size + 1:
                raise ValueError(f"{path.name}:{line_number}: token count does not match TSP{size}")
            coordinates.append(np.asarray(tokens[:marker], dtype=np.float64).reshape(size, 2))
            tours.append(np.asarray(tokens[marker + 1 :], dtype=np.int64))
    xy = np.stack(coordinates)
    route = np.stack(tours)
    if xy.shape != (128, size, 2) or route.shape != (128, size + 1) or not np.isfinite(xy).all():
        raise RuntimeError(f"invalid LEHD TSP{size} dataset shape")
    return xy, route


def _parse_euc2d_gzip(path: Path) -> np.ndarray:
    with gzip.open(path, "rt", encoding="ascii", errors="strict") as handle:
        lines = handle.readlines()
    dimension = None
    coordinates: list[tuple[int, float, float]] = []
    in_coordinates = False
    for raw in lines:
        line = raw.strip()
        upper = line.upper()
        if upper.startswith("DIMENSION"):
            dimension = int(line.replace(":", " ").split()[-1])
        elif upper == "NODE_COORD_SECTION":
            in_coordinates = True
        elif in_coordinates and upper in {"EOF", "DISPLAY_DATA_SECTION", "TOUR_SECTION"}:
            break
        elif in_coordinates and line:
            parts = line.split()
            coordinates.append((int(parts[0]), float(parts[1]), float(parts[2])))
    if dimension is None or len(coordinates) != dimension:
        raise ValueError(f"invalid TSPLIB coordinate file: {path}")
    coordinates.sort(key=lambda item: item[0])
    return np.asarray([(x, y) for _, x, y in coordinates], dtype=np.float64)


def load_original_tsplib70() -> Iterator[tuple[str, np.ndarray, int]]:
    directory = EVALUATION_ROOT / "original_tsplib70"
    with (directory / "source_manifest_bq83.csv").open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["edge_weight_type"] == "EUC_2D"]
    if len(rows) != 70:
        raise RuntimeError(f"OriginalTSPLIB70 source selection contains {len(rows)} rows")
    for row in rows:
        path = directory / "instances" / row["size_bin"] / f"{row['source_instance']}.tsp.gz"
        yield row["source_instance"], _parse_euc2d_gzip(path), int(row["known_optimum"])
