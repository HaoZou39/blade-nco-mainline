"""Integer Euclidean TSP metric used by the validated BQ / Concorde protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


DEFAULT_INTEGER_SCALE = 1_000_000


def integer_distance_matrix(
    coords: np.ndarray, scale: int = DEFAULT_INTEGER_SCALE
) -> np.ndarray:
    xy = np.asarray(coords, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"expected coords shape (N,2), got {xy.shape}")
    diff = xy[:, None, :] - xy[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    d_int = np.rint(dist * float(scale)).astype(np.int64)
    np.fill_diagonal(d_int, 0)
    return d_int


def validate_tour_0based(tour_0based: Sequence[int], n: int) -> tuple[bool, str]:
    tour = [int(x) for x in tour_0based]
    if len(tour) != n:
        return False, f"expected tour length {n}, got {len(tour)}"
    if sorted(tour) != list(range(n)):
        return False, "tour is not a Hamiltonian permutation"
    return True, ""


def normalize_tour_0based(tour: Sequence[int], n: int) -> list[int]:
    out = [int(x) for x in tour]
    if len(out) == n + 1 and out[0] == out[-1]:
        out = out[:-1]
    if len(out) == n and min(out) >= 1 and max(out) == n:
        out = [x - 1 for x in out]
    ok, msg = validate_tour_0based(out, n)
    if not ok:
        raise ValueError(msg)
    return list(out)


def tour_cost_int(dist_int: np.ndarray, tour_0based: Sequence[int]) -> int:
    tour = [int(x) for x in tour_0based]
    n = dist_int.shape[0]
    if len(tour) != n:
        raise ValueError(f"tour length {len(tour)} != N={n}")
    return int(sum(int(dist_int[tour[i], tour[(i + 1) % n]]) for i in range(n)))


def integer_tour_cost(
    coords: np.ndarray, tour_0based: Sequence[int], scale: int = DEFAULT_INTEGER_SCALE
) -> int:
    xy = np.asarray(coords, dtype=np.float64)
    tour = np.asarray(list(tour_0based), dtype=np.int64)
    if tour.shape != (xy.shape[0],):
        raise ValueError(f"tour shape {tour.shape} != ({xy.shape[0]},)")
    ordered = xy[tour]
    edge_lengths = np.sqrt(np.square(ordered - np.roll(ordered, -1, axis=0)).sum(axis=1))
    return int(np.rint(edge_lengths * float(scale)).astype(np.int64).sum())


def write_explicit_tsp(path: Path, dist_int: np.ndarray, name: str = "instance") -> None:
    n = int(dist_int.shape[0])
    lines = [
        f"NAME: {name}",
        "TYPE: TSP",
        f"DIMENSION: {n}",
        "EDGE_WEIGHT_TYPE: EXPLICIT",
        "EDGE_WEIGHT_FORMAT: FULL_MATRIX",
        "EDGE_WEIGHT_SECTION",
    ]
    for i in range(n):
        lines.append(" ".join(str(int(dist_int[i, j])) for j in range(n)))
    lines.append("EOF")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def expert_closed_coordinates(coords: np.ndarray, tour_0based: Sequence[int]) -> np.ndarray:
    """Return expert-ordered closed coordinates of shape (N+1, 2)."""

    xy = np.asarray(coords, dtype=np.float64)
    tour = normalize_tour_0based(tour_0based, xy.shape[0])
    ordered = xy[np.asarray(tour, dtype=np.int64)]
    return np.vstack([ordered, ordered[:1]]).astype(np.float32)


def decode_tour_to_permutation(raw_tour: np.ndarray, n_cities: int) -> list[int]:
    mapped: list[int] = []
    for raw in np.asarray(raw_tour).astype(int).tolist():
        node = 0 if raw >= n_cities else int(raw)
        if not mapped or mapped[-1] != node:
            mapped.append(node)
    if len(mapped) == n_cities + 1 and mapped[0] == mapped[-1]:
        mapped = mapped[:-1]
    ok, msg = validate_tour_0based(mapped, n_cities)
    if not ok:
        raise RuntimeError(msg)
    if mapped[0] != 0:
        raise RuntimeError(f"decoded tour starts at {mapped[0]}, expected node 0")
    return mapped
