"""Geometry shared by the frozen lattice and motif operators.

The call order inside each operator matches the audited Unified-12 source.
These helpers do not draw random numbers except motif_sites, which draws only
the random numbers of the selected motif.
"""

from __future__ import annotations

from typing import Any, Mapping
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

from .result import MutationResult, coordinate_hash


def contain(sites: np.ndarray, center: np.ndarray, lo: float = 0.02, hi: float = 0.98) -> np.ndarray:
    relative = np.asarray(sites, dtype=np.float64) - center
    scale = 1.0
    for offset, origin in zip(relative, np.broadcast_to(center, relative.shape)):
        for value, base in zip(offset, origin):
            if abs(value) <= 1e-12:
                continue
            bound = hi if value > 0.0 else lo
            scale = min(scale, (bound - base) / value)
    return center + relative * max(scale, 0.0)


def match(origins: np.ndarray, sites: np.ndarray) -> np.ndarray:
    cost = np.sum((origins[:, None, :] - sites[None, :, :]) ** 2, axis=2)
    assignment = linear_sum_assignment(cost)
    return sites[assignment[1]]


def spacing(sites: np.ndarray) -> float:
    if len(sites) < 2:
        return 1.0
    delta = sites[:, None, :] - sites[None, :, :]
    distance = np.sqrt(np.sum(delta * delta, axis=2))
    np.fill_diagonal(distance, np.inf)
    return float(distance.min())


def valid_child(coordinates: np.ndarray, *, node_count: int, min_pair: float) -> str:
    if coordinates.shape != (node_count, 2) or not np.isfinite(coordinates).all():
        return "invalid_shape"
    if np.any((coordinates < 0.0) | (coordinates > 1.0)):
        return "out_of_bounds"
    if spacing(coordinates) < min_pair:
        return "near_duplicate"
    return ""


def rotate(sites: np.ndarray, center: np.ndarray, degrees: float) -> np.ndarray:
    radians = np.deg2rad(degrees)
    matrix = np.array(
        [[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]],
        dtype=np.float64,
    )
    return (sites - center) @ matrix.T + center


def _rays(center: np.ndarray, scale: float, theta: float, counts: np.ndarray, turn: float) -> np.ndarray:
    sites = [np.asarray(center, dtype=np.float64)]
    for arm, arm_count in enumerate(counts):
        direction = theta + arm * turn
        vector = np.array([np.cos(direction), np.sin(direction)], dtype=np.float64)
        for step in range(1, int(arm_count) + 1):
            sites.append(center + vector * scale * step / max(int(arm_count), 1))
    return np.asarray(sites, dtype=np.float64)


def motif_sites(
    kind: str,
    count: int,
    center: np.ndarray,
    scale: float,
    orientation: float,
    rng: np.random.Generator,
    settings: Mapping[str, Any],
) -> np.ndarray:
    theta = np.deg2rad(orientation)
    if kind == "arc":
        span = np.deg2rad(float(rng.uniform(settings["arc_span_degrees"]["low"], settings["arc_span_degrees"]["high"])))
        start = np.deg2rad(float(rng.uniform(settings["arc_start_degrees"]["low"], settings["arc_start_degrees"]["high"])))
        angles = np.linspace(start, start + span, count)
        return center + scale * np.column_stack((np.cos(angles), np.sin(angles)))
    if kind == "star":
        low = int(settings["star_arms"]["low"])
        high = int(settings["star_arms"]["high"])
        if low < 5 or high > 6 or low > high:
            raise ValueError("star arms must be chosen from {5, 6}")
        arms = int(rng.integers(low, high + 1))
        per_arm = np.full(arms, (count - 1) // arms, dtype=int)
        per_arm[: (count - 1) % arms] += 1
        return _rays(center, scale, theta, per_arm, 2.0 * np.pi / arms)
    if kind == "l_shape":
        fraction = float(rng.uniform(settings["l_shape_fraction"]["low"], settings["l_shape_fraction"]["high"]))
        first = max(1, min(count - 2, int(round((count - 1) * fraction))))
        second = count - 1 - first
        return _rays(center, scale, theta, np.asarray((first, second), dtype=int), np.pi / 2.0)
    if kind == "cross":
        if count < 5:
            raise ValueError("cross requires >=5 nodes")
        per_arm = np.full(4, (count - 1) // 4, dtype=int)
        per_arm[: (count - 1) % 4] += 1
        return _rays(center, scale, theta, per_arm, np.pi / 2.0)
    raise ValueError(f"unsupported motif: {kind}")


def structural_result(
    operator_name: str,
    parent: np.ndarray,
    child: np.ndarray | None,
    parameters: dict[str, Any],
    affected: np.ndarray,
    *,
    status: str,
    failure_reason: str,
    attempts: int,
    started: float,
) -> MutationResult:
    """Wrap a no-repair structural child in the shared mutation record."""

    final = parent.copy() if child is None else np.asarray(child, dtype=np.float64)
    indices = np.asarray(affected, dtype=int)
    mask = np.zeros(parent.shape[0], dtype=bool)
    if indices.size:
        mask[indices] = True
    accepted = status == "accepted"
    return MutationResult(
        operator_name=operator_name,
        input_coordinates=parent.copy(),
        raw_coordinates=final.copy(),
        final_coordinates=final.copy(),
        sampled_parameters=parameters,
        affected_indices=indices.copy(),
        affected_mask=mask,
        n_affected=int(indices.size),
        repair_indices=np.array([], dtype=int),
        repair_count=0,
        repair_fraction=0.0,
        valid_before_repair=accepted,
        valid_after_repair=accepted,
        proposal_status=status,
        failure_reason=failure_reason,
        input_hash=coordinate_hash(parent),
        raw_output_hash=coordinate_hash(final),
        final_output_hash=coordinate_hash(final),
        lineage_metadata={"operator_name": operator_name, "parameters": parameters, "attempts": attempts, "repair": "none"},
        timing={"total_operator_s": time.perf_counter() - started},
    )
