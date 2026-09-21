"""Push points inside a local radius beyond its boundary."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .base import (
    RawMutation,
    execute_operator,
    validate_indices,
    validate_parameter_keys,
    validate_point,
)
from .parameters import (
    resolve_operator_config,
    sample_repair_seed,
    sample_scalar,
)
from .result import MutationResult


OPERATOR_NAME = "explosion_mutation"


def sample_parameters(
    coordinates: np.ndarray,
    rng: np.random.Generator,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    settings = resolve_operator_config(OPERATOR_NAME, config)
    radius = sample_scalar(settings["radius"], rng)
    center = rng.random(2)
    distances = np.linalg.norm(coordinates - center, axis=1)
    indices = np.flatnonzero((distances <= radius) & (distances > 1e-12))
    if indices.size == 0:
        center = np.array([0.5, 0.5], dtype=np.float64)
        distances = np.linalg.norm(coordinates - center, axis=1)
        nearest = np.argsort(distances)[: max(1, min(8, coordinates.shape[0]))]
        radius = float(np.max(distances[nearest]) + 1e-9)
        indices = nearest[distances[nearest] > 1e-12]
    extra_spec = settings["extra_shift"]
    extra_shifts = np.minimum(
        rng.exponential(float(extra_spec["scale"]), size=indices.size),
        float(extra_spec["maximum"]),
    )
    return {
        "center": center.tolist(),
        "radius": radius,
        "affected_indices": indices.tolist(),
        "extra_shifts": extra_shifts.tolist(),
        "repair_seed": sample_repair_seed(rng),
    }


def _mutate(coordinates: np.ndarray, params: Mapping[str, Any]) -> RawMutation:
    validate_parameter_keys(
        params,
        required={
            "center",
            "radius",
            "affected_indices",
            "extra_shifts",
            "repair_seed",
        },
    )
    indices = validate_indices(params["affected_indices"], coordinates.shape[0])
    center = validate_point(params["center"], "center")
    radius = float(params["radius"])
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be positive and finite")
    vectors = coordinates[indices] - center
    distances = np.linalg.norm(vectors, axis=1)
    if np.any(distances <= 1e-12):
        raise ValueError("explosion center must not coincide with an affected point")
    if np.any(distances > radius + 1e-12):
        raise ValueError("affected_indices must lie inside the explosion radius")
    extra_shifts = np.asarray(params["extra_shifts"], dtype=np.float64)
    if extra_shifts.shape != (indices.size,) or not np.isfinite(extra_shifts).all():
        raise ValueError("extra_shifts must have shape (n_affected,)")
    if np.any(extra_shifts < 0.0):
        raise ValueError("extra_shifts must be non-negative")
    directions = vectors / distances[:, None]
    raw = coordinates.copy()
    raw[indices] = center + directions * (radius + extra_shifts)[:, None]
    after_distances = np.linalg.norm(raw[indices] - center, axis=1)
    return RawMutation(
        raw,
        indices,
        {
            "center": center.tolist(),
            "radius": radius,
            "mean_distance_before": float(np.mean(distances)),
            "mean_distance_after": float(np.mean(after_distances)),
        },
    )


def apply_operator(
    coordinates: np.ndarray,
    rng: np.random.Generator,
    params: Mapping[str, Any] | None = None,
) -> MutationResult:
    return execute_operator(
        OPERATOR_NAME,
        coordinates,
        rng,
        params,
        sample_parameters,
        _mutate,
    )
