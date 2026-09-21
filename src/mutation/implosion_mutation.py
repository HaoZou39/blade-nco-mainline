"""Contract points inside a local radius toward an explicit center."""

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


OPERATOR_NAME = "implosion_mutation"


def sample_parameters(
    coordinates: np.ndarray,
    rng: np.random.Generator,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    settings = resolve_operator_config(OPERATOR_NAME, config)
    radius = sample_scalar(settings["radius"], rng)
    center = rng.random(2)
    indices = np.flatnonzero(np.linalg.norm(coordinates - center, axis=1) <= radius)
    if indices.size == 0:
        anchor = int(rng.integers(0, coordinates.shape[0]))
        center = coordinates[anchor].copy()
        indices = np.flatnonzero(np.linalg.norm(coordinates - center, axis=1) <= radius)
    return {
        "center": center.tolist(),
        "radius": radius,
        "strength": sample_scalar(settings["strength"], rng),
        "affected_indices": indices.tolist(),
        "repair_seed": sample_repair_seed(rng),
    }


def _mutate(coordinates: np.ndarray, params: Mapping[str, Any]) -> RawMutation:
    validate_parameter_keys(
        params,
        required={"center", "radius", "strength", "affected_indices", "repair_seed"},
    )
    indices = validate_indices(params["affected_indices"], coordinates.shape[0])
    center = validate_point(params["center"], "center")
    radius = float(params["radius"])
    strength = float(params["strength"])
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be positive and finite")
    if not np.isfinite(strength) or not 0.0 < strength <= 1.0:
        raise ValueError("strength must lie in (0, 1]")
    before_distances = np.linalg.norm(coordinates[indices] - center, axis=1)
    if np.any(before_distances > radius + 1e-12):
        raise ValueError("affected_indices must lie inside the implosion radius")
    raw = coordinates.copy()
    raw[indices] = coordinates[indices] + strength * (center - coordinates[indices])
    after_distances = np.linalg.norm(raw[indices] - center, axis=1)
    return RawMutation(
        raw,
        indices,
        {
            "center": center.tolist(),
            "radius": radius,
            "mean_distance_before": float(np.mean(before_distances)),
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
