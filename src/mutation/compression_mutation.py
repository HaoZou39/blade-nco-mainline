"""Attract points inside a line-centered tube toward the target line."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .base import (
    RawMutation,
    execute_operator,
    normalize_direction,
    project_to_line,
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


OPERATOR_NAME = "compression_mutation"


def sample_parameters(
    coordinates: np.ndarray,
    rng: np.random.Generator,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    settings = resolve_operator_config(OPERATOR_NAME, config)
    tube_width = sample_scalar(settings["tube_width"], rng)
    indices = np.array([], dtype=int)
    line_point = np.zeros(2)
    line_direction = np.ones(2)
    for _ in range(64):
        angle = np.deg2rad(sample_scalar(settings["line_angle_degrees"], rng))
        line_point = rng.random(2)
        line_direction = np.array([np.cos(angle), np.sin(angle)])
        _, signed_distances, _ = project_to_line(
            coordinates,
            line_point,
            line_direction,
        )
        indices = np.flatnonzero(np.abs(signed_distances) < tube_width)
        if indices.size >= min(2, coordinates.shape[0]):
            break
    if indices.size == 0:
        raise RuntimeError("could not sample a non-empty compression tube")
    return {
        "line_point": line_point.tolist(),
        "line_direction": line_direction.tolist(),
        "tube_width": tube_width,
        "strength": sample_scalar(settings["strength"], rng),
        "affected_indices": indices.tolist(),
        "repair_seed": sample_repair_seed(rng),
    }


def _mutate(coordinates: np.ndarray, params: Mapping[str, Any]) -> RawMutation:
    validate_parameter_keys(
        params,
        required={
            "line_point",
            "line_direction",
            "tube_width",
            "strength",
            "affected_indices",
            "repair_seed",
        },
    )
    indices = validate_indices(params["affected_indices"], coordinates.shape[0])
    line_point = validate_point(params["line_point"], "line_point")
    line_direction = normalize_direction(params["line_direction"])
    tube_width = float(params["tube_width"])
    strength = float(params["strength"])
    if not np.isfinite(tube_width) or tube_width <= 0.0:
        raise ValueError("tube_width must be positive and finite")
    if not np.isfinite(strength) or not 0.0 < strength < 1.0:
        raise ValueError("strength must lie in (0, 1)")
    projections, signed_distances, _ = project_to_line(
        coordinates,
        line_point,
        line_direction,
    )
    before = np.abs(signed_distances[indices])
    if np.any(before >= tube_width + 1e-12):
        raise ValueError("affected_indices must lie inside the compression tube")
    raw = coordinates.copy()
    raw[indices] = coordinates[indices] + strength * (
        projections[indices] - coordinates[indices]
    )
    _, raw_signed, _ = project_to_line(raw, line_point, line_direction)
    after = np.abs(raw_signed[indices])
    return RawMutation(
        raw,
        indices,
        {
            "line_point": line_point.tolist(),
            "line_direction": line_direction.tolist(),
            "tube_width": tube_width,
            "mean_distance_before": float(np.mean(before)),
            "mean_distance_after": float(np.mean(after)),
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
