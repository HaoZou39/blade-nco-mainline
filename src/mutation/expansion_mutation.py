"""Push points inside a line-centered tube away from the target line."""

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


OPERATOR_NAME = "expansion_mutation"


def _sample_line(
    settings: Mapping[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    angle = np.deg2rad(sample_scalar(settings["line_angle_degrees"], rng))
    return rng.random(2), np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)


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
        line_point, line_direction = _sample_line(settings, rng)
        _, signed_distances, _ = project_to_line(
            coordinates,
            line_point,
            line_direction,
        )
        indices = np.flatnonzero(np.abs(signed_distances) < tube_width)
        if indices.size >= min(2, coordinates.shape[0]):
            break
    if indices.size == 0:
        raise RuntimeError("could not sample a non-empty expansion tube")
    extra_spec = settings["extra_distance"]
    extra = np.minimum(
        rng.exponential(float(extra_spec["scale"]), size=indices.size),
        float(extra_spec["maximum"]),
    )
    return {
        "line_point": line_point.tolist(),
        "line_direction": line_direction.tolist(),
        "tube_width": tube_width,
        "affected_indices": indices.tolist(),
        "extra_distances": extra.tolist(),
        "repair_seed": sample_repair_seed(rng),
    }


def _mutate(coordinates: np.ndarray, params: Mapping[str, Any]) -> RawMutation:
    validate_parameter_keys(
        params,
        required={
            "line_point",
            "line_direction",
            "tube_width",
            "affected_indices",
            "extra_distances",
            "repair_seed",
        },
    )
    indices = validate_indices(params["affected_indices"], coordinates.shape[0])
    line_point = validate_point(params["line_point"], "line_point")
    line_direction = normalize_direction(params["line_direction"])
    tube_width = float(params["tube_width"])
    if not np.isfinite(tube_width) or tube_width <= 0.0:
        raise ValueError("tube_width must be positive and finite")
    projections, signed_distances, normal = project_to_line(
        coordinates,
        line_point,
        line_direction,
    )
    before = np.abs(signed_distances[indices])
    if np.any(before >= tube_width + 1e-12):
        raise ValueError("affected_indices must lie inside the expansion tube")
    extra = np.asarray(params["extra_distances"], dtype=np.float64)
    if extra.shape != (indices.size,) or not np.isfinite(extra).all():
        raise ValueError("extra_distances must have shape (n_affected,)")
    if np.any(extra < 0.0):
        raise ValueError("extra_distances must be non-negative")
    signs = np.where(signed_distances[indices] < 0.0, -1.0, 1.0)
    raw = coordinates.copy()
    raw[indices] = projections[indices] + (
        signs * (tube_width + extra)
    )[:, None] * normal
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
