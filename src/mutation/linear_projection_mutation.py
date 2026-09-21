"""Orthogonally project selected points near an arbitrary target line."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .base import (
    RawMutation,
    count_from_config,
    execute_operator,
    normalize_direction,
    project_to_line,
    sample_indices,
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


OPERATOR_NAME = "linear_projection_mutation"


def sample_parameters(
    coordinates: np.ndarray,
    rng: np.random.Generator,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    settings = resolve_operator_config(OPERATOR_NAME, config)
    fraction = sample_scalar(settings["affected_fraction"], rng)
    count = count_from_config(fraction, coordinates.shape[0])
    indices = sample_indices(coordinates.shape[0], count, rng)
    angle = np.deg2rad(sample_scalar(settings["line_angle_degrees"], rng))
    jitter = sample_scalar(settings["jitter"], rng)
    return {
        "affected_fraction": fraction,
        "affected_indices": indices.tolist(),
        "line_point": rng.random(2).tolist(),
        "line_direction": [float(np.cos(angle)), float(np.sin(angle))],
        "jitter": jitter,
        "jitter_noise": rng.normal(0.0, jitter, size=count).tolist(),
        "repair_seed": sample_repair_seed(rng),
    }


def _mutate(coordinates: np.ndarray, params: Mapping[str, Any]) -> RawMutation:
    validate_parameter_keys(
        params,
        required={
            "affected_indices",
            "line_point",
            "line_direction",
            "jitter",
            "jitter_noise",
            "repair_seed",
        },
        optional={"affected_fraction"},
    )
    indices = validate_indices(params["affected_indices"], coordinates.shape[0])
    line_point = validate_point(params["line_point"], "line_point")
    line_direction = normalize_direction(params["line_direction"])
    jitter = float(params["jitter"])
    if not np.isfinite(jitter) or jitter < 0.0:
        raise ValueError("jitter must be non-negative and finite")
    noise = np.asarray(params["jitter_noise"], dtype=np.float64)
    if noise.shape != (indices.size,) or not np.isfinite(noise).all():
        raise ValueError("jitter_noise must have shape (n_affected,)")
    projections, signed_distances, normal = project_to_line(
        coordinates,
        line_point,
        line_direction,
    )
    raw = coordinates.copy()
    raw[indices] = projections[indices] + noise[:, None] * normal
    _, after_signed, _ = project_to_line(raw, line_point, line_direction)
    return RawMutation(
        raw,
        indices,
        {
            "line_point": line_point.tolist(),
            "line_direction": line_direction.tolist(),
            "mean_distance_before": float(np.mean(np.abs(signed_distances[indices]))),
            "mean_distance_after": float(np.mean(np.abs(after_signed[indices]))),
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
