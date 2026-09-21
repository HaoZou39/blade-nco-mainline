"""Rotate selected points about an explicit center by an explicit angle."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .base import (
    RawMutation,
    count_from_config,
    execute_operator,
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


OPERATOR_NAME = "rotation_mutation"


def sample_parameters(
    coordinates: np.ndarray,
    rng: np.random.Generator,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    settings = resolve_operator_config(OPERATOR_NAME, config)
    fraction = sample_scalar(settings["affected_fraction"], rng)
    count = count_from_config(fraction, coordinates.shape[0], minimum=2)
    indices = sample_indices(coordinates.shape[0], count, rng)
    center = np.mean(coordinates[indices], axis=0)
    return {
        "affected_fraction": fraction,
        "affected_indices": indices.tolist(),
        "center": center.tolist(),
        "angle": sample_scalar(settings["angle_degrees"], rng),
        "angle_unit": "degrees",
        "repair_seed": sample_repair_seed(rng),
    }


def _mutate(coordinates: np.ndarray, params: Mapping[str, Any]) -> RawMutation:
    validate_parameter_keys(
        params,
        required={
            "affected_indices",
            "center",
            "angle",
            "angle_unit",
            "repair_seed",
        },
        optional={"affected_fraction"},
    )
    indices = validate_indices(params["affected_indices"], coordinates.shape[0], minimum=2)
    center = validate_point(params["center"], "center")
    angle = float(params["angle"])
    if not np.isfinite(angle):
        raise ValueError("angle must be finite")
    unit = str(params["angle_unit"])
    if unit == "degrees":
        radians = np.deg2rad(angle)
    elif unit == "radians":
        radians = angle
    else:
        raise ValueError("angle_unit must be 'degrees' or 'radians'")
    rotation = np.array(
        [
            [np.cos(radians), -np.sin(radians)],
            [np.sin(radians), np.cos(radians)],
        ],
        dtype=np.float64,
    )
    raw = coordinates.copy()
    raw[indices] = (coordinates[indices] - center) @ rotation.T + center
    radii_before = np.linalg.norm(coordinates[indices] - center, axis=1)
    radii_after = np.linalg.norm(raw[indices] - center, axis=1)
    return RawMutation(
        raw,
        indices,
        {
            "center": center.tolist(),
            "angle_radians": float(radians),
            "maximum_radius_error": float(np.max(np.abs(radii_after - radii_before))),
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
