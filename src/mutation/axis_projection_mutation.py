"""Project selected points near a horizontal or vertical target line."""

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
)
from .parameters import (
    resolve_operator_config,
    sample_repair_seed,
    sample_scalar,
)
from .result import MutationResult


OPERATOR_NAME = "axis_projection_mutation"


def sample_parameters(
    coordinates: np.ndarray,
    rng: np.random.Generator,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    settings = resolve_operator_config(OPERATOR_NAME, config)
    fraction = sample_scalar(settings["affected_fraction"], rng)
    count = count_from_config(fraction, coordinates.shape[0])
    indices = sample_indices(coordinates.shape[0], count, rng)
    jitter = sample_scalar(settings["jitter"], rng)
    return {
        "affected_fraction": fraction,
        "affected_indices": indices.tolist(),
        "axis": str(rng.choice(settings["axis"]["values"])),
        "target": sample_scalar(settings["target"], rng),
        "jitter": jitter,
        "jitter_noise": rng.normal(0.0, jitter, size=count).tolist(),
        "repair_seed": sample_repair_seed(rng),
    }


def _mutate(coordinates: np.ndarray, params: Mapping[str, Any]) -> RawMutation:
    validate_parameter_keys(
        params,
        required={
            "affected_indices",
            "axis",
            "target",
            "jitter",
            "jitter_noise",
            "repair_seed",
        },
        optional={"affected_fraction"},
    )
    indices = validate_indices(params["affected_indices"], coordinates.shape[0])
    axis = str(params["axis"])
    if axis not in {"x", "y"}:
        raise ValueError("axis must be 'x' or 'y'")
    target = float(params["target"])
    jitter = float(params["jitter"])
    if not np.isfinite(target) or not 0.0 <= target <= 1.0:
        raise ValueError("target must lie in [0, 1]")
    if not np.isfinite(jitter) or jitter < 0.0:
        raise ValueError("jitter must be non-negative and finite")
    noise = np.asarray(params["jitter_noise"], dtype=np.float64)
    if noise.shape != (indices.size,) or not np.isfinite(noise).all():
        raise ValueError("jitter_noise must have shape (n_affected,)")
    dimension = 0 if axis == "x" else 1
    before = np.abs(coordinates[indices, dimension] - target)
    raw = coordinates.copy()
    raw[indices, dimension] = target + noise
    after = np.abs(raw[indices, dimension] - target)
    return RawMutation(
        raw,
        indices,
        {
            "axis": axis,
            "target": target,
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
