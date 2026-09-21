"""Apply local Gaussian perturbations to an explicit subset of points."""

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


OPERATOR_NAME = "normal_mutation"


def sample_parameters(
    coordinates: np.ndarray,
    rng: np.random.Generator,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    settings = resolve_operator_config(OPERATOR_NAME, config)
    fraction = sample_scalar(settings["affected_fraction"], rng)
    sigma = sample_scalar(settings["sigma"], rng)
    count = count_from_config(fraction, coordinates.shape[0])
    indices = sample_indices(coordinates.shape[0], count, rng)
    return {
        "affected_fraction": fraction,
        "sigma": sigma,
        "affected_indices": indices.tolist(),
        "noise": rng.normal(0.0, sigma, size=(count, 2)).tolist(),
        "repair_seed": sample_repair_seed(rng),
    }


def _mutate(coordinates: np.ndarray, params: Mapping[str, Any]) -> RawMutation:
    validate_parameter_keys(
        params,
        required={"affected_indices", "sigma", "noise", "repair_seed"},
        optional={"affected_fraction"},
    )
    indices = validate_indices(params["affected_indices"], coordinates.shape[0])
    sigma = float(params["sigma"])
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be positive and finite")
    noise = np.asarray(params["noise"], dtype=np.float64)
    if noise.shape != (indices.size, 2) or not np.isfinite(noise).all():
        raise ValueError("noise must have shape (n_affected, 2)")
    raw = coordinates.copy()
    raw[indices] += noise
    return RawMutation(raw, indices, {"sigma": sigma})


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
