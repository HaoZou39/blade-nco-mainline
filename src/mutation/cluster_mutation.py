"""Reorganize selected points into a Gaussian cluster."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .base import (
    RawMutation,
    execute_operator,
    sample_indices,
    validate_indices,
    validate_parameter_keys,
    validate_point,
)
from .parameters import (
    resolve_operator_config,
    sample_integer,
    sample_repair_seed,
    sample_scalar,
)
from .result import MutationResult


OPERATOR_NAME = "cluster_mutation"


def sample_parameters(
    coordinates: np.ndarray,
    rng: np.random.Generator,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    settings = resolve_operator_config(OPERATOR_NAME, config)
    count = min(sample_integer(settings["affected_count"], rng), coordinates.shape[0])
    indices = sample_indices(coordinates.shape[0], count, rng)
    center = rng.random(2)
    sigma = sample_scalar(settings["sigma"], rng)
    return {
        "affected_count": count,
        "affected_indices": indices.tolist(),
        "center": center.tolist(),
        "sigma": sigma,
        "offsets": rng.normal(0.0, sigma, size=(count, 2)).tolist(),
        "repair_seed": sample_repair_seed(rng),
    }


def _mutate(coordinates: np.ndarray, params: Mapping[str, Any]) -> RawMutation:
    validate_parameter_keys(
        params,
        required={
            "affected_indices",
            "center",
            "sigma",
            "offsets",
            "repair_seed",
        },
        optional={"affected_count"},
    )
    indices = validate_indices(params["affected_indices"], coordinates.shape[0])
    center = validate_point(params["center"], "center")
    sigma = float(params["sigma"])
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be positive and finite")
    offsets = np.asarray(params["offsets"], dtype=np.float64)
    if offsets.shape != (indices.size, 2) or not np.isfinite(offsets).all():
        raise ValueError("offsets must have shape (n_affected, 2)")
    raw = coordinates.copy()
    raw[indices] = center + offsets
    return RawMutation(
        raw,
        indices,
        {
            "center": center.tolist(),
            "sigma": sigma,
            "covariance_trace": float(np.trace(np.cov(raw[indices], rowvar=False)))
            if indices.size > 1
            else 0.0,
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
