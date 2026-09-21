"""Uniformly relocate an explicit subset of points in the unit square."""

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


OPERATOR_NAME = "uniform_mutation"


def sample_parameters(
    coordinates: np.ndarray,
    rng: np.random.Generator,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    settings = resolve_operator_config(OPERATOR_NAME, config)
    fraction = sample_scalar(settings["affected_fraction"], rng)
    count = count_from_config(fraction, coordinates.shape[0])
    indices = sample_indices(coordinates.shape[0], count, rng)
    return {
        "affected_fraction": fraction,
        "affected_indices": indices.tolist(),
        "replacement_coordinates": rng.random((count, 2)).tolist(),
        "repair_seed": sample_repair_seed(rng),
    }


def _mutate(coordinates: np.ndarray, params: Mapping[str, Any]) -> RawMutation:
    validate_parameter_keys(
        params,
        required={"affected_indices", "replacement_coordinates", "repair_seed"},
        optional={"affected_fraction"},
    )
    indices = validate_indices(params["affected_indices"], coordinates.shape[0])
    replacements = np.asarray(params["replacement_coordinates"], dtype=np.float64)
    if replacements.shape != (indices.size, 2) or not np.isfinite(replacements).all():
        raise ValueError("replacement_coordinates must have shape (n_affected, 2)")
    if np.any((replacements < 0.0) | (replacements > 1.0)):
        raise ValueError("uniform replacements must lie in [0, 1]^2")
    raw = coordinates.copy()
    raw[indices] = replacements
    return RawMutation(
        raw,
        indices,
        {"replacement_coordinate_uniqueness": int(np.unique(replacements, axis=0).shape[0])},
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
