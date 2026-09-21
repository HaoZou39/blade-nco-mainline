"""The single runtime registry for the eleven formal mutation operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from . import (
    axis_projection_mutation,
    cluster_mutation,
    compression_mutation,
    expansion_mutation,
    explosion_mutation,
    grid_mutation,
    implosion_mutation,
    linear_projection_mutation,
    normal_mutation,
    rotation_mutation,
    uniform_mutation,
)
from .result import MutationResult


ApplyFunction = Callable[
    [np.ndarray, np.random.Generator, Mapping[str, Any] | None],
    MutationResult,
]
Sampler = Callable[
    [np.ndarray, np.random.Generator, Mapping[str, Any] | None],
    dict[str, Any],
]


class UnknownOperatorError(KeyError):
    """Raised when a name is not one of the eleven formal operator names."""


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    apply: ApplyFunction
    sample_parameters: Sampler
    description: str


_MODULES = (
    (uniform_mutation, "Relocate selected points uniformly in the unit square."),
    (normal_mutation, "Apply local Gaussian perturbations to selected points."),
    (rotation_mutation, "Rotate selected points around an explicit center."),
    (implosion_mutation, "Contract points in a local region toward its center."),
    (explosion_mutation, "Push points in a local region beyond its boundary."),
    (cluster_mutation, "Reorganize selected points into a Gaussian cluster."),
    (expansion_mutation, "Push points inside a tube away from its target line."),
    (compression_mutation, "Attract points inside a tube toward its target line."),
    (axis_projection_mutation, "Project selected points near an axis-parallel line."),
    (linear_projection_mutation, "Project selected points near an arbitrary line."),
    (grid_mutation, "Reorganize points in a local box onto a grid."),
)


OPERATOR_REGISTRY: dict[str, OperatorSpec] = {
    module.OPERATOR_NAME: OperatorSpec(
        name=module.OPERATOR_NAME,
        apply=module.apply_operator,
        sample_parameters=module.sample_parameters,
        description=description,
    )
    for module, description in _MODULES
}

OPERATOR_NAMES: tuple[str, ...] = tuple(OPERATOR_REGISTRY)


def get_operator(name: str) -> OperatorSpec:
    try:
        return OPERATOR_REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(OPERATOR_NAMES)
        raise UnknownOperatorError(
            f"unknown operator {name!r}; expected one of: {known}"
        ) from exc


def list_available_operators() -> list[str]:
    return list(OPERATOR_NAMES)


def sample_operator_parameters(
    name: str,
    coordinates: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, Any]:
    return get_operator(name).sample_parameters(coordinates, rng, None)


def apply_operator(
    name: str,
    coordinates: np.ndarray,
    rng: np.random.Generator,
    params: Mapping[str, Any] | None = None,
) -> MutationResult:
    return get_operator(name).apply(coordinates, rng, params)
