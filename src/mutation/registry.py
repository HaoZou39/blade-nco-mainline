"""The default Unified-12 registry.

linear_projection_mutation stays callable for an explicit FOGA11 replay.
It is not part of OPERATOR_NAMES and is not drawn by sample_operator_name.
"""

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
    stripe_reorganization,
    linear_projection_mutation,
    motif_assembly,
    normal_mutation,
    rotation_mutation,
    uniform_mutation,
)
from .parameters import load_named_profile
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
    """Raised when a name is not a known mutation operator."""


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
    (grid_mutation, "Reorganize points in a local box onto a grid."),
    (stripe_reorganization, "Blend selected points toward a rotated stripe template."),
    (motif_assembly, "Blend a local neighborhood toward one motif."),
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

_HISTORICAL: dict[str, OperatorSpec] = {
    linear_projection_mutation.OPERATOR_NAME: OperatorSpec(
        name=linear_projection_mutation.OPERATOR_NAME,
        apply=linear_projection_mutation.apply_operator,
        sample_parameters=linear_projection_mutation.sample_parameters,
        description="Project selected points near an arbitrary line.",
    )
}

FOGA11_OPERATOR_NAMES: tuple[str, ...] = (
    "uniform_mutation",
    "normal_mutation",
    "rotation_mutation",
    "implosion_mutation",
    "explosion_mutation",
    "cluster_mutation",
    "expansion_mutation",
    "compression_mutation",
    "axis_projection_mutation",
    "linear_projection_mutation",
    "grid_mutation",
)


def get_operator(name: str) -> OperatorSpec:
    spec = OPERATOR_REGISTRY.get(name) or _HISTORICAL.get(name)
    if spec is None:
        known = ", ".join(OPERATOR_NAMES)
        raise UnknownOperatorError(f"unknown operator {name!r}; expected one of: {known}")
    return spec


def sample_operator_name(rng: np.random.Generator) -> str:
    """Draw one default operator with equal probability. Repeats are allowed."""

    return str(rng.choice(np.asarray(OPERATOR_NAMES)))


def apply_foga11_operator(
    name: str,
    coordinates: np.ndarray,
    rng: np.random.Generator,
) -> MutationResult:
    """Replay one FOGA11_BASE_V1 operator, including linear_projection."""

    if name not in FOGA11_OPERATOR_NAMES:
        raise UnknownOperatorError(f"{name!r} is not a FOGA11 operator")
    profile = load_named_profile("FOGA11_BASE_V1")
    return apply_operator_with_config(name, coordinates, rng, profile)


def list_available_operators() -> list[str]:
    return list(OPERATOR_NAMES)


def sample_operator_parameters(
    name: str,
    coordinates: np.ndarray,
    rng: np.random.Generator,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return get_operator(name).sample_parameters(coordinates, rng, config)


def apply_operator(
    name: str,
    coordinates: np.ndarray,
    rng: np.random.Generator,
    params: Mapping[str, Any] | None = None,
) -> MutationResult:
    return get_operator(name).apply(coordinates, rng, params)


def apply_operator_with_config(
    name: str,
    coordinates: np.ndarray,
    rng: np.random.Generator,
    config: Mapping[str, Any],
) -> MutationResult:
    """Sample from a resolved profile, then use the unchanged operator path.

    Repair settings in named production profiles are required to match the
    canonical Base repair contract.  Sampling is explicit here so the exact
    materialized parameters remain part of ``MutationResult`` and can be
    reconstructed from the deterministic invocation seed.
    """

    if name in ("stripe_reorganization", "motif_assembly"):
        return get_operator(name).apply(coordinates, rng, None, config)
    spec = get_operator(name)
    params = spec.sample_parameters(coordinates, rng, config)
    return spec.apply(coordinates, rng, params)
