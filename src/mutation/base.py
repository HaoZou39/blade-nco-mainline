"""Shared validation, geometry, and execution contract for all operators."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Mapping
import time

import numpy as np

from .parameters import repair_config
from .repair_adapter import check_validity, repair_coordinates
from .result import MutationResult, coordinate_hash


@dataclass(frozen=True)
class RawMutation:
    coordinates: np.ndarray
    affected_indices: np.ndarray
    metadata: Mapping[str, Any]


Sampler = Callable[
    [np.ndarray, np.random.Generator, Mapping[str, Any] | None],
    dict[str, Any],
]
Mutator = Callable[[np.ndarray, Mapping[str, Any]], RawMutation]


def validate_coordinates(coordinates: np.ndarray) -> np.ndarray:
    """Return a defensive float64 copy of a valid unit-square point set."""

    array = np.asarray(coordinates, dtype=np.float64)
    if array.ndim != 2 or array.shape[1:] != (2,) or array.shape[0] < 2:
        raise ValueError(f"coordinates must have shape (N, 2), N >= 2; got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("coordinates must contain only finite values")
    if np.any((array < 0.0) | (array > 1.0)):
        raise ValueError("input coordinates must lie in [0, 1]^2")
    return array.copy()


def validate_parameter_keys(
    params: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    """Reject missing and silently ignored operator parameters."""

    optional = optional or set()
    keys = set(params)
    missing = required - keys
    unknown = keys - required - optional
    if missing:
        raise ValueError(f"missing parameters: {sorted(missing)}")
    if unknown:
        raise ValueError(f"unknown parameters: {sorted(unknown)}")


def validate_indices(
    indices: Any,
    n_points: int,
    *,
    minimum: int = 1,
) -> np.ndarray:
    """Validate an ordered, unique integer index vector."""

    array = np.asarray(indices)
    if array.ndim != 1 or array.size < minimum:
        raise ValueError(f"affected_indices must contain at least {minimum} entries")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError("affected_indices must be integers")
    array = array.astype(int)
    if np.any((array < 0) | (array >= n_points)):
        raise ValueError("affected_indices contain an out-of-range index")
    if np.unique(array).size != array.size:
        raise ValueError("affected_indices must be unique")
    return array


def count_from_config(fraction: float, n_points: int, *, minimum: int = 1) -> int:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("affected_fraction must lie in (0, 1]")
    return int(np.clip(round(float(fraction) * n_points), minimum, n_points))


def sample_indices(
    n_points: int,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if not 1 <= count <= n_points:
        raise ValueError("affected_count must lie in [1, N]")
    return np.asarray(rng.choice(n_points, size=count, replace=False), dtype=int)


def normalize_direction(direction: Any) -> np.ndarray:
    vector = np.asarray(direction, dtype=np.float64)
    if vector.shape != (2,) or not np.isfinite(vector).all():
        raise ValueError("line_direction must contain two finite values")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("line_direction must be non-zero")
    return vector / norm


def validate_point(point: Any, name: str) -> np.ndarray:
    array = np.asarray(point, dtype=np.float64)
    if array.shape != (2,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain two finite values")
    return array


def project_to_line(
    coordinates: np.ndarray,
    line_point: Any,
    line_direction: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    point = validate_point(line_point, "line_point")
    direction = normalize_direction(line_direction)
    normal = np.array([-direction[1], direction[0]], dtype=np.float64)
    offsets = coordinates - point
    signed_distances = offsets @ normal
    projections = coordinates - signed_distances[:, None] * normal
    return projections, signed_distances, normal


def jsonable(value: Any) -> Any:
    """Convert nested NumPy values to stable JSON/YAML-compatible values."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return deepcopy(value)


def execute_operator(
    operator_name: str,
    coordinates: np.ndarray,
    rng: np.random.Generator,
    params: Mapping[str, Any] | None,
    sampler: Sampler,
    mutator: Mutator,
    *,
    config: Mapping[str, Any] | None = None,
) -> MutationResult:
    """Apply raw mutation, deterministic repair, gates, hashes, and lineage."""

    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be numpy.random.Generator")
    total_started = time.perf_counter()
    started = time.perf_counter(); input_coordinates = validate_coordinates(coordinates); input_validation_s = time.perf_counter()-started
    started = time.perf_counter()
    materialized = (
        sampler(input_coordinates, rng, config)
        if params is None
        else jsonable(dict(params))
    )
    sampling_s = time.perf_counter()-started
    if "repair_seed" not in materialized:
        raise ValueError("explicit parameters must include repair_seed")
    started = time.perf_counter(); raw = mutator(input_coordinates, materialized); raw_mutation_s = time.perf_counter()-started
    raw_coordinates = np.asarray(raw.coordinates, dtype=np.float64)
    if raw_coordinates.shape != input_coordinates.shape:
        raise ValueError("raw mutation changed coordinate shape")
    if not np.isfinite(raw_coordinates).all():
        raise ValueError("raw mutation produced non-finite coordinates")
    affected_indices = validate_indices(
        raw.affected_indices,
        input_coordinates.shape[0],
    )
    settings = repair_config(config)
    minimum_distance = float(settings["min_pair_distance"])
    started = time.perf_counter(); before = check_validity(
        raw_coordinates,
        min_pair_distance=minimum_distance,
    )
    raw_validation_s = time.perf_counter()-started
    started = time.perf_counter(); repair = repair_coordinates(
        raw_coordinates,
        repair_seed=int(materialized["repair_seed"]),
        min_pair_distance=minimum_distance,
        max_passes=int(settings["max_passes"]),
    )
    repair_s = time.perf_counter()-started
    repair_fraction = float(repair.indices.size / input_coordinates.shape[0])
    max_fraction = float(settings["max_repair_fraction"])
    failure_reason = ""
    if repair_fraction > max_fraction:
        failure_reason = "excessive_repair"
    elif not repair.after.valid:
        failure_reason = "final_instance_invalid"
    proposal_status = "accepted" if not failure_reason else "failed"
    mask = np.zeros(input_coordinates.shape[0], dtype=bool)
    mask[affected_indices] = True
    lineage = {
        "operator_name": operator_name,
        "parameters": jsonable(materialized),
        "raw_metadata": jsonable(raw.metadata),
        "repair_reasons": repair.reasons,
    }
    started = time.perf_counter()
    result = MutationResult(
        operator_name=operator_name,
        input_coordinates=input_coordinates.copy(),
        raw_coordinates=raw_coordinates.copy(),
        final_coordinates=repair.coordinates.copy(),
        sampled_parameters=jsonable(materialized),
        affected_indices=affected_indices.copy(),
        affected_mask=mask,
        n_affected=int(affected_indices.size),
        repair_indices=repair.indices.copy(),
        repair_count=int(repair.indices.size),
        repair_fraction=repair_fraction,
        valid_before_repair=before.valid,
        valid_after_repair=repair.after.valid,
        proposal_status=proposal_status,
        failure_reason=failure_reason,
        input_hash=coordinate_hash(input_coordinates),
        raw_output_hash=coordinate_hash(raw_coordinates),
        final_output_hash=coordinate_hash(repair.coordinates),
        lineage_metadata=lineage,
        timing={},
    )
    finalize_hash_s = time.perf_counter()-started
    from dataclasses import replace
    return replace(result,timing={"input_validation_s":input_validation_s,"parameter_sampling_s":sampling_s,
        "raw_mutation_s":raw_mutation_s,"raw_validation_s":raw_validation_s,"repair_s":repair_s,
        "finalize_and_hash_s":finalize_hash_s,"total_operator_s":time.perf_counter()-total_started})
