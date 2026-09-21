"""Reorganize points in a local box onto a rotated, jittered grid."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .base import (
    RawMutation,
    execute_operator,
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


OPERATOR_NAME = "grid_mutation"


def _grid_points(count: int, width: float, height: float, center: np.ndarray) -> np.ndarray:
    columns = int(np.ceil(np.sqrt(count * width / max(height, 1e-12))))
    rows = int(np.ceil(count / columns))
    xs = np.linspace(center[0] - width * 0.38, center[0] + width * 0.38, columns)
    ys = np.linspace(center[1] - height * 0.38, center[1] + height * 0.38, rows)
    return np.asarray([(x, y) for y in ys for x in xs], dtype=np.float64)[:count]


def sample_parameters(
    coordinates: np.ndarray,
    rng: np.random.Generator,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    settings = resolve_operator_config(OPERATOR_NAME, config)
    minimum = min(int(settings["minimum_affected_nodes"]), coordinates.shape[0])
    indices = np.array([], dtype=int)
    center = np.zeros(2)
    width = height = 0.0
    for _ in range(64):
        width = sample_scalar(settings["box_width"], rng)
        height = sample_scalar(settings["box_height"], rng)
        center = np.array(
            [
                rng.uniform(width / 2.0, 1.0 - width / 2.0),
                rng.uniform(height / 2.0, 1.0 - height / 2.0),
            ]
        )
        lower = center - np.array([width / 2.0, height / 2.0])
        upper = center + np.array([width / 2.0, height / 2.0])
        indices = np.flatnonzero(np.all((coordinates >= lower) & (coordinates <= upper), axis=1))
        if indices.size >= minimum:
            break
    if indices.size < minimum:
        raise RuntimeError("could not sample a sufficiently populated grid box")
    rotate = bool(rng.random() < float(settings["rotation_probability"]))
    angle = sample_scalar(settings["angle_degrees"], rng) if rotate else 0.0
    jitter = sample_scalar(settings["jitter"], rng)
    base_grid = _grid_points(indices.size, width, height, center)
    return {
        "box_center": center.tolist(),
        "width": width,
        "height": height,
        "angle": angle,
        "angle_unit": "degrees",
        "jitter": jitter,
        "affected_indices": indices.tolist(),
        "base_grid_points": base_grid.tolist(),
        "jitter_noise": rng.normal(0.0, jitter, size=(indices.size, 2)).tolist(),
        "repair_seed": sample_repair_seed(rng),
    }


def _mutate(coordinates: np.ndarray, params: Mapping[str, Any]) -> RawMutation:
    validate_parameter_keys(
        params,
        required={
            "box_center",
            "width",
            "height",
            "angle",
            "angle_unit",
            "jitter",
            "affected_indices",
            "base_grid_points",
            "jitter_noise",
            "repair_seed",
        },
    )
    indices = validate_indices(params["affected_indices"], coordinates.shape[0], minimum=2)
    center = validate_point(params["box_center"], "box_center")
    width = float(params["width"])
    height = float(params["height"])
    jitter = float(params["jitter"])
    if not np.isfinite(width) or not np.isfinite(height) or width <= 0.0 or height <= 0.0:
        raise ValueError("width and height must be positive and finite")
    if not np.isfinite(jitter) or jitter < 0.0:
        raise ValueError("jitter must be non-negative and finite")
    angle = float(params["angle"])
    unit = str(params["angle_unit"])
    if not np.isfinite(angle):
        raise ValueError("angle must be finite")
    if unit == "degrees":
        radians = np.deg2rad(angle)
    elif unit == "radians":
        radians = angle
    else:
        raise ValueError("angle_unit must be 'degrees' or 'radians'")
    base_grid = np.asarray(params["base_grid_points"], dtype=np.float64)
    noise = np.asarray(params["jitter_noise"], dtype=np.float64)
    expected_shape = (indices.size, 2)
    if base_grid.shape != expected_shape or not np.isfinite(base_grid).all():
        raise ValueError("base_grid_points must have shape (n_affected, 2)")
    if noise.shape != expected_shape or not np.isfinite(noise).all():
        raise ValueError("jitter_noise must have shape (n_affected, 2)")
    rotation = np.array(
        [
            [np.cos(radians), -np.sin(radians)],
            [np.sin(radians), np.cos(radians)],
        ]
    )
    target_grid = (base_grid - center) @ rotation.T + center
    raw = coordinates.copy()
    raw[indices] = target_grid + noise
    return RawMutation(
        raw,
        indices,
        {
            "box_center": center.tolist(),
            "width": width,
            "height": height,
            "angle_radians": float(radians),
            "target_grid_points": target_grid.tolist(),
            "grid_assignment_unique": bool(np.unique(target_grid, axis=0).shape[0] == indices.size),
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
