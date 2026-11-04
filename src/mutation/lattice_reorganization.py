"""Frozen lattice reorganization from Library A.

One call draws one template. A failed draw is not repaired. Up to three
independent draws are attempted, matching the audited Unified-12 call.
"""

from __future__ import annotations

from typing import Any, Mapping
import time

import numpy as np

from .parameters import resolve_operator_config, sample_integer, sample_scalar
from .structural_geometry import contain, match, rotate, spacing, structural_result, valid_child


OPERATOR_NAME = "lattice_reorganization"


def _once(coordinates: np.ndarray, rng: np.random.Generator, settings: Mapping[str, Any]) -> dict[str, Any]:
    scope = str(rng.choice(tuple(settings["scope"]["values"]), p=tuple(settings["scope"]["probabilities"])))
    template = str(rng.choice(tuple(settings["template"]["values"])))
    angle = float(rng.uniform(settings["angle_degrees"]["low"], settings["angle_degrees"]["high"]))
    alpha = sample_scalar(settings["alpha"], rng)
    if scope == "global":
        selected = np.arange(coordinates.shape[0])
        center = np.array([0.5, 0.5])
        width = height = 1.0
    else:
        side_spec = settings["local_side"] if scope == "local" else settings["regional_side"]
        side = sample_scalar(side_spec, rng)
        center = np.array([
            rng.uniform(side / 2.0, 1.0 - side / 2.0),
            rng.uniform(side / 2.0, 1.0 - side / 2.0),
        ])
        lower, upper = center - side / 2.0, center + side / 2.0
        selected = np.flatnonzero(np.all((coordinates >= lower) & (coordinates <= upper), axis=1))
        width = height = side
    parameters: dict[str, Any] = {
        "scope": scope,
        "template": template,
        "angle_degrees": angle,
        "alpha": alpha,
        "center": center.tolist(),
        "status": "failed",
    }
    if selected.size < int(settings["minimum_selected"]):
        parameters["failure_reason"] = "empty_region"
        return parameters
    count = int(selected.size)
    interior = tuple(settings["interior"])
    if template == "grid":
        columns = max(1, int(np.round(np.sqrt(count * width / max(height, 1e-12)))))
        rows = int(np.ceil(count / columns))
        span = float(settings["grid_half_span"])
        x0, x1 = center[0] - width * span, center[0] + width * span
        y0, y1 = center[1] - height * span, center[1] + height * span
        xs = np.linspace(x0, x1, columns)
        ys = np.linspace(y0, y1, rows)
        sites = np.asarray([(x, y) for y in ys for x in xs], dtype=np.float64)[:count]
        parameters.update({"rows": rows, "columns": columns, "stripe_axis": "", "stripe_count": 0})
    else:
        stripe_axis = str(rng.choice(tuple(settings["stripe_axis"]["values"])))
        stripe_count = sample_integer(settings["stripe_count"], rng)
        tangent = 0 if stripe_axis == "y" else 1
        order = np.argsort(coordinates[selected, tangent], kind="mergesort")
        blocks = np.array_split(order, stripe_count)
        sites = np.empty((count, 2), dtype=np.float64)
        if scope == "global":
            low_b, high_b = interior
        else:
            low_b, high_b = float(center[0] - width / 2.0), float(center[0] + width / 2.0)
        normal_low, normal_high = (center[1] - height / 2.0, center[1] + height / 2.0)
        if stripe_axis == "x":
            low_b, high_b = float(center[1] - height / 2.0), float(center[1] + height / 2.0)
            normal_low, normal_high = center[0] - width / 2.0, center[0] + width / 2.0
        normal_positions = np.linspace(normal_low, normal_high, stripe_count)
        cursor = 0
        for block, normal_value in zip(blocks, normal_positions):
            if block.size == 0:
                continue
            along = np.linspace(low_b, high_b, block.size)
            for offset, along_value in enumerate(along):
                point = [normal_value, along_value] if stripe_axis == "x" else [along_value, normal_value]
                sites[cursor + offset] = point
            cursor += block.size
        sites = sites[:cursor]
        parameters.update({
            "rows": 0,
            "columns": 0,
            "stripe_axis": stripe_axis,
            "stripe_count": stripe_count,
            "tangent_order": order.tolist(),
        })
        if sites.shape[0] != count:
            parameters["failure_reason"] = "stripe_count_mismatch"
            return parameters
    sites = rotate(sites, center, angle)
    sites = contain(sites, center, interior[0], interior[1])
    gap = spacing(sites)
    parameters["spacing"] = gap
    if gap < float(settings["minimum_template_spacing"]):
        parameters["failure_reason"] = "spacing"
        return parameters
    jitter_scale = float(rng.uniform(0.0, gap / 4.0))
    sites = sites + rng.normal(0.0, jitter_scale, size=sites.shape)
    parameters["jitter_scale"] = jitter_scale
    matched = match(coordinates[selected], sites)
    child = coordinates.copy()
    child[selected] = (1.0 - alpha) * coordinates[selected] + alpha * matched
    reason = valid_child(child, node_count=int(settings["node_count"]), min_pair=float(settings["min_pair_distance"]))
    parameters["n_affected"] = count
    parameters["affected_indices"] = selected.astype(int).tolist()
    parameters["matched"] = matched.tolist()
    if reason:
        parameters["failure_reason"] = reason
        return parameters
    parameters["status"] = "accepted"
    parameters["failure_reason"] = ""
    parameters["_child"] = child
    return parameters


def sample_parameters(
    coordinates: np.ndarray,
    rng: np.random.Generator,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    drawn = _once(np.asarray(coordinates, dtype=np.float64), rng, resolve_operator_config(OPERATOR_NAME, config))
    drawn.pop("_child", None)
    return drawn


def _from_explicit(coordinates: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
    indices = np.asarray(params["affected_indices"], dtype=int)
    matched = np.asarray(params["matched"], dtype=np.float64)
    alpha = float(params["alpha"])
    child = coordinates.copy()
    child[indices] = (1.0 - alpha) * coordinates[indices] + alpha * matched
    return child


def apply_operator(
    coordinates: np.ndarray,
    rng: np.random.Generator,
    params: Mapping[str, Any] | None = None,
) -> MutationResult:
    started = time.perf_counter()
    parent = np.array(coordinates, dtype=np.float64, copy=True)
    if params is not None:
        child = _from_explicit(parent, params)
        stored = {key: value for key, value in params.items() if key != "_child"}
        return structural_result(
            OPERATOR_NAME, parent, child, stored,
            np.asarray(params["affected_indices"], dtype=int),
            status="accepted", failure_reason="", attempts=1, started=started,
        )
    settings = resolve_operator_config(OPERATOR_NAME, None)
    last: dict[str, Any] = {"status": "failed", "failure_reason": "not_attempted"}
    for attempt in range(1, int(settings["max_attempts"]) + 1):
        last = _once(parent, rng, settings)
        if last["status"] == "accepted":
            child = last.pop("_child")
            return structural_result(
                OPERATOR_NAME, parent, child, last,
                np.asarray(last["affected_indices"], dtype=int),
                status="accepted", failure_reason="", attempts=attempt, started=started,
            )
    return structural_result(
        OPERATOR_NAME, parent, None, {key: value for key, value in last.items() if key != "_child"},
        np.array([], dtype=int),
        status="failed", failure_reason=str(last.get("failure_reason", "")),
        attempts=int(settings["max_attempts"]), started=started,
    )
