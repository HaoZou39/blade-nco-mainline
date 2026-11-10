"""Versioned structural V2 / Motif V3.

One call builds one motif. k is an integer in [20, 40]. A failed draw is not
repaired. Up to three independent draws are attempted.
"""

from __future__ import annotations

from typing import Any, Mapping
from pathlib import Path
import time

import numpy as np

from .parameters import load_operator_config, resolve_operator_config, sample_integer, sample_scalar
from .structural_geometry import contain, match, spacing, structural_result, valid_child
from .structural_geometry_v2 import motif_sites_v3


OPERATOR_NAME = "motif_assembly_v3"


def _v2_settings(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    profile = load_operator_config(Path(__file__).with_name("unified12_structural_v2.yaml")) if config is None else config
    return resolve_operator_config(OPERATOR_NAME, profile)


def _once(coordinates: np.ndarray, rng: np.random.Generator, settings: Mapping[str, Any]) -> dict[str, Any]:
    kind = str(rng.choice(tuple(settings["motif"]["values"])))
    count = min(sample_integer(settings["affected_count"], rng), coordinates.shape[0])
    scale = sample_scalar(settings["scale"], rng)
    orientation = sample_scalar(settings["orientation_degrees"], rng)
    alpha = sample_scalar(settings["alpha"], rng)
    box = settings["center_box"]
    candidates = rng.uniform(float(box["low"]), float(box["high"]), size=(int(settings["center_candidates"]), 2))
    scores = []
    neighborhoods = []
    for candidate in candidates:
        distance = np.linalg.norm(coordinates - candidate, axis=1)
        nearest = np.argpartition(distance, count - 1)[:count]
        scores.append(float(distance[nearest].mean()))
        neighborhoods.append(nearest)
    choice = int(np.argmax(scores))
    selected = neighborhoods[choice]
    center = candidates[choice]
    parameters: dict[str, Any] = {
        "motif": kind,
        "k": count,
        "version": settings["version"],
        "scale": scale,
        "orientation_degrees": orientation,
        "alpha": alpha,
        "center": center.tolist(),
        "center_score": scores[choice],
        "status": "failed",
    }
    sites = motif_sites_v3(kind, count, center, scale, orientation, rng, settings)
    interior = tuple(settings["interior"])
    for _ in range(int(settings["shrink_passes"])):
        if np.all((sites >= interior[0]) & (sites <= interior[1])):
            break
        sites = center + (sites - center) * float(settings["shrink_factor"])
    sites = contain(sites, center, interior[0], interior[1])
    gap = spacing(sites)
    parameters["spacing"] = gap
    jitter_scale = float(rng.uniform(0.0, max(gap, 1e-6) / 4.0))
    sites = sites + rng.normal(0.0, jitter_scale, size=sites.shape)
    parameters["jitter_scale"] = jitter_scale
    matched = match(coordinates[selected], sites)
    child = coordinates.copy()
    child[selected] = (1.0 - alpha) * coordinates[selected] + alpha * matched
    reason = valid_child(child, node_count=int(settings["node_count"]), min_pair=float(settings["min_pair_distance"]))
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
    drawn = _once(np.asarray(coordinates, dtype=np.float64), rng, _v2_settings(config))
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
):
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
    settings = _v2_settings()
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
