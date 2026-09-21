"""Project-level validity checks and deterministic coordinate repair."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ValidityReport:
    valid: bool
    shape_ok: bool
    finite_ok: bool
    bounds_ok: bool
    spacing_ok: bool
    minimum_pairwise_distance: float | None
    duplicate_or_too_close_count: int


@dataclass(frozen=True)
class RepairOutcome:
    coordinates: np.ndarray
    indices: np.ndarray
    reasons: dict[str, list[int]]
    before: ValidityReport
    after: ValidityReport


def _pairwise_distances(coordinates: np.ndarray) -> np.ndarray:
    delta = coordinates[:, None, :] - coordinates[None, :, :]
    distances = np.sqrt(np.sum(delta * delta, axis=2))
    np.fill_diagonal(distances, np.inf)
    return distances


def check_validity(
    coordinates: np.ndarray,
    *,
    min_pair_distance: float,
) -> ValidityReport:
    """Check shape, finite values, unit-square bounds, and point spacing."""

    array = np.asarray(coordinates, dtype=np.float64)
    shape_ok = bool(array.ndim == 2 and array.shape[1:] == (2,) and array.shape[0] >= 2)
    finite_ok = bool(shape_ok and np.isfinite(array).all())
    bounds_ok = bool(
        finite_ok and np.all((array >= 0.0) & (array <= 1.0))
    )
    if not finite_ok:
        return ValidityReport(
            valid=False,
            shape_ok=shape_ok,
            finite_ok=finite_ok,
            bounds_ok=False,
            spacing_ok=False,
            minimum_pairwise_distance=None,
            duplicate_or_too_close_count=0,
        )
    distances = _pairwise_distances(array)
    minimum = float(np.min(distances))
    close_pairs = np.argwhere(np.triu(distances < min_pair_distance, k=1))
    spacing_ok = bool(close_pairs.size == 0)
    return ValidityReport(
        valid=bool(shape_ok and finite_ok and bounds_ok and spacing_ok),
        shape_ok=shape_ok,
        finite_ok=finite_ok,
        bounds_ok=bounds_ok,
        spacing_ok=spacing_ok,
        minimum_pairwise_distance=minimum,
        duplicate_or_too_close_count=int(close_pairs.shape[0]),
    )


def _bad_indices(
    coordinates: np.ndarray,
    *,
    min_pair_distance: float,
) -> tuple[np.ndarray, dict[str, list[int]]]:
    bounds = np.flatnonzero(
        ~np.isfinite(coordinates).all(axis=1)
        | np.any((coordinates < 0.0) | (coordinates > 1.0), axis=1)
    )
    safe = coordinates.copy()
    safe[~np.isfinite(safe)] = 0.0
    distances = _pairwise_distances(safe)
    close_pairs = np.argwhere(np.triu(distances < min_pair_distance, k=1))
    too_close = np.unique(close_pairs[:, 1]) if close_pairs.size else np.array([], dtype=int)
    combined = np.unique(np.concatenate([bounds, too_close])).astype(int)
    return combined, {
        "out_of_bounds_or_nonfinite": bounds.astype(int).tolist(),
        "duplicate_or_too_close": too_close.astype(int).tolist(),
    }


def repair_coordinates(
    coordinates: np.ndarray,
    *,
    repair_seed: int,
    min_pair_distance: float,
    max_passes: int,
) -> RepairOutcome:
    """Deterministically resample only invalid points using an explicit seed."""

    array = np.asarray(coordinates, dtype=np.float64).copy()
    before = check_validity(array, min_pair_distance=min_pair_distance)
    rng = np.random.default_rng(int(repair_seed))
    repaired: set[int] = set()
    reasons: dict[str, set[int]] = {
        "out_of_bounds_or_nonfinite": set(),
        "duplicate_or_too_close": set(),
    }
    for _ in range(int(max_passes)):
        bad, pass_reasons = _bad_indices(
            array,
            min_pair_distance=min_pair_distance,
        )
        if bad.size == 0:
            break
        for reason, indices in pass_reasons.items():
            reasons[reason].update(indices)
        array[bad] = rng.random((bad.size, 2))
        repaired.update(int(index) for index in bad)
    after = check_validity(array, min_pair_distance=min_pair_distance)
    return RepairOutcome(
        coordinates=array,
        indices=np.asarray(sorted(repaired), dtype=int),
        reasons={key: sorted(values) for key, values in reasons.items()},
        before=before,
        after=after,
    )
