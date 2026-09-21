"""Difficulty-blind deterministic structural GP0 probe candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class StructuralCandidate:
    instance_hash: str
    niche_id: int
    normalized_qd4: tuple[float, float, float, float]


@dataclass(frozen=True)
class ProbeSelection:
    instance_hash: str
    niche_id: int
    centroid_distance: float


def select_centroid_nearest_probes(
    candidates: Iterable[StructuralCandidate],
    centroids: np.ndarray,
) -> tuple[ProbeSelection, ...]:
    """Return one centroid-nearest unique candidate per occupied niche."""

    centers = np.asarray(centroids, dtype=np.float64)
    if centers.shape != (1000, 4):
        raise ValueError("centroids must have shape (1000,4)")
    seen: dict[str, StructuralCandidate] = {}
    winners: dict[int, ProbeSelection] = {}
    for candidate in candidates:
        prior = seen.get(candidate.instance_hash)
        if prior is not None:
            if prior != candidate:
                raise ValueError("duplicate hash has conflicting structural metadata")
            continue
        seen[candidate.instance_hash] = candidate
        niche = int(candidate.niche_id)
        if not 0 <= niche < 1000:
            raise ValueError(f"invalid niche {niche}")
        point = np.asarray(candidate.normalized_qd4, dtype=np.float64)
        if point.shape != (4,) or not np.isfinite(point).all():
            raise ValueError(f"invalid normalized QD4 for {candidate.instance_hash}")
        distance = float(np.linalg.norm(point - centers[niche]))
        proposal = ProbeSelection(candidate.instance_hash, niche, distance)
        incumbent = winners.get(niche)
        if incumbent is None or (distance, candidate.instance_hash) < (
            incumbent.centroid_distance,
            incumbent.instance_hash,
        ):
            winners[niche] = proposal
    return tuple(winners[niche] for niche in sorted(winners))

