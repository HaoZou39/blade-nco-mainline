"""Frozen acquisition policies.

The historical layered round-robin helper remains available for development
audit reproducibility.  Formal execution uses :func:`centroid_global_acquire`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from .archive import CurrentSurrogateScore, NICHES


@dataclass(frozen=True)
class AcquisitionSelection:
    instance_hash: str
    selection_rank_within_niche: int
    round_robin_layer: int
    niche_id: int
    surrogate_score: float


@dataclass(frozen=True)
class FinalAcquisitionSelection:
    instance_hash: str
    component: str
    niche_id: int
    centroid_distance: float
    surrogate_score: float


def centroid_global_acquire(
    candidates: Iterable[str],
    scores: Mapping[str, CurrentSurrogateScore],
    niche_ids: Mapping[str, int],
    normalized_qd4: Mapping[str, tuple[float, float, float, float]],
    centroids: np.ndarray,
    budget: int = 10_000,
    *,
    current_score_version: str,
) -> tuple[FinalAcquisitionSelection, ...]:
    """Centroid-one coverage followed by a global GP top-k fill.

    Coverage selection is deliberately independent of hardness.  Both phases
    use coordinate hash ascending as their deterministic final tie-break.
    """

    unique = tuple(dict.fromkeys(str(value) for value in candidates))
    if budget <= 0 or len(unique) < budget:
        raise ValueError(f"need at least {budget} unique candidates, got {len(unique)}")
    centers = np.asarray(centroids, dtype=np.float64)
    if centers.shape != (NICHES, 4):
        raise ValueError(f"centroids must have shape ({NICHES},4)")
    by_niche: dict[int, list[tuple[float, str]]] = {}
    checked: dict[str, tuple[int, float, float]] = {}
    for instance_hash in unique:
        if instance_hash not in niche_ids or instance_hash not in normalized_qd4 or instance_hash not in scores:
            raise ValueError(f"candidate metadata incomplete: {instance_hash}")
        niche = int(niche_ids[instance_hash])
        if not 0 <= niche < NICHES:
            raise ValueError(f"niche outside [0,{NICHES}): {niche}")
        vector = np.asarray(normalized_qd4[instance_hash], dtype=np.float64)
        score = scores[instance_hash]
        if vector.shape != (4,) or not np.isfinite(vector).all() or not np.isfinite(score.value):
            raise ValueError(f"non-finite candidate metadata: {instance_hash}")
        if score.version != current_score_version:
            raise ValueError(f"stale score forbidden for {instance_hash}")
        distance = float(np.linalg.norm(vector - centers[niche]))
        checked[instance_hash] = (niche, distance, float(score.value))
        by_niche.setdefault(niche, []).append((distance, instance_hash))

    floor_hashes: list[str] = []
    for niche in sorted(by_niche):
        by_niche[niche].sort(key=lambda item: (item[0], item[1]))
        floor_hashes.append(by_niche[niche][0][1])
    if len(floor_hashes) > budget:
        raise RuntimeError("centroid coverage floor exceeds active budget")
    floor_set = set(floor_hashes)
    fill_hashes = sorted(
        (item for item in unique if item not in floor_set),
        key=lambda item: (-checked[item][2], item),
    )[: budget - len(floor_hashes)]
    chosen = floor_hashes + fill_hashes
    if len(chosen) != budget or len(set(chosen)) != budget:
        raise RuntimeError("final acquisition did not produce the exact unique budget")
    output: list[FinalAcquisitionSelection] = []
    for instance_hash in chosen:
        niche, distance, value = checked[instance_hash]
        output.append(FinalAcquisitionSelection(
            instance_hash=instance_hash,
            component="centroid_floor" if instance_hash in floor_set else "gp_fill",
            niche_id=niche,
            centroid_distance=distance,
            surrogate_score=value,
        ))
    return tuple(output)


def qd_balanced_acquire(
    candidates: Iterable[str],
    scores: Mapping[str, CurrentSurrogateScore],
    niche_ids: Mapping[str, int],
    budget: int = 10_000,
    *,
    current_score_version: str,
) -> tuple[AcquisitionSelection, ...]:
    """Select exactly ``budget`` unique hashes using niche-rank layers."""

    if budget <= 0:
        raise ValueError("budget must be positive")
    unique = tuple(dict.fromkeys(candidates))
    if len(unique) < budget:
        raise ValueError(f"only {len(unique)} unique eligible candidates for budget {budget}")
    groups: dict[int, list[tuple[str, float]]] = {}
    for instance_hash in unique:
        if instance_hash not in scores:
            raise ValueError(f"missing score for {instance_hash}")
        score = scores[instance_hash]
        if score.version != current_score_version:
            raise ValueError(f"stale score forbidden for {instance_hash}")
        if not np.isfinite(score.value):
            raise ValueError(f"non-finite score for {instance_hash}")
        if instance_hash not in niche_ids:
            raise ValueError(f"missing niche for {instance_hash}")
        niche = int(niche_ids[instance_hash])
        if not 0 <= niche < NICHES:
            raise ValueError(f"niche outside [0,{NICHES}): {niche}")
        groups.setdefault(niche, []).append((instance_hash, float(score.value)))
    for group in groups.values():
        group.sort(key=lambda item: (-item[1], item[0]))

    selected: list[AcquisitionSelection] = []
    layer = 0
    ordered_niches = sorted(groups)
    while len(selected) < budget:
        progressed = False
        for niche in ordered_niches:
            group = groups[niche]
            if layer >= len(group):
                continue
            instance_hash, score = group[layer]
            selected.append(
                AcquisitionSelection(
                    instance_hash=instance_hash,
                    selection_rank_within_niche=layer + 1,
                    round_robin_layer=layer + 1,
                    niche_id=niche,
                    surrogate_score=score,
                )
            )
            progressed = True
            if len(selected) == budget:
                break
        if not progressed:
            raise RuntimeError("candidate groups exhausted before budget was filled")
        layer += 1
    if len({item.instance_hash for item in selected}) != budget:
        raise RuntimeError("acquisition duplicated an instance")
    return tuple(selected)
