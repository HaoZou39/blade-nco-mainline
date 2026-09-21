"""Sparse, one-elite-per-niche QD frontier memory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np


NICHES = 1000
MAX_ELITES_PER_NICHE = 1


@dataclass(frozen=True)
class CurrentSurrogateScore:
    """A score explicitly tied to the current solver/surrogate state."""

    value: float
    version: str


@dataclass(frozen=True)
class ArchiveEntry:
    instance_hash: str
    coordinate_hash: str
    niche_id: int
    normalized_qd4: tuple[float, float, float, float]
    current_surrogate_score: float
    score_version: str
    root_hash: str | None = None
    parent_hash: str | None = None
    operator_id: str | None = None
    cycle: int = 0


class SparseQDArchive:
    """Persistent structural frontier with zero or one elite per niche."""

    def __init__(self, entries: Iterable[ArchiveEntry] = ()) -> None:
        materialized = tuple(entries)
        self._by_niche = {entry.niche_id: entry for entry in materialized}
        if len(self._by_niche) != len(materialized):
            raise ValueError("SparseQDArchive permits at most one elite per niche")
        self.validate()

    @property
    def entries(self) -> tuple[ArchiveEntry, ...]:
        return tuple(self._by_niche[key] for key in sorted(self._by_niche))

    def __len__(self) -> int:
        return len(self._by_niche)

    def validate(self) -> None:
        if len(self._by_niche) > NICHES:
            raise ValueError("sparse archive exceeds K=1000")
        hashes: set[str] = set()
        for niche, entry in self._by_niche.items():
            if niche != entry.niche_id or not 0 <= niche < NICHES:
                raise ValueError(f"niche outside [0,{NICHES}): {niche}")
            if entry.instance_hash in hashes:
                raise ValueError("archive instance hashes must be unique")
            hashes.add(entry.instance_hash)
            descriptor = np.asarray(entry.normalized_qd4, dtype=np.float64)
            if descriptor.shape != (4,) or not np.isfinite(descriptor).all():
                raise ValueError(f"invalid QD4 descriptor for {entry.instance_hash}")
            if not np.isfinite(entry.current_surrogate_score):
                raise ValueError(f"invalid score for {entry.instance_hash}")
            if not entry.score_version:
                raise ValueError("score_version is required")


def update_sparse_qd_archive(
    incumbents: Iterable[ArchiveEntry],
    candidates: Iterable[ArchiveEntry],
    current_scores: Mapping[str, CurrentSurrogateScore],
    niche_ids: Mapping[str, int],
    *,
    current_score_version: str,
) -> SparseQDArchive:
    """Rescore an incumbent/candidate union and retain one winner per niche.

    Every participant must have an explicitly current score. This prevents an
    incumbent's score from a previous solver state from being reused.
    """

    if not current_score_version:
        raise ValueError("current_score_version is required")
    unique: dict[str, ArchiveEntry] = {}
    for entry in (*tuple(incumbents), *tuple(candidates)):
        prior = unique.get(entry.instance_hash)
        if prior is not None:
            if prior.coordinate_hash != entry.coordinate_hash:
                raise ValueError("duplicate instance hash has conflicting coordinates")
            continue
        unique[entry.instance_hash] = entry
    winners: dict[int, ArchiveEntry] = {}
    for instance_hash, entry in unique.items():
        if instance_hash not in current_scores:
            raise ValueError(f"missing current score for {instance_hash}")
        score = current_scores[instance_hash]
        if score.version != current_score_version:
            raise ValueError(f"stale score forbidden for {instance_hash}")
        if instance_hash not in niche_ids:
            raise ValueError(f"missing current niche assignment for {instance_hash}")
        niche = int(niche_ids[instance_hash])
        if not 0 <= niche < NICHES:
            raise ValueError(f"niche outside [0,{NICHES}): {niche}")
        refreshed = ArchiveEntry(
            instance_hash=entry.instance_hash,
            coordinate_hash=entry.coordinate_hash,
            niche_id=niche,
            normalized_qd4=entry.normalized_qd4,
            current_surrogate_score=float(score.value),
            score_version=current_score_version,
            root_hash=entry.root_hash,
            parent_hash=entry.parent_hash,
            operator_id=entry.operator_id,
            cycle=entry.cycle,
        )
        incumbent = winners.get(niche)
        key = (-refreshed.current_surrogate_score, refreshed.coordinate_hash)
        if incumbent is None or key < (-incumbent.current_surrogate_score, incumbent.coordinate_hash):
            winners[niche] = refreshed
    return SparseQDArchive(winners.values())

